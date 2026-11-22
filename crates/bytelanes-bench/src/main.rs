//! bytelanes-bench: GPU versus CPU, measured through the same
//! `ChannelBackend` API, so GPU times include every transfer.
//!
//! ```text
//! bytelanes-bench adapters
//! bytelanes-bench ops       --machine ryzen --host-link "PCIe 3.0 x16" [options]
//! bytelanes-bench commitlog --machine ryzen --host-link "PCIe 3.0 x16" [options]
//! bytelanes-bench stream    --machine ryzen --host-link "PCIe 3.0 x16" [options]
//! ```
//!
//! `ops` writes one run file per op (`<date>-<op>.json`); `commitlog` writes
//! `<date>-commitlog.json`; `stream` (throughput: many commit-log batches in
//! flight, GPU with overlap off and on, CPU on 1..N threads) writes
//! `<date>-stream.json`. All go to `--out`, by default
//! `../paralanes/atlas/runs/<machine>/`. An existing file is never
//! overwritten: the new run gets a `-2`, `-3`, ... suffix.
//!
//! Options: `--machine-desc <text>`, `--note <text>`, `--min-time <s>`
//! (default 1), `--max-size <bytes>` (ops, default 256 MiB), `--max-txns
//! <n>` (commitlog, default 1048576), `--ops a,b`, `--out <dir>`; for
//! `stream`: `--txns 4096,16384,...` (per batch), `--batches <k>` (default
//! 16), `--cpu-threads 1,8,16` (default 1, half and all logical CPUs). Set
//! `WGPU_BACKEND=vulkan` or `=dx12` to measure one API only.

#![forbid(unsafe_code)]

mod measure;
mod meta;
mod stream;
mod workloads;

use std::path::PathBuf;
use std::time::Duration;

use bytelanes::abi::Status;
use bytelanes::cpu::CpuBackend;
use bytelanes::{Buffers, ChannelBackend};
use bytelanes_wgpu::{WgpuBackend, hardware_adapters};
use serde_json::{Value, json};

use measure::{Budget, Expected, Series};
use workloads::Case;

struct Args {
    mode: String,
    machine: String,
    machine_desc: String,
    host_link: String,
    note: String,
    out: Option<PathBuf>,
    min_time: f64,
    max_size: u64,
    max_txns: u64,
    ops: Vec<String>,
    /// Explicit sizes in bytes (ops); replaces the 64 B x 4^n series.
    sizes: Option<Vec<u64>>,
    /// Stream: transactions per batch, batches in flight, CPU thread counts.
    stream_txns: Vec<u64>,
    batches: u64,
    cpu_threads: Vec<usize>,
}

fn usage() -> ! {
    eprintln!(
        "usage: bytelanes-bench <adapters|ops|commitlog|stream> --machine <name> --host-link <text> \
         [--machine-desc <text>] [--note <text>] [--out <dir>] [--min-time <s>] \
         [--max-size <bytes>] [--sizes 16M,24M,...] [--max-txns <n>] [--ops checksum,encrypt,decrypt,filter] [--txns 4096,...] [--batches <k>] [--cpu-threads 1,8,16]"
    );
    std::process::exit(2)
}

fn parse() -> Args {
    let mut it = std::env::args().skip(1);
    let mode = it.next().unwrap_or_else(|| usage());
    let mut a = Args {
        mode,
        machine: String::new(),
        machine_desc: String::new(),
        host_link: String::new(),
        note: String::new(),
        out: None,
        min_time: 1.0,
        max_size: 256 << 20,
        max_txns: 1 << 20,
        ops: workloads::OPS.iter().map(|s| (*s).to_owned()).collect(),
        sizes: None,
        stream_txns: vec![4096, 16384, 65536, 262_144],
        batches: 16,
        cpu_threads: {
            let all = std::thread::available_parallelism().map_or(1, usize::from);
            let mut v = vec![1, (all / 2).max(1), all];
            v.dedup();
            v
        },
    };
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--machine" => a.machine = val(),
            "--machine-desc" => a.machine_desc = val(),
            "--host-link" => a.host_link = val(),
            "--note" => a.note = val(),
            "--out" => a.out = Some(val().into()),
            "--min-time" => a.min_time = val().parse().unwrap_or_else(|_| usage()),
            "--max-size" => a.max_size = val().parse().unwrap_or_else(|_| usage()),
            "--max-txns" => a.max_txns = val().parse().unwrap_or_else(|_| usage()),
            "--ops" => a.ops = val().split(',').map(str::to_owned).collect(),
            "--sizes" => {
                a.sizes = Some(
                    val()
                        .split(',')
                        .map(|s| parse_size(s).unwrap_or_else(|| usage()))
                        .collect(),
                )
            }
            "--txns" => {
                a.stream_txns = val()
                    .split(',')
                    .map(|s| s.parse().unwrap_or_else(|_| usage()))
                    .collect()
            }
            "--batches" => a.batches = val().parse().unwrap_or_else(|_| usage()),
            "--cpu-threads" => {
                a.cpu_threads = val()
                    .split(',')
                    .map(|s| s.parse().unwrap_or_else(|_| usage()))
                    .collect()
            }
            _ => usage(),
        }
    }
    a
}

/// A backend under test.
enum Engine {
    Cpu,
    Gpu(String, Box<WgpuBackend>),
}

impl Engine {
    fn label(&self) -> &str {
        match self {
            Engine::Cpu => "cpu",
            Engine::Gpu(l, _) => l,
        }
    }
}

fn load<B: Buffers>(b: &B, case: &Case) -> Result<(), Status> {
    for (id, data) in &case.host {
        b.put_host(*id, data.clone())?;
    }
    for (id, len) in &case.vram {
        b.alloc_vram(*id, *len)?;
    }
    b.set_key(workloads::KEY_SLOT, workloads::KEY)
}

fn measure_on<B: ChannelBackend + Buffers>(
    b: &B,
    case: &Case,
    want: &[Expected],
    budget: Budget,
) -> Result<Series, String> {
    load(b, case).map_err(|s| format!("loading buffers: {s:?}"))?;
    measure::run(b, &case.programs, want, budget)
}

/// The CPU reference's completions: what every engine must reproduce.
fn expected(case: &Case) -> Vec<Expected> {
    let cpu = CpuBackend::new();
    load(&cpu, case).expect("CPU backend accepts the buffers");
    case.programs
        .iter()
        .map(|p| {
            let c = cpu.run(p, bytelanes::abi::Tag(0));
            assert_eq!(c.status(), Some(Status::Ok), "reference failed: {c:?}");
            Expected {
                bytes: c.bytes(),
                crc: c.crc(),
            }
        })
        .collect()
}

fn ms(d: Duration) -> String {
    format!("{:.3}", d.as_secs_f64() * 1e3)
}

/// Measures `case` on every engine and returns the point's JSON.
fn point(engines: &[Engine], case: &Case, budget: Budget, label: &str) -> Value {
    let want = expected(case);
    let mut results = serde_json::Map::new();
    let mut line = format!("{label:>12}");
    for e in engines {
        let series = match e {
            Engine::Cpu => measure_on(&CpuBackend::new(), case, &want, budget),
            Engine::Gpu(_, b) => {
                b.clear();
                let s = measure_on(b.as_ref(), case, &want, budget);
                b.clear();
                s
            }
        };
        let name = e.label();
        let series = series.unwrap_or_else(|err| panic!("{label} on {name}: {err}"));
        let mut j = series.to_json(case.input_bytes);
        let per_unit = series.cpu_work_per_rep().as_nanos() as f64 / case.units as f64;
        let units_per_s = case.units as f64 / series.p50().as_secs_f64();
        j["cpu_work_ns_per_unit"] = json!(per_unit);
        j["units_per_s_at_p50"] = json!(units_per_s);
        line += &format!(
            " | {:>6} p50 {:>9} ms, cpu {:>9} ms",
            e.label(),
            ms(series.p50()),
            ms(series.cpu_work_per_rep())
        );
        results.insert(e.label().to_owned(), j);
    }
    eprintln!("{line}");
    json!({
        "input_bytes": case.input_bytes,
        "units": case.units,
        "programs": case.programs.len(),
        "expected": want
            .iter()
            .map(|w| json!({ "bytes": w.bytes, "crc": format!("{:08x}", w.crc) }))
            .collect::<Vec<_>>(),
        "engines": results,
    })
}

/// The smallest `key` from which `wins` holds at that point and every
/// larger one, or null.
fn from_here_on(points: &[Value], key: &str, wins: impl Fn(&Value) -> bool) -> Value {
    let mut answer = Value::Null;
    for p in points.iter().rev() {
        if wins(p) {
            answer = p[key].clone();
        } else {
            break;
        }
    }
    answer
}

fn summary(points: &[Value], key: &str, gpus: &[String]) -> Value {
    let p50 = |p: &Value, e: &str| p["engines"][e]["latency_ns"]["p50"].as_u64().unwrap_or(0);
    let work = |p: &Value, e: &str| p["engines"][e]["cpu_work_ns_per_rep"].as_u64().unwrap_or(0);
    let mut out = serde_json::Map::new();
    for g in gpus {
        out.insert(
            g.clone(),
            json!({
                "latency_crossover": from_here_on(points, key, |p| p50(p, g) < p50(p, "cpu")),
                "cpu_time_crossover": from_here_on(points, key, |p| work(p, g) < work(p, "cpu")),
                "definition": format!(
                    "latency_crossover: smallest {key} from which the GPU's median latency is \
                     below the CPU reference's at every larger measured {key}. \
                     cpu_time_crossover: the same for CPU time spent on the work. \
                     null: the GPU does not win from any measured {key} upward."
                ),
            }),
        );
    }
    Value::Object(out)
}

fn run_file(a: &Args, kind: &str, engines: &[Engine], budget: Budget) -> Value {
    let gpus: Vec<Value> = engines
        .iter()
        .filter_map(|e| match e {
            Engine::Gpu(_, b) => Some(meta::adapter(b.adapter_info(), b.strategy_name())),
            Engine::Cpu => None,
        })
        .collect();
    json!({
        "kind": kind,
        "milestone": "M5",
        "date": meta::utc_date(),
        "started_utc": meta::utc_time(),
        "machine": { "name": a.machine, "description": a.machine_desc },
        "os": std::env::consts::OS,
        "os_build": meta::os_build(),
        "rust": format!("{} (release profile)", meta::rustc()),
        "wgpu": meta::wgpu(),
        "gpus": gpus,
        "gpu_host_link": a.host_link,
        "network": "n/a: single machine",
        "commits": meta::git(),
        "command": meta::command(),
        "note": a.note,
        "method": {
            "api": "every program goes through ChannelBackend submit and poll, one program in flight; GPU times include upload, download and the result CRC",
            "cpu_reference": "bytelanes CpuBackend on the calling thread: crc32c and RustCrypto aes/ctr with hardware CRC and AES instructions; one core",
            "warmup_reps": budget.warmup,
            "min_reps": budget.min_reps,
            "min_time_s": budget.min_time.as_secs_f64(),
            "verification": "every repetition's completion (status, bytes, CRC-32C of the result) must equal the CPU reference's, or the run aborts",
            "cpu_work": "process CPU time (all threads) minus the caller's CPU time spinning in poll, per repetition. Windows updates these counters at scheduler ticks (about 15.6 ms); series run at least min_time_s to bound that error",
            "data": "generated from fixed SplitMix64 seeds",
        },
    })
}

fn out_dir(a: &Args) -> PathBuf {
    a.out
        .clone()
        .unwrap_or_else(|| PathBuf::from("../paralanes/atlas/runs").join(&a.machine))
}

/// Writes `v` to `<dir>/<date>-<name>.json`, adding `-2`, `-3`... rather
/// than overwriting an existing run file.
fn write(dir: &std::path::Path, name: &str, v: &Value) -> PathBuf {
    std::fs::create_dir_all(dir).expect("create output directory");
    let date = v["date"].as_str().unwrap_or("unknown");
    let mut path = dir.join(format!("{date}-{name}.json"));
    let mut k = 2;
    while path.exists() {
        path = dir.join(format!("{date}-{name}-{k}.json"));
        k += 1;
    }
    let text = serde_json::to_string_pretty(v).expect("serialise") + "\n";
    std::fs::write(&path, text).expect("write run file");
    eprintln!("wrote {}", path.display());
    path
}

fn engines() -> Vec<Engine> {
    let mut v = vec![Engine::Cpu];
    for a in hardware_adapters() {
        let label = format!("{:?}", a.get_info().backend).to_lowercase();
        let b = WgpuBackend::new(&a).unwrap_or_else(|e| panic!("open {label}: {e}"));
        v.push(Engine::Gpu(label, Box::new(b)));
    }
    v
}

/// `123`, `64K`, `16M` or `1G` as bytes.
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mul) = match s.chars().last()? {
        'K' | 'k' => (&s[..s.len() - 1], 1u64 << 10),
        'M' | 'm' => (&s[..s.len() - 1], 1 << 20),
        'G' | 'g' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    num.parse::<u64>().ok()?.checked_mul(mul)
}

fn sizes(max: u64) -> Vec<u64> {
    std::iter::successors(Some(64u64), |s| Some(s * 4))
        .take_while(|&s| s <= max)
        .collect()
}

fn main() {
    let a = parse();
    if a.mode == "adapters" {
        for e in engines() {
            if let Engine::Gpu(_, b) = e {
                println!("{}", meta::adapter(b.adapter_info(), b.strategy_name()));
            }
        }
        return;
    }
    if cfg!(debug_assertions) {
        eprintln!("bytelanes-bench: build with --release; debug timings are not results");
        std::process::exit(2);
    }
    if a.machine.is_empty() || a.host_link.is_empty() {
        usage();
    }
    let budget = Budget {
        min_time: Duration::from_secs_f64(a.min_time),
        min_reps: 5,
        max_reps: 200_000,
        warmup: 3,
    };
    let engines = engines();
    let gpus: Vec<String> = engines
        .iter()
        .skip(1)
        .map(|e| e.label().to_owned())
        .collect();
    eprintln!(
        "engines: {:?}",
        engines.iter().map(Engine::label).collect::<Vec<_>>()
    );
    let dir = out_dir(&a);

    match a.mode.as_str() {
        "ops" => {
            for op in &a.ops {
                let mut file = run_file(&a, "bytelanes-op", &engines, budget);
                file["op"] = json!(op);
                file["purpose"] = json!(format!(
                    "M5 atlas: CPU-versus-GPU crossover input size for {op}"
                ));
                file["workload"] = match op.as_str() {
                    "filter" => json!(
                        "u32 column, Gt threshold u32::MAX - u32::MAX/100 (about 1% of rows match), indices to host"
                    ),
                    "fft" => json!(format!(
                        "fixed-point FFT of {}-sample frames of generated full-scale i32 samples (clamped to +-2^30 by the op), bins 0..=N/2 to host; sizes below one frame are skipped",
                        1u32 << workloads::FFT_LOG2N
                    )),
                    _ => json!("generated bytes, host buffer to host buffer"),
                };
                eprintln!("== {op}");
                let series = a.sizes.clone().unwrap_or_else(|| sizes(a.max_size));
                let points: Vec<Value> = series
                    .into_iter()
                    .filter(|&n| workloads::op_fits(op, n))
                    .map(|n| {
                        let mut p = point(
                            &engines,
                            &workloads::op_case(op, n),
                            budget,
                            &format!("{n} B"),
                        );
                        p["size_bytes"] = json!(n);
                        p
                    })
                    .collect();
                file["summary"] = summary(&points, "size_bytes", &gpus);
                file["points"] = Value::Array(points);
                write(&dir, op, &file);
            }
        }
        "commitlog" => {
            let mut file = run_file(&a, "bytelanes-commitlog", &engines, budget);
            file["purpose"] = json!(
                "M5: a transactional workload. Batches of generated payment records; per batch, the write path (fraud-rule filter on the amount column, then encrypt the records for the log, whose result CRC is the segment CRC) and the read path (decrypt an encrypted amount column into device memory and filter it there, one chained program)"
            );
            file["workload"] = json!({
                "record_bytes": workloads::RECORD,
                "record_layout": "txn id u64, account u64, amount i64 cents, timestamp u64, merchant u32, currency u16, flags u16, 216 bytes payload",
                "fraud_rule": format!("amount > {} cents; about 1% of rows", workloads::FLAG_ABOVE_CENTS),
                "commit_programs": "two programs submitted together, both in flight: [Filter i64 Gt (amount column, host to host)] and [Encrypt AES-256-CTR (records, host to host)]",
                "audit_program": ["Decrypt AES-256-CTR (encrypted amount column, host to vram, chained)", "Filter i64 Gt (vram to host)"],
                "units": "transactions",
            });
            let txns: Vec<u64> = std::iter::successors(Some(1u64), |t| Some(t * 4))
                .take_while(|&t| t <= a.max_txns)
                .collect();
            for (name, make) in [
                ("commit", workloads::commit_case as fn(u64) -> Case),
                ("audit", workloads::audit_case),
            ] {
                eprintln!("== {name}");
                let points: Vec<Value> = txns
                    .iter()
                    .map(|&t| {
                        let mut p = point(&engines, &make(t), budget, &format!("{t} txn"));
                        p["transactions"] = json!(t);
                        p
                    })
                    .collect();
                file[name] = json!({
                    "summary": summary(&points, "transactions", &gpus),
                    "points": points,
                });
            }
            write(&dir, "commitlog", &file);
        }
        "stream" => {
            let mut file = run_file(&a, "bytelanes-stream", &engines, budget);
            file["purpose"] = json!(
                "M5: throughput with many commit-log batches in flight. GPU with program overlap off (serial) and on; CPU reference on 1, half and all logical CPUs (one CpuBackend per thread, batches spread round-robin)"
            );
            file["workload"] = json!({
                "batches_in_flight": a.batches,
                "commit": "per batch: [Filter i64 Gt on the amount column] and [Encrypt AES-256-CTR of the records], both host to host; every batch reads the same generated records with its own IV",
                "audit": "per batch: [Decrypt the encrypted amount column into device memory, chained into Filter i64 Gt, indices to host]",
                "cpu_threads": a.cpu_threads,
                "units": "transactions",
            });
            file["method_stream"] = json!(
                "one repetition = all batches submitted (GPU) or run across the threads (CPU), then every completion collected and checked against the CPU reference; transactions_per_s uses the median repetition; cpu_work_ns_per_transaction = process CPU time (all threads) minus the caller's polling, over the series"
            );
            let mut list: Vec<stream::StreamEngine<'_>> = a
                .cpu_threads
                .iter()
                .map(|&n| stream::StreamEngine::Cpu(n))
                .collect();
            for e in &engines {
                if let Engine::Gpu(label, b) = e {
                    for overlap in [false, true] {
                        list.push(stream::StreamEngine::Gpu {
                            label: label.clone(),
                            backend: b,
                            overlap,
                        });
                    }
                }
            }
            for (name, make) in [
                (
                    "commit",
                    stream::Stream::commit as fn(u64, u64) -> stream::Stream,
                ),
                ("audit", stream::Stream::audit),
            ] {
                eprintln!("== {name}");
                let mut points = Vec::new();
                for &t in &a.stream_txns {
                    let s = make(t, a.batches);
                    let mut results = serde_json::Map::new();
                    let mut line = format!("{t:>8} txn x {}", a.batches);
                    for e in &list {
                        let label = e.label();
                        let j = stream::measure(e, &s, budget)
                            .unwrap_or_else(|err| panic!("{name} {t} on {label}: {err}"));
                        line += &format!(
                            " | {label} {:.2} M txn/s, {:.0} ns cpu/txn",
                            j["transactions_per_s_at_p50"].as_f64().unwrap_or(0.0) / 1e6,
                            j["cpu_work_ns_per_transaction"].as_f64().unwrap_or(0.0)
                        );
                        results.insert(label, j);
                    }
                    eprintln!("{line}");
                    points.push(json!({
                        "transactions_per_batch": t,
                        "batches": a.batches,
                        "engines": results,
                    }));
                }
                file[name] = Value::Array(points);
            }
            write(&dir, "stream", &file);
        }
        _ => usage(),
    }
}
