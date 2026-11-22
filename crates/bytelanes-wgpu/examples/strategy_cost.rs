//! What moving bytes costs under each buffer strategy, on each hardware
//! adapter: the M4 "copy versus no-copy" measurement.
//!
//! For every strategy that applies to an adapter and every input size it
//! times four things:
//!
//! - `in`: [`BufferStrategy::upload`] of the input, then waiting until the
//!   GPU can use it.
//! - `out`: [`BufferStrategy::download`] of a result buffer the GPU has
//!   written.
//! - `checksum`: a whole `Checksum` program, host to host, through
//!   [`WgpuBackend`] (input in, 4 bytes and the CRC out).
//! - `decrypt`: a whole `Decrypt` program, host to host (input in, the same
//!   size out, then the CRC of the result).
//!
//! Every sample is kept. Run it in release mode, plugged in, with the lid
//! open:
//!
//! ```text
//! cargo run --release -p bytelanes-wgpu --example strategy_cost -- \
//!     --machine macbook --out run.json
//! ```
//!
//! The run file records the adapter, the OS, the Rust version and the
//! command. Fields wgpu cannot see (power source, power mode, host link) are
//! read with the `--probe` commands or passed with `--note key=value`.

use std::fmt::Write as _;
use std::process::Command;
use std::time::{Duration, Instant};

use bytelanes::abi::{BufId, ChannelCmd, ChannelProgram, Completion, Extent, Status, Tag};
use bytelanes::conformance::SplitMix64;
use bytelanes::{Buffers, ChannelBackend, verify};
use bytelanes_wgpu::{
    BufferStrategy, CopyingStrategy, DEFAULT_QUEUE_DEPTH, UnifiedMemoryStrategy, WgpuBackend,
    hardware_adapters, wgpu,
};

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;
const SIZES: [usize; 8] = [
    4 * KIB,
    64 * KIB,
    MIB,
    4 * MIB,
    16 * MIB,
    64 * MIB,
    256 * MIB,
    1024 * MIB,
];
const WARMUP: usize = 3;
const MIN_REPS: usize = 10;
const MAX_REPS: usize = 200;
const MIN_TIME: Duration = Duration::from_millis(1500);

const IN: BufId = BufId(1);
const OUT: BufId = BufId(2);
const KEY: u32 = 1;

fn strategy(name: &str, adapter: &wgpu::Adapter) -> Option<Box<dyn BufferStrategy>> {
    match name {
        "copying" => Some(Box::new(CopyingStrategy::new())),
        "unified" => UnifiedMemoryStrategy::for_adapter(adapter)
            .ok()
            .map(|s| Box::new(s) as Box<dyn BufferStrategy>),
        _ => None,
    }
}

/// Runs `f` until it has `MIN_REPS` samples and `MIN_TIME` has passed, or
/// `MAX_REPS` samples, after `WARMUP` untimed runs. Nanoseconds.
fn sample(mut f: impl FnMut()) -> Vec<u64> {
    for _ in 0..WARMUP {
        f();
    }
    let begin = Instant::now();
    let mut out = Vec::new();
    while out.len() < MAX_REPS && (out.len() < MIN_REPS || begin.elapsed() < MIN_TIME) {
        let t = Instant::now();
        f();
        out.push(t.elapsed().as_nanos() as u64);
    }
    out
}

fn wait(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("device poll");
}

/// One program through the backend: submit, then spin until it completes.
fn run(backend: &WgpuBackend, prog: &ChannelProgram, tag: u64) -> Completion {
    backend.submit(prog, Tag(tag)).expect("submit");
    let mut c = [Completion::error(Tag(0), 0, Status::DeviceError, 0)];
    loop {
        if backend.poll(&mut c) == 1 {
            return c[0];
        }
        std::thread::yield_now();
    }
}

struct Series {
    adapter: String,
    strategy: &'static str,
    what: &'static str,
    bytes: usize,
    ns: Vec<u64>,
}

fn measure(adapter: &wgpu::Adapter, name: &str, sizes: &[usize], out: &mut Vec<Series>) {
    let Some(s) = strategy(name, adapter) else {
        return;
    };
    let strategy_name = s.name();
    let info = adapter.get_info();
    let label = format!("{} ({:?})", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("strategy_cost"),
        required_features: s.required_features(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("open device");
    let backend = WgpuBackend::with_strategy(adapter, s, DEFAULT_QUEUE_DEPTH).expect("backend");
    // The strategy moved into the backend; this copy measures it directly.
    let s = strategy(name, adapter).expect("strategy");
    backend.set_key(KEY, [0x5A; 32]).expect("key");

    for &n in sizes {
        if (n as u64 + 32) > adapter.limits().max_storage_buffer_binding_size {
            eprintln!("{label} {strategy_name}: {n} bytes exceeds the binding limit, skipped");
            continue;
        }
        let data = SplitMix64::new(n as u64).bytes(n);
        let cap = (n as u64).next_multiple_of(16) + 16;
        let mut push = |what, ns: Vec<u64>| {
            let mut sorted = ns.clone();
            sorted.sort_unstable();
            eprintln!(
                "{label} {strategy_name:8} {what:8} {n:>11} B: median {:>10.1} us over {} runs",
                sorted[sorted.len() / 2] as f64 / 1e3,
                sorted.len()
            );
            out.push(Series {
                adapter: label.clone(),
                strategy: strategy_name,
                what,
                bytes: n,
                ns,
            });
        };

        // in: upload, then wait for the GPU to have it.
        push(
            "in",
            sample(|| {
                let b = s.upload(&device, &queue, &data, cap);
                queue.submit([]);
                wait(&device);
                drop(b);
            }),
        );

        // out: download of a buffer the GPU wrote.
        let src = s.upload(&device, &queue, &data, cap);
        let dst = s.alloc(&device, "result", cap);
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src, 0, &dst, 0, cap);
        queue.submit([enc.finish()]);
        wait(&device);
        let check = s
            .download(&device, &queue, &dst, 0, n as u64)
            .expect("download");
        assert!(check == data, "download returned other bytes");
        push(
            "out",
            sample(|| {
                let b = s
                    .download(&device, &queue, &dst, 0, n as u64)
                    .expect("download");
                assert_eq!(b.len(), n);
            }),
        );
        drop((src, dst));

        // Whole programs through the backend.
        backend.put_host(IN, data.clone()).expect("put");
        backend.put_host(OUT, vec![0; n]).expect("put");
        let nn = n as u64;
        let checksum = ChannelProgram::from_commands(&[ChannelCmd::checksum(
            Extent::host(IN, 0, nn),
            Extent::host(OUT, 0, 4),
        )])
        .expect("program");
        let decrypt = ChannelProgram::from_commands(&[ChannelCmd::decrypt(
            Extent::host(IN, 0, nn),
            Extent::host(OUT, 0, nn),
            KEY,
            [7; 16],
        )])
        .expect("program");
        for (what, prog, len) in [("checksum", checksum, 4), ("decrypt", decrypt, n)] {
            let c = run(&backend, &prog, 0);
            let result = backend.get_host(OUT).expect("out")[..len].to_vec();
            verify(&c, &result).expect("result verifies");
            let mut tag = 0;
            push(
                what,
                sample(|| {
                    tag += 1;
                    let c = run(&backend, &prog, tag);
                    assert_eq!(c.status(), Some(Status::Ok));
                }),
            );
        }
        backend.clear();
        backend.set_key(KEY, [0x5A; 32]).expect("key");
    }
}

/// Output of a command, trimmed, or why it could not run.
fn probe(cmd: &str) -> String {
    let mut parts = cmd.split_whitespace();
    let Some(prog) = parts.next() else {
        return String::new();
    };
    match Command::new(prog).args(parts).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        Err(e) => format!("unavailable: {e}"),
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path = None;
    let mut machine = String::from("unknown");
    let mut notes: Vec<(String, String)> = Vec::new();
    let mut probes: Vec<String> = Vec::new();
    let mut max = *SIZES.last().unwrap_or(&MIB);
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_default();
        match a.as_str() {
            "--out" => out_path = Some(val()),
            "--machine" => machine = val(),
            "--max-bytes" => max = val().parse().expect("--max-bytes takes a number"),
            "--note" => {
                let v = val();
                let (k, v) = v.split_once('=').expect("--note takes key=value");
                notes.push((k.to_owned(), v.to_owned()));
            }
            "--probe" => probes.push(val()),
            other => panic!("unknown argument {other}"),
        }
    }
    let sizes: Vec<usize> = SIZES.iter().copied().filter(|&s| s <= max).collect();

    let adapters = hardware_adapters();
    if adapters.is_empty() {
        eprintln!("no hardware GPU adapter found");
        std::process::exit(2);
    }
    let mut series = Vec::new();
    for a in &adapters {
        for name in ["copying", "unified"] {
            measure(a, name, &sizes, &mut series);
        }
    }

    let mut j = String::from("{\n");
    let _ = writeln!(j, "  \"kind\": \"buffer-strategy-cost\",");
    let _ = writeln!(j, "  \"milestone\": \"M4\",");
    let _ = writeln!(j, "  \"machine\": {},", json_str(&machine));
    let _ = writeln!(j, "  \"command\": {},", json_str(&args.join(" ")));
    let _ = writeln!(j, "  \"rust\": {},", json_str(&probe("rustc -V")));
    let _ = writeln!(
        j,
        "  \"profile\": {},",
        json_str(if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        })
    );
    let _ = writeln!(j, "  \"wgpu\": \"30.0.2\",");
    let _ = writeln!(j, "  \"adapters\": [");
    for (i, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        let _ = writeln!(
            j,
            "    {{\"name\": {}, \"backend\": {}, \"device_type\": {}, \"driver\": {}, \"driver_info\": {}, \"unified_strategy\": {}}}{}",
            json_str(&info.name),
            json_str(&format!("{:?}", info.backend)),
            json_str(&format!("{:?}", info.device_type)),
            json_str(&info.driver),
            json_str(&info.driver_info),
            json_str(&match UnifiedMemoryStrategy::for_adapter(a) {
                Ok(_) => "applies".to_owned(),
                Err(e) => e,
            }),
            if i + 1 < adapters.len() { "," } else { "" }
        );
    }
    let _ = writeln!(j, "  ],");
    let _ = writeln!(j, "  \"notes\": {{");
    for (i, (k, v)) in notes.iter().enumerate() {
        let comma = if i + 1 < notes.len() { "," } else { "" };
        let _ = writeln!(j, "    {}: {}{comma}", json_str(k), json_str(v));
    }
    let _ = writeln!(j, "  }},");
    let _ = writeln!(j, "  \"probes\": {{");
    for (i, p) in probes.iter().enumerate() {
        let comma = if i + 1 < probes.len() { "," } else { "" };
        let _ = writeln!(j, "    {}: {}{comma}", json_str(p), json_str(&probe(p)));
    }
    let _ = writeln!(j, "  }},");
    let _ = writeln!(
        j,
        "  \"method\": {},",
        json_str(&format!(
            "per series: {WARMUP} untimed runs, then timed runs until at least {MIN_REPS} samples and {} ms, at most {MAX_REPS} samples; wall clock (std::time::Instant) in ns; input bytes from SplitMix64(seed = size)",
            MIN_TIME.as_millis()
        ))
    );
    let _ = writeln!(j, "  \"series\": [");
    for (i, s) in series.iter().enumerate() {
        let mut sorted = s.ns.clone();
        sorted.sort_unstable();
        let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
        let samples: Vec<String> = s.ns.iter().map(u64::to_string).collect();
        let _ = writeln!(
            j,
            "    {{\"adapter\": {}, \"strategy\": {}, \"what\": {}, \"bytes\": {}, \"runs\": {}, \"min_ns\": {}, \"p10_ns\": {}, \"median_ns\": {}, \"p90_ns\": {}, \"max_ns\": {}, \"samples_ns\": [{}]}}{}",
            json_str(&s.adapter),
            json_str(s.strategy),
            json_str(s.what),
            s.bytes,
            sorted.len(),
            sorted[0],
            q(0.1),
            q(0.5),
            q(0.9),
            sorted[sorted.len() - 1],
            samples.join(", "),
            if i + 1 < series.len() { "," } else { "" }
        );
    }
    let _ = writeln!(j, "  ]\n}}");

    match out_path {
        Some(p) => std::fs::write(&p, j).expect("write run file"),
        None => print!("{j}"),
    }
}
