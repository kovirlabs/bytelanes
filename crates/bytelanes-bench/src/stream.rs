//! Throughput: many commit-log batches in flight at once.
//!
//! One repetition pushes `k` batches of `t` transactions through an engine:
//!
//! - **GPU:** every program of every batch is submitted, then all
//!   completions are collected, with the backend's overlap off or on.
//! - **CPU with `n` threads:** each thread owns a `CpuBackend`, loaded
//!   before timing with the shared inputs and the outputs of its batches
//!   (batch `b` goes to thread `b % n`), and runs them in order.
//!
//! Recorded per point: repetition time percentiles, transactions per
//! second at the median, and CPU time per transaction (process CPU time
//! over the series, all threads, minus the caller's polling).

use std::time::{Duration, Instant};

use bytelanes::abi::{
    BufId, ChannelCmd, ChannelProgram, CmpOp, Completion, Extent, FilterSpec, Status, Tag,
};
use bytelanes::cpu::CpuBackend;
use bytelanes::{Buffers, ChannelBackend};
use bytelanes_wgpu::WgpuBackend;
use cpu_time::{ProcessTime, ThreadTime};
use serde_json::{Value, json};

use crate::measure::Budget;
use crate::workloads::{self, FLAG_ABOVE_CENTS, KEY, KEY_SLOT, RECORD};

const IN: BufId = BufId(1);
const AMT: BufId = BufId(3);
const ENC_AMT: BufId = BufId(5);

fn flag(b: u64) -> BufId {
    BufId(1000 + b)
}
fn enc(b: u64) -> BufId {
    BufId(2000 + b)
}
fn vram(b: u64) -> BufId {
    BufId(3000 + b)
}

fn iv(b: u64) -> [u8; 16] {
    let mut iv = [0u8; 16];
    iv[8..].copy_from_slice(&b.to_be_bytes());
    iv
}

fn program(cmds: &[ChannelCmd]) -> ChannelProgram {
    ChannelProgram::from_commands(cmds).expect("benchmark programs are valid")
}

/// `k` batches of `t` transactions: shared inputs, per-batch outputs.
pub struct Stream {
    pub t: u64,
    pub k: u64,
    shared: Vec<(BufId, Vec<u8>)>,
    batches: Vec<Batch>,
}

/// One batch: its programs, its host outputs (id, len) and Vram (id, len).
struct Batch {
    progs: Vec<ChannelProgram>,
    host: Vec<(BufId, u64)>,
    vram: Vec<(BufId, u64)>,
}

impl Stream {
    /// Write path: per batch, [fraud filter] and [encrypt the records].
    pub fn commit(t: u64, k: u64) -> Self {
        let (records, amounts) = workloads::transactions(t, t);
        let n = t * RECORD;
        let batches = (0..k)
            .map(|b| {
                let progs = vec![
                    program(&[ChannelCmd::filter(
                        Extent::host(AMT, 0, t * 8),
                        Extent::host(flag(b), 0, t * 4),
                        FilterSpec::i64(CmpOp::Gt, FLAG_ABOVE_CENTS, 0),
                    )]),
                    program(&[ChannelCmd::encrypt(
                        Extent::host(IN, 0, n),
                        Extent::host(enc(b), 0, n),
                        KEY_SLOT,
                        iv(b),
                    )]),
                ];
                Batch {
                    progs,
                    host: vec![(flag(b), t * 4), (enc(b), n)],
                    vram: vec![],
                }
            })
            .collect();
        Stream {
            t,
            k,
            shared: vec![(IN, records), (AMT, amounts)],
            batches,
        }
    }

    /// Read path: per batch, [decrypt the amount column into device memory,
    /// chained into the fraud filter].
    pub fn audit(t: u64, k: u64) -> Self {
        let (_, amounts) = workloads::transactions(t, t);
        let mut column = vec![0u8; amounts.len()];
        bytelanes::cpu::ops::aes256_ctr(&KEY, &iv(0), &amounts, &mut column);
        let len = t * 8;
        let batches = (0..k)
            .map(|b| {
                let progs = vec![program(&[
                    ChannelCmd::decrypt(
                        Extent::host(ENC_AMT, 0, len),
                        Extent::vram(vram(b), 0, len),
                        KEY_SLOT,
                        iv(0),
                    )
                    .chained(),
                    ChannelCmd::filter(
                        Extent::vram(vram(b), 0, len),
                        Extent::host(flag(b), 0, t * 4),
                        FilterSpec::i64(CmpOp::Gt, FLAG_ABOVE_CENTS, 0),
                    ),
                ])];
                Batch {
                    progs,
                    host: vec![(flag(b), t * 4)],
                    vram: vec![(vram(b), len)],
                }
            })
            .collect();
        Stream {
            t,
            k,
            shared: vec![(ENC_AMT, column)],
            batches,
        }
    }

    pub fn transactions(&self) -> u64 {
        self.t * self.k
    }

    pub fn input_bytes(&self) -> u64 {
        self.shared.iter().map(|(_, d)| d.len() as u64).sum::<u64>() * self.k
    }

    /// Loads the shared inputs, the key and the buffers of `batches`.
    fn load<B: Buffers>(&self, b: &B, batches: impl Iterator<Item = u64>) {
        for (id, data) in &self.shared {
            b.put_host(*id, data.clone()).expect("put_host");
        }
        b.set_key(KEY_SLOT, KEY).expect("set_key");
        for i in batches {
            let batch = &self.batches[i as usize];
            for (id, len) in &batch.host {
                b.put_host(*id, vec![0; *len as usize]).expect("put_host");
            }
            for (id, len) in &batch.vram {
                b.alloc_vram(*id, *len).expect("alloc_vram");
            }
        }
    }

    /// The expected completion of every program of every batch, from the
    /// CPU reference.
    fn expected(&self) -> Vec<Vec<(u32, u32)>> {
        let cpu = CpuBackend::new();
        self.load(&cpu, 0..self.k);
        self.batches
            .iter()
            .map(|batch| {
                batch
                    .progs
                    .iter()
                    .map(|p| {
                        let c = cpu.run(p, Tag(0));
                        assert_eq!(c.status(), Some(Status::Ok), "reference failed: {c:?}");
                        (c.bytes(), c.crc())
                    })
                    .collect()
            })
            .collect()
    }
}

fn check(c: &Completion, want: (u32, u32)) -> Result<(), String> {
    if c.status() == Some(Status::Ok) && (c.bytes(), c.crc()) == want {
        Ok(())
    } else {
        Err(format!("completion {c:?}, expected bytes/crc {want:?}"))
    }
}

/// An engine for the stream benchmark.
pub enum StreamEngine<'a> {
    Cpu(usize),
    Gpu {
        label: String,
        backend: &'a WgpuBackend,
        overlap: bool,
    },
}

impl StreamEngine<'_> {
    pub fn label(&self) -> String {
        match self {
            StreamEngine::Cpu(n) => format!("cpu-{n}t"),
            StreamEngine::Gpu { label, overlap, .. } => {
                format!("{label}-{}", if *overlap { "overlap" } else { "serial" })
            }
        }
    }
}

struct Series {
    reps: Vec<Duration>,
    cpu_process: Duration,
    cpu_wait: Duration,
}

fn gpu_rep(b: &WgpuBackend, s: &Stream, want: &[Vec<(u32, u32)>]) -> Result<Duration, String> {
    // Tag = batch * 16 + program index.
    let mut total = 0;
    for (i, batch) in s.batches.iter().enumerate() {
        for (j, p) in batch.progs.iter().enumerate() {
            b.submit(p, Tag((i * 16 + j) as u64)).map_err(|_| "busy")?;
            total += 1;
        }
    }
    let tw = ThreadTime::now();
    let mut slot = [Completion::ok(Tag(0), 0, 0, 0)];
    for _ in 0..total {
        while b.poll(&mut slot) == 0 {
            std::hint::spin_loop();
        }
        let tag = slot[0].tag().0 as usize;
        check(&slot[0], want[tag / 16][tag % 16])?;
    }
    Ok(tw.elapsed())
}

fn cpu_rep(backends: &[CpuBackend], s: &Stream, want: &[Vec<(u32, u32)>]) -> Result<(), String> {
    let n = backends.len();
    std::thread::scope(|scope| {
        let handles: Vec<_> = backends
            .iter()
            .enumerate()
            .map(|(th, cpu)| {
                scope.spawn(move || -> Result<(), String> {
                    for b in (th..s.k as usize).step_by(n) {
                        for (j, p) in s.batches[b].progs.iter().enumerate() {
                            check(&cpu.run(p, Tag(0)), want[b][j])?;
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().map_err(|_| "worker panicked".to_owned())?)
    })
}

/// Measures `s` on `e` and returns the engine's JSON.
pub fn measure(e: &StreamEngine<'_>, s: &Stream, budget: Budget) -> Result<Value, String> {
    let want = s.expected();
    let mut series = Series {
        reps: Vec::new(),
        cpu_process: Duration::ZERO,
        cpu_wait: Duration::ZERO,
    };
    let enough = |r: &Series, start: Instant| {
        r.reps.len() >= budget.max_reps
            || (r.reps.len() >= budget.min_reps && start.elapsed() >= budget.min_time)
    };
    match e {
        StreamEngine::Gpu {
            backend, overlap, ..
        } => {
            backend.clear();
            backend.set_overlap(*overlap);
            s.load(*backend, 0..s.k);
            for _ in 0..budget.warmup {
                gpu_rep(backend, s, &want)?;
            }
            let p0 = ProcessTime::now();
            let start = Instant::now();
            while !enough(&series, start) {
                let t = Instant::now();
                series.cpu_wait += gpu_rep(backend, s, &want)?;
                series.reps.push(t.elapsed());
            }
            series.cpu_process = p0.elapsed();
            backend.clear();
            backend.set_overlap(true);
        }
        StreamEngine::Cpu(n) => {
            let backends: Vec<CpuBackend> = (0..*n)
                .map(|th| {
                    let cpu = CpuBackend::new();
                    s.load(&cpu, (th as u64..s.k).step_by(*n));
                    cpu
                })
                .collect();
            for _ in 0..budget.warmup {
                cpu_rep(&backends, s, &want)?;
            }
            let p0 = ProcessTime::now();
            let start = Instant::now();
            while !enough(&series, start) {
                let t = Instant::now();
                cpu_rep(&backends, s, &want)?;
                series.reps.push(t.elapsed());
            }
            series.cpu_process = p0.elapsed();
        }
    }
    let mut v = series.reps.clone();
    v.sort_unstable();
    let q = |q: f64| v[((v.len() as f64 * q).ceil() as usize).clamp(1, v.len()) - 1];
    let ns = |d: Duration| d.as_nanos() as u64;
    let p50 = q(0.5);
    let txns = s.transactions() as f64;
    let reps = v.len() as f64;
    let work = series.cpu_process.saturating_sub(series.cpu_wait);
    Ok(json!({
        "reps": v.len(),
        "rep_ns": { "min": ns(v[0]), "p50": ns(p50), "p90": ns(q(0.9)), "max": ns(v[v.len() - 1]) },
        "transactions_per_s_at_p50": txns / p50.as_secs_f64(),
        "input_gb_per_s_at_p50": s.input_bytes() as f64 / p50.as_secs_f64() / 1e9,
        "cpu_process_ns": ns(series.cpu_process),
        "cpu_caller_waiting_ns": ns(series.cpu_wait),
        "cpu_work_ns_per_transaction": work.as_nanos() as f64 / (reps * txns),
    }))
}
