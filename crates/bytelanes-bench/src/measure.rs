//! Timing one program repeatedly on one backend.
//!
//! Each repetition submits the program and polls until it completes, with
//! one program in flight. Per repetition we record wall-clock latency. Over
//! the whole series we record the process's CPU time (every thread: the
//! caller, the backend's worker, driver threads in this process) and the
//! caller's CPU time spent spinning in `poll` while waiting. `cpu_work` is
//! the first minus the second: the CPU the work itself cost.
//!
//! Windows updates thread and process CPU times at scheduler ticks (about
//! 15.6 ms), so each series runs for at least `min_time` (1 s by default)
//! to keep that error near 1-2%.

use std::time::{Duration, Instant};

use bytelanes::abi::{ChannelProgram, Completion, Status, Tag};
use bytelanes::{Buffers, ChannelBackend};
use cpu_time::{ProcessTime, ThreadTime};
use serde_json::{Value, json};

/// How long and how often to repeat.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub min_time: Duration,
    pub min_reps: usize,
    pub max_reps: usize,
    pub warmup: usize,
}

/// The completion every repetition must reproduce.
#[derive(Clone, Copy, Debug)]
pub struct Expected {
    pub bytes: u32,
    pub crc: u32,
}

/// One series on one backend.
#[derive(Debug)]
pub struct Series {
    pub latency: Vec<Duration>,
    pub wall: Duration,
    pub cpu_process: Duration,
    pub cpu_wait: Duration,
}

impl Series {
    pub fn reps(&self) -> usize {
        self.latency.len()
    }

    fn sorted(&self) -> Vec<Duration> {
        let mut v = self.latency.clone();
        v.sort_unstable();
        v
    }

    /// The `q` quantile (0..=1) of latency, nearest rank.
    pub fn quantile(&self, q: f64) -> Duration {
        let v = self.sorted();
        let i = ((v.len() as f64 * q).ceil() as usize).clamp(1, v.len()) - 1;
        v[i]
    }

    pub fn p50(&self) -> Duration {
        self.quantile(0.5)
    }

    /// CPU time per repetition spent on the work (process CPU minus the
    /// caller's polling).
    pub fn cpu_work_per_rep(&self) -> Duration {
        self.cpu_process.saturating_sub(self.cpu_wait) / self.reps().max(1) as u32
    }

    pub fn to_json(&self, bytes_per_rep: u64) -> Value {
        let v = self.sorted();
        let mean = self.latency.iter().sum::<Duration>() / self.reps().max(1) as u32;
        let ns = |d: Duration| d.as_nanos() as u64;
        let p50 = self.p50();
        let gbps = |d: Duration| {
            if d.is_zero() {
                0.0
            } else {
                bytes_per_rep as f64 / d.as_secs_f64() / 1e9
            }
        };
        json!({
            "reps": self.reps(),
            "latency_ns": {
                "min": ns(v[0]),
                "p50": ns(p50),
                "p90": ns(self.quantile(0.90)),
                "p99": ns(self.quantile(0.99)),
                "max": ns(v[v.len() - 1]),
                "mean": ns(mean),
            },
            "throughput_gb_per_s_at_p50": gbps(p50),
            "wall_ns": ns(self.wall),
            "cpu_process_ns": ns(self.cpu_process),
            "cpu_caller_waiting_ns": ns(self.cpu_wait),
            "cpu_work_ns_per_rep": ns(self.cpu_work_per_rep()),
            "cpu_work_ns_per_byte": if bytes_per_rep == 0 { 0.0 } else {
                self.cpu_work_per_rep().as_nanos() as f64 / bytes_per_rep as f64
            },
        })
    }
}

fn wait_one<B: ChannelBackend>(b: &B, slot: &mut [Completion; 1]) {
    while b.poll(slot) == 0 {
        std::hint::spin_loop();
    }
}

fn check(c: &Completion, want: Expected) -> Result<(), String> {
    match c.status() {
        Some(Status::Ok) if c.bytes() == want.bytes && c.crc() == want.crc => Ok(()),
        s => Err(format!(
            "completion {:?} bytes {} crc {:08x}, expected Ok bytes {} crc {:08x}",
            s,
            c.bytes(),
            c.crc(),
            want.bytes,
            want.crc
        )),
    }
}

/// One repetition: submit every program (tag = its index), then wait for
/// all of their completions, checking each against its expectation.
/// Returns the caller's CPU time spent waiting.
fn once<B: ChannelBackend>(
    b: &B,
    progs: &[ChannelProgram],
    want: &[Expected],
    slot: &mut [Completion; 1],
) -> Result<Duration, String> {
    for (i, p) in progs.iter().enumerate() {
        b.submit(p, Tag(i as u64)).map_err(|_| "busy")?;
    }
    let tw = ThreadTime::now();
    for _ in progs {
        wait_one(b, slot);
        let i = slot[0].tag().0 as usize;
        let w = want.get(i).ok_or("completion with an unknown tag")?;
        check(&slot[0], *w)?;
    }
    Ok(tw.elapsed())
}

/// Runs `progs` (one repetition = all of them in flight, then all
/// completions) until the budget is spent. Every completion is checked
/// against its expectation, so a backend that produced wrong bytes cannot
/// post a time.
pub fn run<B: ChannelBackend + Buffers>(
    b: &B,
    progs: &[ChannelProgram],
    want: &[Expected],
    budget: Budget,
) -> Result<Series, String> {
    let mut slot = [Completion::ok(Tag(0), 0, 0, 0)];
    for _ in 0..budget.warmup {
        once(b, progs, want, &mut slot)?;
    }
    let mut latency = Vec::new();
    let mut cpu_wait = Duration::ZERO;
    let p0 = ProcessTime::now();
    let w0 = Instant::now();
    while latency.len() < budget.max_reps
        && (latency.len() < budget.min_reps || w0.elapsed() < budget.min_time)
    {
        let t = Instant::now();
        cpu_wait += once(b, progs, want, &mut slot)?;
        latency.push(t.elapsed());
    }
    Ok(Series {
        latency,
        wall: w0.elapsed(),
        cpu_process: p0.elapsed(),
        cpu_wait,
    })
}
