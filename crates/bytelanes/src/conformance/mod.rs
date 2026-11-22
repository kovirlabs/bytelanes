//! Conformance: golden vectors and seeded random cases that every backend
//! must reproduce byte for byte.
//!
//! A [`Case`] is a channel program plus the buffers and keys it needs and the
//! completion and result it must produce. Two sources of cases:
//!
//! - [`load_vectors`] reads the committed golden vectors in `vectors/`.
//!   [`golden_cases`] builds the same set in memory; it is what wrote them.
//! - [`random_case`] builds a case for one op from a seed, with the expected
//!   result computed by the CPU reference. A GPU backend proves R7 by
//!   running seeds `0..10_000` for each op through [`run_case`].
//!
//! ```no_run,standalone_crate
//! use bytelanes::conformance::{self, run_case};
//! use bytelanes::cpu::CpuBackend;
//!
//! let backend = CpuBackend::new();
//! for case in conformance::load_vectors(conformance::vectors_dir()).unwrap() {
//!     run_case(&backend, &case).unwrap();
//! }
//! ```

mod files;
pub mod golden;
mod random;

use std::fmt;
use std::time::{Duration, Instant};

pub use files::{VectorError, load_vectors, vectors_dir, write_vectors};
pub use golden::golden_cases;
pub use random::{SplitMix64, random_case};

use crate::abi::{BufId, Caps, ChannelProgram, Completion, KeySlot, Op, Status, Tag};
use crate::{Buffers, ChannelBackend};

/// One conformance case.
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    /// Unique name, `<group>/<case>`; also the case's directory in `vectors/`.
    pub name: String,
    /// What the case checks.
    pub description: String,
    /// The program, possibly invalid on purpose. Submitted as is.
    pub program: ChannelProgram,
    /// Host buffers to create before submitting.
    pub host: Vec<(BufId, Vec<u8>)>,
    /// Device buffers to allocate (zero-filled) before submitting.
    pub vram: Vec<(BufId, u64)>,
    /// Keys to load before submitting.
    pub keys: Vec<(KeySlot, [u8; 32])>,
    /// The expected outcome.
    pub expect: Expect,
}

impl Case {
    /// The ops with defined semantics that this case's commands use.
    pub fn ops(&self) -> Vec<Op> {
        let mut ops: Vec<Op> = self
            .program
            .commands()
            .iter()
            .filter_map(|c| c.op())
            .filter(|op| op.is_defined())
            .collect();
        ops.sort_unstable();
        ops.dedup();
        ops
    }

    /// Whether a backend with `caps` supports every op this case uses. A
    /// backend that has not implemented an op yet skips the other cases
    /// (and says how many it skipped) rather than failing them.
    pub fn runs_on(&self, caps: &Caps) -> bool {
        self.ops().iter().all(|&op| caps.supports(op))
    }
}

/// Expected outcome of a [`Case`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expect {
    /// Completion status.
    pub status: Status,
    /// Completion command index.
    pub cmd_index: u16,
    /// Completion byte count.
    pub bytes: u32,
    /// Completion CRC (0 unless `status` is `Ok`).
    pub crc: u32,
    /// Host buffer holding the result (checked only when `status` is `Ok`).
    pub output_buf: BufId,
    /// Offset of the result in `output_buf`.
    pub output_offset: u64,
    /// The result bytes (empty unless `status` is `Ok`).
    pub output: Vec<u8>,
}

/// How a backend's run of a [`Case`] differed from the expectation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mismatch {
    /// Setting up buffers or keys failed.
    Setup(Status),
    /// The backend stayed busy or produced no completion in time.
    Timeout,
    /// The completion carried a different tag.
    WrongTag(u64),
    /// A completion field differed.
    Completion {
        /// Field name.
        field: &'static str,
        /// Expected value.
        expected: u64,
        /// Actual value.
        actual: u64,
    },
    /// The result buffer is missing or too short.
    MissingOutput,
    /// The result bytes differ; `first_diff` is the first differing offset.
    Output {
        /// First differing byte offset within the result.
        first_diff: usize,
    },
    /// The completion CRC does not match the result read back.
    Crc(Status),
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mismatch::Setup(s) => write!(f, "setup failed: {s:?}"),
            Mismatch::Timeout => f.write_str("no completion within the timeout"),
            Mismatch::WrongTag(t) => write!(f, "completion has wrong tag {t}"),
            Mismatch::Completion {
                field,
                expected,
                actual,
            } => write!(f, "completion {field}: expected {expected}, got {actual}"),
            Mismatch::MissingOutput => f.write_str("result buffer missing or short"),
            Mismatch::Output { first_diff } => {
                write!(f, "result differs from byte {first_diff}")
            }
            Mismatch::Crc(s) => write!(f, "result CRC check failed: {s:?}"),
        }
    }
}

impl std::error::Error for Mismatch {}

/// Time allowed for one case's completion.
pub const CASE_TIMEOUT: Duration = Duration::from_secs(30);

const CASE_TAG: Tag = Tag(0x6279_7465_6C61_6E65); // "bytelane"

/// Load `case` into `backend`, run it, and compare the outcome.
pub fn run_case<B: ChannelBackend + Buffers>(backend: &B, case: &Case) -> Result<(), Mismatch> {
    for (buf, data) in &case.host {
        backend
            .put_host(*buf, data.clone())
            .map_err(Mismatch::Setup)?;
    }
    for (buf, len) in &case.vram {
        backend.alloc_vram(*buf, *len).map_err(Mismatch::Setup)?;
    }
    for (slot, key) in &case.keys {
        backend.set_key(*slot, *key).map_err(Mismatch::Setup)?;
    }

    let deadline = Instant::now() + CASE_TIMEOUT;
    while backend.submit(&case.program, CASE_TAG).is_err() {
        drain_or_timeout(backend, deadline)?;
    }
    let mut out = [Completion::error(Tag(0), 0, Status::DeviceError, 0)];
    while backend.poll(&mut out) == 0 {
        if Instant::now() > deadline {
            return Err(Mismatch::Timeout);
        }
        std::thread::yield_now();
    }
    check(backend, case, &out[0])
}

fn drain_or_timeout<B: ChannelBackend>(backend: &B, deadline: Instant) -> Result<(), Mismatch> {
    let mut sink = [Completion::error(Tag(0), 0, Status::DeviceError, 0); 64];
    backend.poll(&mut sink);
    if Instant::now() > deadline {
        return Err(Mismatch::Timeout);
    }
    std::thread::yield_now();
    Ok(())
}

fn field(name: &'static str, expected: u64, actual: u64) -> Result<(), Mismatch> {
    if expected == actual {
        Ok(())
    } else {
        Err(Mismatch::Completion {
            field: name,
            expected,
            actual,
        })
    }
}

fn check<B: Buffers>(backend: &B, case: &Case, c: &Completion) -> Result<(), Mismatch> {
    let e = &case.expect;
    if c.tag() != CASE_TAG {
        return Err(Mismatch::WrongTag(c.tag().0));
    }
    field("status", e.status.code().into(), c.status_code().into())?;
    field("cmd_index", e.cmd_index.into(), c.cmd_index().into())?;
    field("bytes", e.bytes.into(), c.bytes().into())?;
    field("crc", e.crc.into(), c.crc().into())?;
    if e.status != Status::Ok {
        return Ok(());
    }
    let buf = backend
        .get_host(e.output_buf)
        .ok_or(Mismatch::MissingOutput)?;
    let start = usize::try_from(e.output_offset).map_err(|_| Mismatch::MissingOutput)?;
    let got = buf
        .get(start..start + e.output.len())
        .ok_or(Mismatch::MissingOutput)?;
    if let Some(first_diff) = got.iter().zip(&e.output).position(|(a, b)| a != b) {
        return Err(Mismatch::Output { first_diff });
    }
    crate::verify(c, got).map_err(Mismatch::Crc)
}

/// Build the expectation for `program` by running it on a fresh CPU
/// reference backend loaded with the case's buffers and keys.
pub fn expect_from_cpu(
    program: &ChannelProgram,
    host: &[(BufId, Vec<u8>)],
    vram: &[(BufId, u64)],
    keys: &[(KeySlot, [u8; 32])],
    output_buf: BufId,
    output_offset: u64,
) -> Expect {
    let cpu = crate::cpu::CpuBackend::new();
    for (b, d) in host {
        let _ = cpu.put_host(*b, d.clone());
    }
    for (b, l) in vram {
        let _ = cpu.alloc_vram(*b, *l);
    }
    for (s, k) in keys {
        let _ = cpu.set_key(*s, *k);
    }
    let c = cpu.run(program, CASE_TAG);
    let status = c.status().unwrap_or(Status::DeviceError);
    let output = if status == Status::Ok {
        let buf = cpu.get_host(output_buf).unwrap_or_default();
        let start = output_offset as usize;
        buf[start..start + c.bytes() as usize].to_vec()
    } else {
        Vec::new()
    };
    Expect {
        status,
        cmd_index: c.cmd_index(),
        bytes: c.bytes(),
        crc: c.crc(),
        output_buf,
        output_offset,
        output,
    }
}
