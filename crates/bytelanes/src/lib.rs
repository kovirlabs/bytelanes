//! bytelanes: the `ChannelBackend` trait and the CPU reference backend.
//!
//! Every op (Checksum, Filter, Decrypt; later FFT) has a CPU reference in
//! [`cpu`] that every GPU backend must match byte for byte. [`conformance`]
//! holds the golden vectors and the replay and random-case harness that GPU
//! backends run to prove it.
//!
//! The op format itself lives in [`abi`] (the `bytelanes-abi` crate).

#![forbid(unsafe_code)]

pub use bytelanes_abi as abi;

pub mod conformance;
pub mod cpu;

use abi::{BufId, Caps, ChannelProgram, Completion, KeySlot, PathState, Status, Tag};

/// Returned by [`ChannelBackend::submit`] when the backend's queue is full.
/// Poll completions, then retry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Busy;

impl core::fmt::Display for Busy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("backend queue is full")
    }
}

impl std::error::Error for Busy {}

/// A path that runs channel programs: the CPU reference, a local GPU, or a
/// remote channel daemon.
///
/// `submit` never blocks on the work itself. A program that is invalid or
/// fails while running is still accepted; its [`Completion`] carries the
/// status. Every accepted program produces exactly one completion.
pub trait ChannelBackend: Send + Sync {
    /// What this backend can run.
    fn capabilities(&self) -> Caps;

    /// Queue `prog`; its completion will carry `tag`.
    fn submit(&self, prog: &ChannelProgram, tag: Tag) -> Result<(), Busy>;

    /// Move up to `out.len()` finished completions into `out`; returns how
    /// many were written. Never blocks.
    fn poll(&self, out: &mut [Completion]) -> usize;

    /// Current health of this path.
    fn health(&self) -> PathState;
}

/// Buffer and key management for a backend.
///
/// Channel programs name buffers and keys by id; this is how a caller
/// provides them. The conformance harness uses it to load golden vectors.
/// Unlike the wire format, this interface is not frozen.
pub trait Buffers {
    /// Create or replace host buffer `buf` with `data`.
    fn put_host(&self, buf: BufId, data: Vec<u8>) -> Result<(), Status>;

    /// A copy of host buffer `buf`, if it exists.
    fn get_host(&self, buf: BufId) -> Option<Vec<u8>>;

    /// Create or replace device buffer `buf` with `len` zero bytes.
    fn alloc_vram(&self, buf: BufId, len: u64) -> Result<(), Status>;

    /// Store an AES-256 key in `slot`.
    fn set_key(&self, slot: KeySlot, key: [u8; 32]) -> Result<(), Status>;
}

/// Check a completion against the result bytes the receiver read.
///
/// Every result leaving a device carries a CRC-32C computed where it was
/// produced; the receiver recomputes it here. Returns the completion's status
/// if it is not `Ok`, and [`Status::CrcMismatch`] if the length or CRC
/// disagree.
pub fn verify(c: &Completion, result: &[u8]) -> Result<(), Status> {
    match c.status() {
        Some(Status::Ok) => {}
        Some(s) => return Err(s),
        None => return Err(Status::DeviceError),
    }
    if u64::from(c.bytes()) != result.len() as u64 || cpu::ops::checksum(result) != c.crc() {
        return Err(Status::CrcMismatch);
    }
    Ok(())
}
