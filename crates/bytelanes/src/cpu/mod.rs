//! The CPU reference: pure op functions in [`ops`] and a [`CpuBackend`] that
//! runs whole channel programs with them.
//!
//! The op functions use the `crc32c`, `aes` and `ctr` crates, which pick the
//! CPU's CRC32 and AES instructions at runtime, so benchmarks compare GPUs
//! against a fast CPU.

mod backend;
pub mod ops;

pub use backend::CpuBackend;
