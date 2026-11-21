//! bytelanes: the `ChannelBackend` trait and the CPU reference backend.
//!
//! Every op (Checksum, Filter, Decrypt; later FFT) has a CPU reference here that
//! every GPU backend must match byte for byte.
//!
//! Status: stub. Built in M1 (spec: `cf-backend`).

#![forbid(unsafe_code)]
