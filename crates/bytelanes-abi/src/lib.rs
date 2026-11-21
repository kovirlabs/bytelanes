//! bytelanes-abi: the op format, the contract between all paralanes repos.
//!
//! Channel program wire format, status codes and zero-copy `#[repr(C)]` structs.
//! `no_std`. Frozen after M1: changes only through a pull request that passes on
//! all three machines.
//!
//! Status: stub. Built in M1 (spec: `cf-abi`).

#![no_std]
#![forbid(unsafe_code)]
