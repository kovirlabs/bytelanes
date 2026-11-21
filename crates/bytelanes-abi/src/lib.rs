//! bytelanes-abi: the op format, the contract between all paralanes repos.
//!
//! A core describes bulk data work as a [`ChannelProgram`]: up to
//! [`MAX_COMMANDS`] [`ChannelCmd`]s, each naming an [`Op`], a source and a
//! destination [`Extent`], and 32 bytes of op-specific parameters. A backend
//! runs the program and reports one [`Completion`].
//!
//! # Wire format
//!
//! Every type that crosses a boundary (ring, network, file) is a `#[repr(C)]`
//! struct built only from bytes and little-endian integers, so it has
//! alignment 1, no padding, and the same bytes on every machine. Read one
//! from a byte slice with [`zerocopy::FromBytes`] (re-exported here) and
//! write one out with [`zerocopy::IntoBytes::as_bytes`]. Sizes are fixed and
//! checked at compile time:
//!
//! | Type | Bytes |
//! | --- | --- |
//! | [`Extent`] | 32 |
//! | [`ChannelCmd`] | 104 |
//! | [`ChannelProgram`] | 1672 |
//! | [`Completion`] | 24 |
//! | [`Caps`] | 16 |
//!
//! All reserved fields and all unused command slots must be zero, so a valid
//! program has exactly one byte encoding.
//!
//! # Op semantics
//!
//! The semantics every backend must reproduce byte for byte are documented on
//! [`Op`]. The CPU reference in the `bytelanes` crate implements them, and
//! the golden vectors in `vectors/` record its output.
//!
//! `no_std`, no allocation. Frozen after M1: changes only through a pull
//! request that passes on all three machines.

#![no_std]
#![forbid(unsafe_code)]

mod caps;
mod cmd;
mod completion;
mod extent;
mod op;
mod params;
mod program;

pub use caps::{Caps, Features, PathState};
pub use cmd::ChannelCmd;
pub use completion::{Completion, Status, Tag};
pub use extent::{BufId, BufRange, DevId, Extent, ExtentKind, ExtentRef, NvmeRange};
pub use op::{Flags, Op};
pub use params::{
    CmpOp, CryptParams, ElemType, FilterOutput, FilterParams, FilterSpec, PARAMS_LEN,
};
pub use program::{ChannelProgram, NO_INDEX, ProgramError};

/// Re-export of the `zerocopy` version the wire types implement, so callers
/// use matching `FromBytes` and `IntoBytes` traits.
pub use zerocopy;

/// Wire format version carried in every [`ChannelProgram`].
pub const FORMAT_VERSION: u16 = 1;

/// Maximum number of commands in one [`ChannelProgram`].
pub const MAX_COMMANDS: usize = 16;

/// Identifier of an AES-256 key held by a backend. Keys never travel inside a
/// channel program; a command names the slot.
pub type KeySlot = u32;

const _: () = {
    use core::mem::size_of;
    assert!(size_of::<Extent>() == 32);
    assert!(size_of::<ChannelCmd>() == 104);
    assert!(size_of::<ChannelProgram>() == 8 + MAX_COMMANDS * 104);
    assert!(size_of::<Completion>() == 24);
    assert!(size_of::<Caps>() == 16);
    assert!(size_of::<CryptParams>() == PARAMS_LEN);
    assert!(size_of::<FilterParams>() == PARAMS_LEN);
};
