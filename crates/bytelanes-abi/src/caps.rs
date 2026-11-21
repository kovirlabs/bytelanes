//! Backend capabilities and path health.

use zerocopy::byteorder::little_endian::U32;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::{ExtentKind, Op};

/// Optional backend features.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Features(pub u8);

impl Features {
    /// No optional features.
    pub const NONE: Features = Features(0);
    /// Host and device share memory; `Host` extents need no copy.
    pub const UNIFIED_MEMORY: Features = Features(1);
    /// `Nvme` extents move directly between the device and device memory.
    pub const DIRECT_NVME: Features = Features(2);

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: Features) -> bool {
        self.0 & other.0 == other.0
    }

    /// Union of two feature sets.
    pub const fn union(self, other: Features) -> Features {
        Features(self.0 | other.0)
    }
}

/// What a backend can run: 16 bytes.
///
/// | Bytes | Field |
/// | --- | --- |
/// | 0..4 | op mask, bit [`Op::bit`] per supported op |
/// | 4 | extent mask, bit [`ExtentKind::bit`] per supported kind |
/// | 5 | [`Features`] |
/// | 6..8 | reserved, zero |
/// | 8..12 | queue depth: programs accepted before `submit` reports busy |
/// | 12..16 | reserved, zero |
#[derive(
    FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Clone, Copy, PartialEq, Eq, Debug,
)]
#[repr(C)]
pub struct Caps {
    ops: U32,
    extents: u8,
    features: u8,
    reserved: [u8; 2],
    queue_depth: U32,
    reserved2: [u8; 4],
}

impl Caps {
    /// Capabilities for the given ops, extent kinds, features and queue depth.
    pub fn new(ops: &[Op], extents: &[ExtentKind], features: Features, queue_depth: u32) -> Self {
        let ops = ops.iter().fold(0u32, |m, op| m | op.bit());
        let extents = extents.iter().fold(0u8, |m, k| m | k.bit());
        Caps {
            ops: U32::new(ops),
            extents,
            features: features.0,
            reserved: [0; 2],
            queue_depth: U32::new(queue_depth),
            reserved2: [0; 4],
        }
    }

    /// Whether `op` is supported.
    pub fn supports(&self, op: Op) -> bool {
        self.ops.get() & op.bit() != 0
    }

    /// Whether extents of `kind` are supported.
    pub fn supports_extent(&self, kind: ExtentKind) -> bool {
        self.extents & kind.bit() != 0
    }

    /// Optional features.
    pub const fn features(&self) -> Features {
        Features(self.features)
    }

    /// Queue depth.
    pub fn queue_depth(&self) -> u32 {
        self.queue_depth.get()
    }
}

/// Health of a backend path. Stable wire codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PathState {
    /// Accepting and completing work.
    Online = 1,
    /// Completing work, with errors or reduced throughput.
    Degraded = 2,
    /// Not accepting work; the scheduler resubmits elsewhere.
    Offline = 3,
}

impl PathState {
    /// The state with wire code `code`, if any.
    pub const fn from_code(code: u8) -> Option<PathState> {
        Some(match code {
            1 => PathState::Online,
            2 => PathState::Degraded,
            3 => PathState::Offline,
            _ => return None,
        })
    }
}
