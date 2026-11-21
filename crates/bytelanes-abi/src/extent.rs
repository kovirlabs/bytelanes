//! Extents: where a command reads or writes.

use zerocopy::byteorder::little_endian::U64;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Identifier of a buffer known to a backend: a pinned host buffer or a
/// device-resident (`Vram`) buffer. Channel programs carry ids, never
/// pointers; the backend resolves them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufId(pub u64);

/// Identifier of a block device known to a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DevId(pub u64);

/// A byte range inside a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufRange {
    /// The buffer.
    pub buf: BufId,
    /// Offset of the first byte.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
}

impl BufRange {
    /// A range of `len` bytes at `offset` in `buf`.
    pub const fn new(buf: BufId, offset: u64, len: u64) -> Self {
        BufRange { buf, offset, len }
    }

    /// One past the last byte, or `None` on overflow.
    pub const fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.len)
    }

    /// Whether the two ranges share at least one byte of the same buffer.
    pub fn overlaps(&self, other: &BufRange) -> bool {
        if self.buf != other.buf || self.len == 0 || other.len == 0 {
            return false;
        }
        let (Some(a_end), Some(b_end)) = (self.end(), other.end()) else {
            return true;
        };
        self.offset < b_end && other.offset < a_end
    }
}

/// A run of blocks on a block device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NvmeRange {
    /// The device.
    pub dev: DevId,
    /// First logical block.
    pub lba: u64,
    /// Number of blocks, at most `u32::MAX`.
    pub blocks: u32,
}

/// Wire code of an [`Extent`]'s kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExtentKind {
    /// No extent.
    None = 0,
    /// Pinned host memory, NUMA-local to the requesting core.
    Host = 1,
    /// Device memory; stays on the GPU.
    Vram = 2,
    /// Blocks on an NVMe device.
    Nvme = 3,
}

impl ExtentKind {
    /// The kind with wire code `code`, if any.
    pub const fn from_code(code: u8) -> Option<ExtentKind> {
        Some(match code {
            0 => ExtentKind::None,
            1 => ExtentKind::Host,
            2 => ExtentKind::Vram,
            3 => ExtentKind::Nvme,
            _ => return None,
        })
    }

    /// Bit for this kind in [`Caps`](crate::Caps) extent masks.
    pub const fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

/// The decoded, typed form of an [`Extent`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExtentRef {
    /// No extent.
    None,
    /// A range of a pinned host buffer.
    Host(BufRange),
    /// A range of a device buffer.
    Vram(BufRange),
    /// Blocks on an NVMe device.
    Nvme(NvmeRange),
}

impl ExtentRef {
    /// The kind of this extent.
    pub const fn kind(&self) -> ExtentKind {
        match self {
            ExtentRef::None => ExtentKind::None,
            ExtentRef::Host(_) => ExtentKind::Host,
            ExtentRef::Vram(_) => ExtentKind::Vram,
            ExtentRef::Nvme(_) => ExtentKind::Nvme,
        }
    }

    /// The buffer range, for `Host` and `Vram` extents.
    pub const fn buf_range(&self) -> Option<BufRange> {
        match self {
            ExtentRef::Host(r) | ExtentRef::Vram(r) => Some(*r),
            _ => None,
        }
    }
}

/// Wire form of an extent: 32 bytes.
///
/// | Bytes | Field | `Host`, `Vram` | `Nvme` | `None` |
/// | --- | --- | --- | --- | --- |
/// | 0 | kind | 1, 2 | 3 | 0 |
/// | 1..8 | reserved | 0 | 0 | 0 |
/// | 8..16 | a | buffer id | device id | 0 |
/// | 16..24 | b | offset | first block | 0 |
/// | 24..32 | c | length | block count | 0 |
///
/// For `Host` and `Vram`, `offset + length` must not overflow `u64`. For
/// `Nvme`, the block count is at most `u32::MAX` and `first block + count`
/// must not overflow.
#[derive(
    FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Clone, Copy, PartialEq, Eq, Debug,
)]
#[repr(C)]
pub struct Extent {
    kind: u8,
    reserved: [u8; 7],
    a: U64,
    b: U64,
    c: U64,
}

impl Extent {
    /// No extent.
    pub const NONE: Extent = Extent::raw(ExtentKind::None, 0, 0, 0);

    const fn raw(kind: ExtentKind, a: u64, b: u64, c: u64) -> Self {
        Extent {
            kind: kind as u8,
            reserved: [0; 7],
            a: U64::new(a),
            b: U64::new(b),
            c: U64::new(c),
        }
    }

    /// A range of a pinned host buffer.
    pub const fn host(buf: BufId, offset: u64, len: u64) -> Self {
        Extent::raw(ExtentKind::Host, buf.0, offset, len)
    }

    /// A range of a device buffer.
    pub const fn vram(buf: BufId, offset: u64, len: u64) -> Self {
        Extent::raw(ExtentKind::Vram, buf.0, offset, len)
    }

    /// Blocks on an NVMe device.
    pub const fn nvme(dev: DevId, lba: u64, blocks: u32) -> Self {
        Extent::raw(ExtentKind::Nvme, dev.0, lba, blocks as u64)
    }

    /// The wire form of a typed extent.
    pub const fn from_ref(r: ExtentRef) -> Self {
        match r {
            ExtentRef::None => Extent::NONE,
            ExtentRef::Host(b) => Extent::host(b.buf, b.offset, b.len),
            ExtentRef::Vram(b) => Extent::vram(b.buf, b.offset, b.len),
            ExtentRef::Nvme(n) => Extent::nvme(n.dev, n.lba, n.blocks),
        }
    }

    /// Raw kind byte.
    pub const fn kind_code(&self) -> u8 {
        self.kind
    }

    /// Decode and check this extent.
    pub fn decode(&self) -> Result<ExtentRef, &'static str> {
        if self.reserved != [0; 7] {
            return Err("extent reserved bytes are not zero");
        }
        let (a, b, c) = (self.a.get(), self.b.get(), self.c.get());
        match ExtentKind::from_code(self.kind) {
            None => Err("unknown extent kind"),
            Some(ExtentKind::None) => {
                if a == 0 && b == 0 && c == 0 {
                    Ok(ExtentRef::None)
                } else {
                    Err("None extent has non-zero fields")
                }
            }
            Some(kind @ (ExtentKind::Host | ExtentKind::Vram)) => {
                let r = BufRange::new(BufId(a), b, c);
                if r.end().is_none() {
                    return Err("extent offset + length overflows");
                }
                Ok(if kind == ExtentKind::Host {
                    ExtentRef::Host(r)
                } else {
                    ExtentRef::Vram(r)
                })
            }
            Some(ExtentKind::Nvme) => {
                let Ok(blocks) = u32::try_from(c) else {
                    return Err("Nvme block count exceeds u32");
                };
                if b.checked_add(c).is_none() {
                    return Err("Nvme lba + blocks overflows");
                }
                Ok(ExtentRef::Nvme(NvmeRange {
                    dev: DevId(a),
                    lba: b,
                    blocks,
                }))
            }
        }
    }
}

impl From<ExtentRef> for Extent {
    fn from(r: ExtentRef) -> Self {
        Extent::from_ref(r)
    }
}
