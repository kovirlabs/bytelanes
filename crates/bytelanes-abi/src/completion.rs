//! Completions and status codes.

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Caller-chosen tag that pairs a [`Completion`] with its submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(pub u64);

/// Outcome of a channel program. Stable wire codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Status {
    /// The program ran; the result is in the final `dst`.
    Ok = 0,
    /// The program failed structural validation.
    InvalidProgram = 1,
    /// The backend does not implement this op.
    UnsupportedOp = 2,
    /// The backend does not support this extent kind.
    UnsupportedExtent = 3,
    /// An extent names a buffer the backend does not know.
    UnknownBuffer = 4,
    /// An extent reaches past the end of its buffer.
    OutOfBounds = 5,
    /// `dst` is shorter than the result; `bytes` holds the size needed.
    DstTooSmall = 6,
    /// The input length is invalid for the op (for example, not a whole
    /// number of `Filter` elements).
    InvalidLength = 7,
    /// A crypt command names a key slot the backend does not hold.
    UnknownKey = 8,
    /// The device failed while running the program.
    DeviceError = 9,
    /// Set by a receiver whose check of the completion CRC failed.
    CrcMismatch = 10,
}

impl Status {
    /// The status with wire code `code`, if any.
    pub const fn from_code(code: u16) -> Option<Status> {
        Some(match code {
            0 => Status::Ok,
            1 => Status::InvalidProgram,
            2 => Status::UnsupportedOp,
            3 => Status::UnsupportedExtent,
            4 => Status::UnknownBuffer,
            5 => Status::OutOfBounds,
            6 => Status::DstTooSmall,
            7 => Status::InvalidLength,
            8 => Status::UnknownKey,
            9 => Status::DeviceError,
            10 => Status::CrcMismatch,
            _ => return None,
        })
    }

    /// Wire code.
    pub const fn code(self) -> u16 {
        self as u16
    }
}

/// Result of one channel program: 24 bytes.
///
/// | Bytes | Field |
/// | --- | --- |
/// | 0..8 | tag |
/// | 8..10 | status ([`Status`]) |
/// | 10..12 | index of the command that failed, the last command on success, or [`NO_INDEX`](crate::NO_INDEX) for a program-level failure |
/// | 12..16 | bytes written to the final `dst` on success; size needed on `DstTooSmall`; otherwise 0 |
/// | 16..20 | CRC-32C of the result bytes on success, computed where the result was produced; otherwise 0 |
/// | 20..24 | reserved, zero |
#[derive(
    FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Clone, Copy, PartialEq, Eq, Debug,
)]
#[repr(C)]
pub struct Completion {
    tag: U64,
    status: U16,
    cmd_index: U16,
    bytes: U32,
    crc: U32,
    reserved: [u8; 4],
}

impl Completion {
    /// A successful completion.
    pub const fn ok(tag: Tag, cmd_index: u16, bytes: u32, crc: u32) -> Self {
        Completion {
            tag: U64::new(tag.0),
            status: U16::new(Status::Ok as u16),
            cmd_index: U16::new(cmd_index),
            bytes: U32::new(bytes),
            crc: U32::new(crc),
            reserved: [0; 4],
        }
    }

    /// A failed completion. `bytes` is the size needed for `DstTooSmall` and
    /// should be 0 otherwise.
    pub const fn error(tag: Tag, cmd_index: u16, status: Status, bytes: u32) -> Self {
        Completion {
            tag: U64::new(tag.0),
            status: U16::new(status as u16),
            cmd_index: U16::new(cmd_index),
            bytes: U32::new(bytes),
            crc: U32::new(0),
            reserved: [0; 4],
        }
    }

    /// The submission's tag.
    pub const fn tag(&self) -> Tag {
        Tag(self.tag.get())
    }

    /// The status, or `None` for an unknown code.
    pub const fn status(&self) -> Option<Status> {
        Status::from_code(self.status.get())
    }

    /// Raw status code.
    pub const fn status_code(&self) -> u16 {
        self.status.get()
    }

    /// Command index (see the table above).
    pub const fn cmd_index(&self) -> u16 {
        self.cmd_index.get()
    }

    /// Byte count (see the table above).
    pub const fn bytes(&self) -> u32 {
        self.bytes.get()
    }

    /// Result CRC-32C (see the table above).
    pub const fn crc(&self) -> u32 {
        self.crc.get()
    }

    /// The same completion with a different tag.
    pub const fn with_tag(mut self, tag: Tag) -> Self {
        self.tag = U64::new(tag.0);
        self
    }
}
