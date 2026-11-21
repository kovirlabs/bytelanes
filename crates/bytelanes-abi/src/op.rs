//! Op codes and command flags.

/// One operation a [`ChannelCmd`](crate::ChannelCmd) can request.
///
/// Codes are stable wire values; `0` is never a valid op. Ops without M1
/// semantics (`Read`, `Write`, `Decompress`, `Compress`, `Score`) are reserved
/// names: their parameters must be zero and every M1 backend reports
/// [`Status::UnsupportedOp`](crate::Status::UnsupportedOp) for them. `Fft` is
/// added in M7 through a contract pull request.
///
/// # Semantics
///
/// For every op, `src` is read and `dst` is written; the command's result is
/// the bytes written at the start of `dst`. When a command carries
/// [`Flags::CHAIN`], the next command's `src` is exactly the previous result
/// (its length is the bytes written, not the extent length).
///
/// - **`Checksum`**: CRC-32C (Castagnoli: polynomial `0x1EDC6F41`, reflected,
///   initial value and final XOR `0xFFFFFFFF`) of all `src` bytes. Writes 4
///   bytes, the CRC as a little-endian `u32`. The CRC of an empty input is 0.
/// - **`Encrypt`** and **`Decrypt`**: AES-256 in CTR mode; the two are the
///   same transform. The 16-byte [`CryptParams::iv`](crate::CryptParams) is
///   the first counter block, read as a 128-bit big-endian integer and
///   incremented by one per 16-byte block, wrapping modulo 2^128. Output byte
///   `i` is `src[i]` XOR keystream byte `i`. Writes `src.len` bytes. The key
///   is the 32-byte key the backend holds in
///   [`CryptParams::key_slot`](crate::CryptParams).
/// - **`Filter`**: `src` is a column of fixed-width little-endian values of
///   [`ElemType`](crate::ElemType); its length must be a whole number of
///   elements and at most 2^32 elements. Row `i` matches when its value
///   satisfies the [`CmpOp`](crate::CmpOp) against the operands. Writes the
///   matching row indices in ascending order as little-endian `u32`s, 4 bytes
///   per match. Floats compare as IEEE 754: `NaN` matches only
///   [`CmpOp::Ne`](crate::CmpOp::Ne), and `-0.0` equals `0.0`.
///
/// If `dst` is shorter than the result, the command fails with
/// [`Status::DstTooSmall`](crate::Status::DstTooSmall) and the completion's
/// `bytes` is the size the result needs; `dst` contents are then unspecified.
///
/// # Errors and their order
///
/// A program that fails [`ChannelProgram::validate`](crate::ChannelProgram::validate)
/// completes with `InvalidProgram` and the index the validation names.
/// Otherwise commands run in order and the first failing check is reported,
/// with the command's index. For each command the checks run in this order,
/// so every backend reports the same status for the same program:
///
/// 1. `UnsupportedOp`: the op is reserved or the backend lacks it.
/// 2. `UnsupportedExtent`: the backend lacks the `src` or `dst` kind.
/// 3. `UnknownBuffer`, then `OutOfBounds`: for `src` (unless it is fed by a
///    chain), then for `dst`.
/// 4. `UnknownKey`: a crypt command's key slot is empty.
/// 5. `InvalidLength`: a `Filter` input is not a whole number of elements or
///    has more than 2^32 of them, or a result would exceed `u32::MAX` bytes.
/// 6. `DstTooSmall`.
///
/// Commands before the failing one may already have written their `dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum Op {
    /// Reserved: read from a device into `dst`.
    Read = 1,
    /// Reserved: write `src` to a device.
    Write = 2,
    /// Reserved: decompress `src` into `dst`.
    Decompress = 3,
    /// Reserved: compress `src` into `dst`.
    Compress = 4,
    /// CRC-32C of `src`, written to `dst` as 4 little-endian bytes.
    Checksum = 5,
    /// AES-256-CTR of `src` into `dst`.
    Decrypt = 6,
    /// AES-256-CTR of `src` into `dst` (identical transform to `Decrypt`).
    Encrypt = 7,
    /// Columnar predicate over `src`; writes matching row indices to `dst`.
    Filter = 8,
    /// Reserved: inline model inference.
    Score = 9,
}

impl Op {
    /// Every op, in code order.
    pub const ALL: [Op; 9] = [
        Op::Read,
        Op::Write,
        Op::Decompress,
        Op::Compress,
        Op::Checksum,
        Op::Decrypt,
        Op::Encrypt,
        Op::Filter,
        Op::Score,
    ];

    /// The ops with defined semantics in this format version.
    pub const DEFINED: [Op; 4] = [Op::Checksum, Op::Decrypt, Op::Encrypt, Op::Filter];

    /// Wire code of this op.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// The op with wire code `code`, if any.
    pub const fn from_code(code: u16) -> Option<Op> {
        Some(match code {
            1 => Op::Read,
            2 => Op::Write,
            3 => Op::Decompress,
            4 => Op::Compress,
            5 => Op::Checksum,
            6 => Op::Decrypt,
            7 => Op::Encrypt,
            8 => Op::Filter,
            9 => Op::Score,
            _ => return None,
        })
    }

    /// Bit for this op in [`Caps`](crate::Caps) op masks.
    pub const fn bit(self) -> u32 {
        1 << (self as u16)
    }

    /// Whether this op has semantics defined in this format version.
    pub const fn is_defined(self) -> bool {
        matches!(self, Op::Checksum | Op::Decrypt | Op::Encrypt | Op::Filter)
    }
}

/// Command flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Flags(pub u16);

impl Flags {
    /// No flags.
    pub const NONE: Flags = Flags(0);
    /// Feed this command's result into the next command's `src`. The next
    /// command's `src` must equal this command's `dst`, which must be `Vram`.
    pub const CHAIN: Flags = Flags(1);
    /// Every flag bit defined in this format version. Other bits must be zero.
    pub const KNOWN: u16 = Flags::CHAIN.0;

    /// Raw flag bits.
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Whether every bit of `other` is set in `self`.
    pub const fn contains(self, other: Flags) -> bool {
        self.0 & other.0 == other.0
    }
}
