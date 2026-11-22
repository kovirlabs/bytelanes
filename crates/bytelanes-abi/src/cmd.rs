//! One command of a channel program.

use zerocopy::byteorder::little_endian::U16;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::{
    CryptParams, Extent, ExtentRef, FftParams, FilterParams, FilterSpec, Flags, KeySlot, Op,
    PARAMS_LEN,
};

/// One step of a channel program: 104 bytes.
///
/// | Bytes | Field |
/// | --- | --- |
/// | 0..2 | op code ([`Op`]) |
/// | 2..4 | flags ([`Flags`]) |
/// | 4..8 | reserved, zero |
/// | 8..40 | `src` ([`Extent`]) |
/// | 40..72 | `dst` ([`Extent`]) |
/// | 72..104 | op-specific parameters ([`CryptParams`], [`FilterParams`], or zero) |
#[derive(
    FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Clone, Copy, PartialEq, Eq, Debug,
)]
#[repr(C)]
pub struct ChannelCmd {
    op: U16,
    flags: U16,
    reserved: [u8; 4],
    src: Extent,
    dst: Extent,
    params: [u8; PARAMS_LEN],
}

impl ChannelCmd {
    /// A command from its parts. Prefer the op-specific constructors.
    pub fn new(op: Op, flags: Flags, src: Extent, dst: Extent, params: [u8; PARAMS_LEN]) -> Self {
        ChannelCmd {
            op: U16::new(op.code()),
            flags: U16::new(flags.bits()),
            reserved: [0; 4],
            src,
            dst,
            params,
        }
    }

    /// CRC-32C of `src` into `dst`.
    pub fn checksum(src: Extent, dst: Extent) -> Self {
        ChannelCmd::new(Op::Checksum, Flags::NONE, src, dst, [0; PARAMS_LEN])
    }

    /// AES-256-CTR decrypt of `src` into `dst`.
    pub fn decrypt(src: Extent, dst: Extent, key_slot: KeySlot, iv: [u8; 16]) -> Self {
        let p = CryptParams::new(key_slot, iv);
        ChannelCmd::new(Op::Decrypt, Flags::NONE, src, dst, to_params(p.as_bytes()))
    }

    /// AES-256-CTR encrypt of `src` into `dst`.
    pub fn encrypt(src: Extent, dst: Extent, key_slot: KeySlot, iv: [u8; 16]) -> Self {
        let p = CryptParams::new(key_slot, iv);
        ChannelCmd::new(Op::Encrypt, Flags::NONE, src, dst, to_params(p.as_bytes()))
    }

    /// Filter of the column in `src`; matching row indices into `dst`.
    pub fn filter(src: Extent, dst: Extent, spec: FilterSpec) -> Self {
        let p = spec.to_params();
        ChannelCmd::new(Op::Filter, Flags::NONE, src, dst, to_params(p.as_bytes()))
    }

    /// Fixed-point FFT of the `i32` frames in `src`, `2^log2n` samples each;
    /// half-spectrum bins into `dst`.
    pub fn fft(src: Extent, dst: Extent, log2n: u8) -> Self {
        let p = FftParams::new(log2n);
        ChannelCmd::new(Op::Fft, Flags::NONE, src, dst, to_params(p.as_bytes()))
    }

    /// The same command with [`Flags::CHAIN`] set.
    pub fn chained(mut self) -> Self {
        self.flags = U16::new(self.flags.get() | Flags::CHAIN.bits());
        self
    }

    /// The op, or `None` for an unknown code.
    pub fn op(&self) -> Option<Op> {
        Op::from_code(self.op.get())
    }

    /// Raw op code.
    pub fn op_code(&self) -> u16 {
        self.op.get()
    }

    /// Flags.
    pub fn flags(&self) -> Flags {
        Flags(self.flags.get())
    }

    /// Wire `src`.
    pub fn src(&self) -> &Extent {
        &self.src
    }

    /// Wire `dst`.
    pub fn dst(&self) -> &Extent {
        &self.dst
    }

    /// Raw parameter bytes.
    pub fn params(&self) -> &[u8; PARAMS_LEN] {
        &self.params
    }

    /// Parameters as [`CryptParams`] (meaningful for `Encrypt` and `Decrypt`).
    pub fn crypt_params(&self) -> CryptParams {
        CryptParams::read_from_bytes(&self.params).unwrap_or_else(|_| unreachable_params())
    }

    /// Parameters as [`FftParams`] (meaningful for `Fft`).
    pub fn fft_params(&self) -> FftParams {
        FftParams::read_from_bytes(&self.params).unwrap_or_else(|_| unreachable_params())
    }

    /// Parameters decoded as a [`FilterSpec`] (meaningful for `Filter`).
    pub fn filter_spec(&self) -> Result<FilterSpec, &'static str> {
        FilterParams::read_from_bytes(&self.params)
            .unwrap_or_else(|_| unreachable_params())
            .decode()
    }

    /// Check this command on its own, without reference to its program.
    /// Returns the op and decoded extents.
    pub fn check(&self) -> Result<(Op, ExtentRef, ExtentRef), &'static str> {
        let op = self.op().ok_or("unknown op code")?;
        if self.flags.get() & !Flags::KNOWN != 0 {
            return Err("unknown flag bits set");
        }
        if self.reserved != [0; 4] {
            return Err("command reserved bytes are not zero");
        }
        let src = self.src.decode()?;
        let dst = self.dst.decode()?;
        match op {
            Op::Checksum => {
                if self.params != [0; PARAMS_LEN] {
                    return Err("Checksum params are not zero");
                }
            }
            Op::Decrypt | Op::Encrypt => self.crypt_params().check()?,
            Op::Filter => {
                self.filter_spec()?;
            }
            Op::Fft => self.fft_params().check()?,
            Op::Read | Op::Write | Op::Decompress | Op::Compress | Op::Score => {
                if self.params != [0; PARAMS_LEN] {
                    return Err("params of a reserved op are not zero");
                }
            }
        }
        if op.is_defined() {
            if src == ExtentRef::None {
                return Err("src is None");
            }
            if dst == ExtentRef::None {
                return Err("dst is None");
            }
        }
        Ok((op, src, dst))
    }
}

fn to_params(bytes: &[u8]) -> [u8; PARAMS_LEN] {
    let mut out = [0; PARAMS_LEN];
    out.copy_from_slice(bytes);
    out
}

/// Every parameter struct is exactly `PARAMS_LEN` bytes (checked at compile
/// time in `lib.rs`), so reading one from the parameter block cannot fail.
fn unreachable_params() -> ! {
    unreachable!("parameter structs are PARAMS_LEN bytes")
}
