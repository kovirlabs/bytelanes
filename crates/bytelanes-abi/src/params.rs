//! Op-specific command parameters.

use zerocopy::byteorder::little_endian::U32;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::KeySlot;

/// Length of the op-specific parameter block in every command.
pub const PARAMS_LEN: usize = 32;

/// Parameters of `Encrypt` and `Decrypt`: 32 bytes.
///
/// | Bytes | Field |
/// | --- | --- |
/// | 0..4 | key slot, little-endian `u32` |
/// | 4..16 | reserved, zero |
/// | 16..32 | first counter block (128-bit big-endian counter) |
#[derive(
    FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Clone, Copy, PartialEq, Eq, Debug,
)]
#[repr(C)]
pub struct CryptParams {
    key_slot: U32,
    reserved: [u8; 12],
    /// First counter block.
    pub iv: [u8; 16],
}

impl CryptParams {
    /// Parameters naming `key_slot` with first counter block `iv`.
    pub const fn new(key_slot: KeySlot, iv: [u8; 16]) -> Self {
        CryptParams {
            key_slot: U32::new(key_slot),
            reserved: [0; 12],
            iv,
        }
    }

    /// The key slot.
    pub const fn key_slot(&self) -> KeySlot {
        self.key_slot.get()
    }

    /// Check reserved bytes.
    pub fn check(&self) -> Result<(), &'static str> {
        if self.reserved == [0; 12] {
            Ok(())
        } else {
            Err("crypt params reserved bytes are not zero")
        }
    }
}

/// Element type of a `Filter` column. Values are little-endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ElemType {
    /// `u32`.
    U32 = 1,
    /// `i32`.
    I32 = 2,
    /// IEEE 754 binary32.
    F32 = 3,
    /// `u64`.
    U64 = 4,
    /// `i64`.
    I64 = 5,
    /// IEEE 754 binary64.
    F64 = 6,
}

impl ElemType {
    /// Every element type, in code order.
    pub const ALL: [ElemType; 6] = [
        ElemType::U32,
        ElemType::I32,
        ElemType::F32,
        ElemType::U64,
        ElemType::I64,
        ElemType::F64,
    ];

    /// The type with wire code `code`, if any.
    pub const fn from_code(code: u8) -> Option<ElemType> {
        Some(match code {
            1 => ElemType::U32,
            2 => ElemType::I32,
            3 => ElemType::F32,
            4 => ElemType::U64,
            5 => ElemType::I64,
            6 => ElemType::F64,
            _ => return None,
        })
    }

    /// Width of one element in bytes.
    pub const fn width(self) -> usize {
        match self {
            ElemType::U32 | ElemType::I32 | ElemType::F32 => 4,
            ElemType::U64 | ElemType::I64 | ElemType::F64 => 8,
        }
    }
}

/// Comparison of a `Filter`. `x` is the row value; `a` and `b` the operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CmpOp {
    /// `x == a`.
    Eq = 1,
    /// `x != a`. The only comparison a `NaN` row matches.
    Ne = 2,
    /// `x < a`.
    Lt = 3,
    /// `x <= a`.
    Le = 4,
    /// `x > a`.
    Gt = 5,
    /// `x >= a`.
    Ge = 6,
    /// `a <= x && x <= b`.
    Between = 7,
}

impl CmpOp {
    /// Every comparison, in code order.
    pub const ALL: [CmpOp; 7] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
        CmpOp::Between,
    ];

    /// The comparison with wire code `code`, if any.
    pub const fn from_code(code: u8) -> Option<CmpOp> {
        Some(match code {
            1 => CmpOp::Eq,
            2 => CmpOp::Ne,
            3 => CmpOp::Lt,
            4 => CmpOp::Le,
            5 => CmpOp::Gt,
            6 => CmpOp::Ge,
            7 => CmpOp::Between,
            _ => return None,
        })
    }
}

/// Output form of a `Filter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FilterOutput {
    /// Matching row indices, ascending, as little-endian `u32`s.
    Indices = 1,
}

impl FilterOutput {
    /// The output form with wire code `code`, if any.
    pub const fn from_code(code: u8) -> Option<FilterOutput> {
        match code {
            1 => Some(FilterOutput::Indices),
            _ => None,
        }
    }
}

/// Parameters of `Filter`: 32 bytes.
///
/// | Bytes | Field |
/// | --- | --- |
/// | 0 | element type ([`ElemType`]) |
/// | 1 | comparison ([`CmpOp`]) |
/// | 2 | output form ([`FilterOutput`]) |
/// | 3..16 | reserved, zero |
/// | 16..24 | operand `a`, little-endian |
/// | 24..32 | operand `b`, little-endian; zero unless the comparison is `Between` |
///
/// For 32-bit element types an operand is in the low 4 bytes and the high 4
/// bytes are zero.
#[derive(
    FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Clone, Copy, PartialEq, Eq, Debug,
)]
#[repr(C)]
pub struct FilterParams {
    elem: u8,
    cmp: u8,
    output: u8,
    reserved: [u8; 13],
    a: [u8; 8],
    b: [u8; 8],
}

/// The decoded form of [`FilterParams`]. Operands are raw little-endian bit
/// patterns of the element type, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FilterSpec {
    /// Element type.
    pub elem: ElemType,
    /// Comparison.
    pub cmp: CmpOp,
    /// Output form.
    pub output: FilterOutput,
    /// Operand `a` bits.
    pub a: u64,
    /// Operand `b` bits; zero unless `cmp` is `Between`.
    pub b: u64,
}

impl FilterSpec {
    fn new(elem: ElemType, cmp: CmpOp, a: u64, b: u64) -> Self {
        let b = if cmp == CmpOp::Between { b } else { 0 };
        FilterSpec {
            elem,
            cmp,
            output: FilterOutput::Indices,
            a,
            b,
        }
    }

    /// A `u32` column filter. `b` is used only by `Between`.
    pub fn u32(cmp: CmpOp, a: u32, b: u32) -> Self {
        FilterSpec::new(ElemType::U32, cmp, a.into(), b.into())
    }

    /// An `i32` column filter. `b` is used only by `Between`.
    pub fn i32(cmp: CmpOp, a: i32, b: i32) -> Self {
        FilterSpec::new(ElemType::I32, cmp, (a as u32).into(), (b as u32).into())
    }

    /// An `f32` column filter. `b` is used only by `Between`.
    pub fn f32(cmp: CmpOp, a: f32, b: f32) -> Self {
        FilterSpec::new(ElemType::F32, cmp, a.to_bits().into(), b.to_bits().into())
    }

    /// A `u64` column filter. `b` is used only by `Between`.
    pub fn u64(cmp: CmpOp, a: u64, b: u64) -> Self {
        FilterSpec::new(ElemType::U64, cmp, a, b)
    }

    /// An `i64` column filter. `b` is used only by `Between`.
    pub fn i64(cmp: CmpOp, a: i64, b: i64) -> Self {
        FilterSpec::new(ElemType::I64, cmp, a as u64, b as u64)
    }

    /// An `f64` column filter. `b` is used only by `Between`.
    pub fn f64(cmp: CmpOp, a: f64, b: f64) -> Self {
        FilterSpec::new(ElemType::F64, cmp, a.to_bits(), b.to_bits())
    }

    /// Encode to wire parameters.
    pub fn to_params(&self) -> FilterParams {
        FilterParams {
            elem: self.elem as u8,
            cmp: self.cmp as u8,
            output: self.output as u8,
            reserved: [0; 13],
            a: self.a.to_le_bytes(),
            b: self.b.to_le_bytes(),
        }
    }
}

impl FilterParams {
    /// Decode and check.
    pub fn decode(&self) -> Result<FilterSpec, &'static str> {
        let elem = ElemType::from_code(self.elem).ok_or("unknown filter element type")?;
        let cmp = CmpOp::from_code(self.cmp).ok_or("unknown filter comparison")?;
        let output = FilterOutput::from_code(self.output).ok_or("unknown filter output form")?;
        if self.reserved != [0; 13] {
            return Err("filter params reserved bytes are not zero");
        }
        let a = u64::from_le_bytes(self.a);
        let b = u64::from_le_bytes(self.b);
        if cmp != CmpOp::Between && b != 0 {
            return Err("filter operand b is not zero for a one-operand comparison");
        }
        if elem.width() == 4 && (a >> 32 != 0 || b >> 32 != 0) {
            return Err("filter operand high bytes are not zero for a 32-bit type");
        }
        Ok(FilterSpec {
            elem,
            cmp,
            output,
            a,
            b,
        })
    }
}
