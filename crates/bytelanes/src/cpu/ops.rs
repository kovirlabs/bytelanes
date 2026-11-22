//! Pure CPU implementations of each op, as specified on [`Op`](crate::abi::Op).

use aes::Aes256;
use ctr::cipher::{KeyIvInit, StreamCipher};

use crate::abi::{CmpOp, ElemType, FilterSpec, Status};

/// An AES-256 key.
pub type Key = [u8; 32];

/// CRC-32C (Castagnoli) of `data`.
pub fn checksum(data: &[u8]) -> u32 {
    crc32c::crc32c(data)
}

type Aes256Ctr = ctr::Ctr128BE<Aes256>;

/// AES-256-CTR of `src` into `dst` with first counter block `iv`
/// (128-bit big-endian, wrapping). Encrypt and decrypt are the same call.
///
/// # Panics
///
/// If `src` and `dst` differ in length.
pub fn aes256_ctr(key: &Key, iv: &[u8; 16], src: &[u8], dst: &mut [u8]) {
    let mut c = Aes256Ctr::new(key.into(), iv.into());
    c.apply_keystream_b2b(src, dst);
}

/// AES-256-CTR of `buf` in place. See [`aes256_ctr`].
pub fn aes256_ctr_in_place(key: &Key, iv: &[u8; 16], buf: &mut [u8]) {
    let mut c = Aes256Ctr::new(key.into(), iv.into());
    c.apply_keystream(buf);
}

/// Largest number of rows a `Filter` input may have: indices are `u32`.
pub const MAX_FILTER_ROWS: u64 = 1 << 32;

/// Run `spec` over the column `src`, appending matching row indices to `out`
/// in ascending order.
///
/// Returns [`Status::InvalidLength`] if `src` is not a whole number of
/// elements or has more than [`MAX_FILTER_ROWS`] of them; `out` is then
/// unchanged.
pub fn filter(spec: &FilterSpec, src: &[u8], out: &mut Vec<u32>) -> Result<(), Status> {
    let w = spec.elem.width();
    if !src.len().is_multiple_of(w) || (src.len() / w) as u64 > MAX_FILTER_ROWS {
        return Err(Status::InvalidLength);
    }
    let (a, b) = (spec.a, spec.b);
    match spec.elem {
        ElemType::U32 => scan::<4, _>(src, spec.cmp, a as u32, b as u32, u32::from_le_bytes, out),
        ElemType::I32 => scan::<4, _>(
            src,
            spec.cmp,
            a as u32 as i32,
            b as u32 as i32,
            i32::from_le_bytes,
            out,
        ),
        ElemType::F32 => scan::<4, _>(
            src,
            spec.cmp,
            f32::from_bits(a as u32),
            f32::from_bits(b as u32),
            f32::from_le_bytes,
            out,
        ),
        ElemType::U64 => scan::<8, _>(src, spec.cmp, a, b, u64::from_le_bytes, out),
        ElemType::I64 => scan::<8, _>(src, spec.cmp, a as i64, b as i64, i64::from_le_bytes, out),
        ElemType::F64 => scan::<8, _>(
            src,
            spec.cmp,
            f64::from_bits(a),
            f64::from_bits(b),
            f64::from_le_bytes,
            out,
        ),
    }
    Ok(())
}

/// Dispatch on the comparison once, so each inner loop is branch-free apart
/// from the predicate itself.
fn scan<const W: usize, T: PartialOrd + Copy>(
    src: &[u8],
    cmp: CmpOp,
    a: T,
    b: T,
    decode: fn([u8; W]) -> T,
    out: &mut Vec<u32>,
) {
    match cmp {
        CmpOp::Eq => scan_with(src, decode, out, |x| x == a),
        CmpOp::Ne => scan_with(src, decode, out, |x| x != a),
        CmpOp::Lt => scan_with(src, decode, out, |x| x < a),
        CmpOp::Le => scan_with(src, decode, out, |x| x <= a),
        CmpOp::Gt => scan_with(src, decode, out, |x| x > a),
        CmpOp::Ge => scan_with(src, decode, out, |x| x >= a),
        CmpOp::Between => scan_with(src, decode, out, |x| a <= x && x <= b),
    }
}

fn scan_with<const W: usize, T: Copy>(
    src: &[u8],
    decode: fn([u8; W]) -> T,
    out: &mut Vec<u32>,
    pred: impl Fn(T) -> bool,
) {
    // `filter` has checked that `src` is a whole number of elements, so the
    // remainder is empty.
    let (elems, _) = src.as_chunks::<W>();
    for (i, &raw) in elems.iter().enumerate() {
        if pred(decode(raw)) {
            // Row count is checked against MAX_FILTER_ROWS, so i fits in u32.
            out.push(i as u32);
        }
    }
}

/// Encode filter output indices as little-endian bytes.
pub fn indices_to_bytes(indices: &[u32]) -> Vec<u8> {
    indices.iter().flat_map(|i| i.to_le_bytes()).collect()
}
