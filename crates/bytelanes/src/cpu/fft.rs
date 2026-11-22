//! The fixed-point FFT reference, exactly as specified on
//! [`Op`](crate::abi::Op) (`Fft`).
//!
//! Everything here is integer arithmetic except the one-time twiddle table,
//! which is built from a Taylor series using only `f64` addition and
//! multiplication. Rust never contracts those into fused multiply-adds and
//! IEEE 754 rounds each one exactly, so the table is the same on every
//! platform. The committed copy in `vectors/fft/twiddles.bin` is normative;
//! a test checks this code reproduces it.

use std::sync::OnceLock;

use crate::abi::{FFT_MAX_LOG2N, Status};

/// Length of the full twiddle table: `W[t]` for `t in 0..TWIDDLES`.
pub const TWIDDLES: usize = 1 << (FFT_MAX_LOG2N - 1);

/// Points in the full circle the table divides: `W[t] = exp(-2πi t / CIRCLE)`.
const CIRCLE: usize = 1 << FFT_MAX_LOG2N;

/// Largest magnitude of an input sample after saturation: `2^30`.
pub const SAMPLE_LIMIT: i32 = 1 << 30;

/// One Q30 twiddle: `(round(cos θ · 2^30), round(-sin θ · 2^30))`.
pub type Twiddle = (i32, i32);

/// The Q30 twiddle table (32,768 entries), computed once.
pub fn twiddles() -> &'static [Twiddle] {
    static TABLE: OnceLock<Vec<Twiddle>> = OnceLock::new();
    TABLE.get_or_init(build_twiddles)
}

/// The twiddle table as its normative bytes: pairs of little-endian `i32`.
pub fn twiddle_bytes() -> Vec<u8> {
    twiddles()
        .iter()
        .flat_map(|&(r, i)| r.to_le_bytes().into_iter().chain(i.to_le_bytes()))
        .collect()
}

/// `sin x` and `cos x` for `0 <= x <= π/4`, Taylor series to x^24, using only
/// `f64` `+` and `*` so the result is the same on every platform.
fn sin_cos_small(x: f64) -> (f64, f64) {
    let x2 = x * x;
    // sin: x - x^3/3! + ... - x^23/23!; cos: 1 - x^2/2! + ... + x^24/24!.
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    // Horner from the highest term. Coefficients are exact reciprocals of
    // factorials computed at compile time in f64.
    const SIN: [f64; 12] = sin_coeffs();
    const COS: [f64; 13] = cos_coeffs();
    for &k in SIN.iter().rev() {
        s = s * x2 + k;
    }
    for &k in COS.iter().rev() {
        c = c * x2 + k;
    }
    (s * x, c)
}

const fn sin_coeffs() -> [f64; 12] {
    // (-1)^n / (2n+1)!
    let mut out = [0.0; 12];
    let mut fact = 1.0f64;
    let mut i = 0;
    let mut k = 1u32;
    while i < 12 {
        // fact = (2i+1)!
        if i > 0 {
            fact *= (k as f64) * ((k + 1) as f64);
            k += 2;
        } else {
            k = 2;
        }
        out[i] = if i % 2 == 0 { 1.0 / fact } else { -1.0 / fact };
        i += 1;
    }
    out
}

const fn cos_coeffs() -> [f64; 13] {
    // (-1)^n / (2n)!
    let mut out = [0.0; 13];
    let mut fact = 1.0f64;
    let mut i = 0;
    let mut k = 1u32;
    while i < 13 {
        if i > 0 {
            fact *= (k as f64) * ((k + 1) as f64);
            k += 2;
        }
        out[i] = if i % 2 == 0 { 1.0 / fact } else { -1.0 / fact };
        i += 1;
    }
    out
}

fn q30(v: f64) -> i32 {
    // v is within [-1, 1]; the product by 2^30 is exact, round() is
    // half-away-from-zero and exact.
    (v * (1u64 << 30) as f64).round() as i32
}

fn build_twiddles() -> Vec<Twiddle> {
    let eighth = CIRCLE / 8; // π/4
    let quarter = CIRCLE / 4; // π/2
    let half = CIRCLE / 2; // π
    let step = std::f64::consts::TAU / CIRCLE as f64; // exact: CIRCLE is a power of two
    // First octant directly, 0 <= θ <= π/4.
    let octant: Vec<(i32, i32)> = (0..=eighth)
        .map(|t| {
            let (s, c) = sin_cos_small(t as f64 * step);
            (q30(c), q30(s))
        })
        .collect();
    // cos and sin for 0 <= θ <= π/2 from octant symmetry.
    let quadrant = |t: usize| -> (i32, i32) {
        if t <= eighth {
            octant[t]
        } else {
            let (c, s) = octant[quarter - t];
            (s, c)
        }
    };
    (0..half)
        .map(|t| {
            let (c, s) = if t <= quarter {
                quadrant(t)
            } else {
                let (c, s) = quadrant(half - t);
                (-c, s)
            };
            (c, -s)
        })
        .collect()
}

/// Reverse the low `bits` bits of `i`.
fn bitrev(i: usize, bits: u8) -> usize {
    i.reverse_bits() >> (usize::BITS - u32::from(bits))
}

/// Transform one frame of `2^log2n` samples into bins `0..=N/2`, appending
/// them to `out`. `log2n` must be in `1..=16` and `frame.len()` must be `N`.
pub fn fft_frame(log2n: u8, frame: &[i32], out: &mut Vec<(i32, i32)>) {
    let n = 1usize << log2n;
    debug_assert_eq!(frame.len(), n);
    let w = twiddles();
    let mut a: Vec<(i64, i64)> = (0..n)
        .map(|i| {
            let x = frame[bitrev(i, log2n)].clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT);
            (i64::from(x), 0)
        })
        .collect();
    for s in 1..=log2n {
        let m = 1usize << s;
        let h = m / 2;
        let stride = CIRCLE / m;
        for k in (0..n).step_by(m) {
            for j in 0..h {
                let (wr, wi) = w[j * stride];
                let (wr, wi) = (i64::from(wr), i64::from(wi));
                let (ur, ui) = a[k + j];
                let (br, bi) = a[k + j + h];
                let tr = (wr * br - wi * bi + (1 << 29)) >> 30;
                let ti = (wr * bi + wi * br + (1 << 29)) >> 30;
                a[k + j] = ((ur + tr) >> 1, (ui + ti) >> 1);
                a[k + j + h] = ((ur - tr) >> 1, (ui - ti) >> 1);
            }
        }
    }
    // Every value fits in i32 (inputs are clamped to ±2^30 and each stage
    // halves); the conversion is checked in debug builds.
    out.extend(a[..=n / 2].iter().map(|&(r, i)| (narrow(r), narrow(i))));
}

fn narrow(v: i64) -> i32 {
    debug_assert!(i32::try_from(v).is_ok(), "fft intermediate {v} left i32");
    v as i32
}

/// Bytes of output for `frames` frames of `2^log2n` samples.
pub fn output_len(log2n: u8, frames: usize) -> usize {
    frames * ((1usize << log2n) / 2 + 1) * 8
}

/// FFT of every frame in `src` (little-endian `i32` samples) into `out` as
/// little-endian `(re, im)` pairs. Returns `InvalidLength` if `src` is not a
/// whole number of frames or `log2n` is out of range.
pub fn fft(log2n: u8, src: &[u8], out: &mut Vec<u8>) -> Result<(), Status> {
    if !(1..=FFT_MAX_LOG2N).contains(&log2n) {
        return Err(Status::InvalidLength);
    }
    let frame_bytes = (1usize << log2n) * 4;
    if !src.len().is_multiple_of(frame_bytes) {
        return Err(Status::InvalidLength);
    }
    let mut samples = Vec::with_capacity(1 << log2n);
    let mut bins = Vec::with_capacity((1 << log2n) / 2 + 1);
    out.reserve(output_len(log2n, src.len() / frame_bytes));
    for frame in src.chunks_exact(frame_bytes) {
        samples.clear();
        samples.extend(
            frame
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| i32::from_le_bytes(b)),
        );
        bins.clear();
        fft_frame(log2n, &samples, &mut bins);
        for &(r, i) in &bins {
            out.extend_from_slice(&r.to_le_bytes());
            out.extend_from_slice(&i.to_le_bytes());
        }
    }
    Ok(())
}
