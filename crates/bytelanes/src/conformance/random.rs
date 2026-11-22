//! Seeded random cases.

use super::{Case, expect_from_cpu};
use crate::abi::{BufId, ChannelCmd, ChannelProgram, CmpOp, ElemType, Extent, FilterSpec, Op};

/// SplitMix64: a small, fixed PRNG so seeds give the same cases on every
/// machine and every version of every dependency.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n` (`n > 0`).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// `len` random bytes.
    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len + 8);
        while v.len() < len {
            v.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        v.truncate(len);
        v
    }

    /// `N` random bytes.
    pub fn array<const N: usize>(&mut self) -> [u8; N] {
        let mut a = [0; N];
        a.copy_from_slice(&self.bytes(N));
        a
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

const IN: BufId = BufId(1);
const OUT: BufId = BufId(2);
const KEY_SLOT: u32 = 3;

/// A random case for `op` (`Checksum`, `Encrypt`, `Decrypt`, `Filter` or `Fft`)
/// from `seed`. Inputs are up to 4 KiB, placed at a random offset in the
/// input buffer; the output is written at a random offset in a buffer with
/// room to spare. Expected results come from the CPU reference.
///
/// # Panics
///
/// If `op` has no defined semantics.
pub fn random_case(op: Op, seed: u64) -> Case {
    let mut rng = SplitMix64::new(seed ^ (u64::from(op.code()) << 56));
    let in_off = rng.below(64);
    let out_off = rng.below(64);
    let slack = rng.below(64);

    let mut elem = ElemType::U32;
    let mut log2n = 1u8;
    let (input, cmd_for, out_len, keys) = match op {
        Op::Checksum => {
            let n = rng.below(4097) as usize;
            let input = rng.bytes(n);
            (input, None, 4, Vec::new())
        }
        Op::Encrypt | Op::Decrypt => {
            let n = rng.below(4097) as usize;
            let input = rng.bytes(n);
            let key = rng.array::<32>();
            let mut iv = rng.array::<16>();
            // Exercise counter carries and the 2^128 wrap often.
            match rng.below(4) {
                0 => iv[8..].fill(0xFF),
                1 => iv.fill(0xFF),
                _ => {}
            }
            let n = input.len() as u64;
            (input, Some(iv), n, vec![(KEY_SLOT, key)])
        }
        Op::Filter => {
            elem = rng.pick(&ElemType::ALL);
            let rows = rng.below(1025) as usize;
            let input = column(&mut rng, elem, rows);
            (input, None, rows as u64 * 4, Vec::new())
        }
        Op::Fft => {
            log2n = 1 + rng.below(12) as u8;
            let frames = 1 + rng.below(3) as usize;
            let input = fft_samples(&mut rng, frames << log2n);
            let out = crate::cpu::fft::output_len(log2n, frames) as u64;
            (input, None, out, Vec::new())
        }
        _ => panic!("random_case: {op:?} has no defined semantics"),
    };

    let in_len = input.len() as u64;
    let mut in_buf = rng.bytes(in_off as usize);
    in_buf.extend_from_slice(&input);
    let tail = rng.below(16) as usize;
    in_buf.extend(rng.bytes(tail));
    let out_cap = out_len + slack;
    let out_buf = vec![0u8; (out_off + out_cap) as usize];

    let src = Extent::host(IN, in_off, in_len);
    let dst = Extent::host(OUT, out_off, out_cap);
    let cmd = match op {
        Op::Checksum => ChannelCmd::checksum(src, dst),
        Op::Encrypt => ChannelCmd::encrypt(src, dst, KEY_SLOT, cmd_for.unwrap_or_default()),
        Op::Decrypt => ChannelCmd::decrypt(src, dst, KEY_SLOT, cmd_for.unwrap_or_default()),
        Op::Fft => ChannelCmd::fft(src, dst, log2n),
        _ => ChannelCmd::filter(src, dst, filter_spec(&mut rng, elem, &input)),
    };
    let program = ChannelProgram::from_commands(&[cmd]).expect("random program is valid");
    let host = vec![(IN, in_buf), (OUT, out_buf)];
    let expect = expect_from_cpu(&program, &host, &[], &keys, OUT, out_off);
    Case {
        name: format!("random/{}-{seed}", format!("{op:?}").to_lowercase()),
        description: format!("random {op:?} case from seed {seed}"),
        program,
        host,
        vram: Vec::new(),
        keys,
        expect,
    }
}

/// A column of `rows` values of `elem`, mixing uniform bits with values from
/// a small pool so that equality matches happen, plus float specials.
pub(super) fn column(rng: &mut SplitMix64, elem: ElemType, rows: usize) -> Vec<u8> {
    let pool: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
    let mut out = Vec::with_capacity(rows * elem.width());
    for _ in 0..rows {
        let bits = match rng.below(8) {
            0..=3 => rng.next_u64(),
            4..=6 => pool[rng.below(8) as usize],
            _ => special(rng, elem),
        };
        out.extend_from_slice(&bits.to_le_bytes()[..elem.width()]);
    }
    out
}

fn special(rng: &mut SplitMix64, elem: ElemType) -> u64 {
    let f32s = [
        0.0f32.to_bits(),
        (-0.0f32).to_bits(),
        f32::NAN.to_bits(),
        0xFFC0_0001, // negative NaN with payload
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        1, // smallest subnormal
        f32::MAX.to_bits(),
    ];
    let f64s = [
        0.0f64.to_bits(),
        (-0.0f64).to_bits(),
        f64::NAN.to_bits(),
        0xFFF8_0000_0000_0001,
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        1,
        f64::MAX.to_bits(),
    ];
    let ints = [
        0,
        1,
        u64::MAX,
        i64::MIN as u64,
        i64::MAX as u64,
        0x8000_0000,
        0x7FFF_FFFF,
        0xFFFF_FFFF,
    ];
    match elem {
        ElemType::F32 => u64::from(rng.pick(&f32s)),
        ElemType::F64 => rng.pick(&f64s),
        _ => rng.pick(&ints),
    }
}

/// A filter spec over `elem` whose operands are often values present in
/// `column`, sometimes random or special.
pub(super) fn filter_spec(rng: &mut SplitMix64, elem: ElemType, column: &[u8]) -> FilterSpec {
    let w = elem.width();
    let rows = column.len() / w;
    let operand = |rng: &mut SplitMix64| -> u64 {
        if rows > 0 && rng.below(4) != 0 {
            let i = rng.below(rows as u64) as usize;
            let mut raw = [0u8; 8];
            raw[..w].copy_from_slice(&column[i * w..(i + 1) * w]);
            u64::from_le_bytes(raw)
        } else if rng.below(2) == 0 {
            special(rng, elem)
        } else {
            rng.next_u64()
        }
    };
    let mask = if w == 4 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    };
    let a = operand(rng) & mask;
    let b = operand(rng) & mask;
    let cmp = rng.pick(&CmpOp::ALL);
    FilterSpec {
        elem,
        cmp,
        output: crate::abi::FilterOutput::Indices,
        a,
        b: if cmp == CmpOp::Between { b } else { 0 },
    }
}

/// `count` little-endian `i32` samples for `Fft`: a mix of in-range values,
/// out-of-range values that saturate, the limits, small values and tones.
pub(super) fn fft_samples(rng: &mut SplitMix64, count: usize) -> Vec<u8> {
    let limit = i64::from(crate::cpu::fft::SAMPLE_LIMIT);
    let style = rng.below(4);
    let period = 2 + rng.below(61) as i64;
    let amp = 1 + rng.below(limit as u64) as i64;
    let mut out = Vec::with_capacity(count * 4);
    for t in 0..count as i64 {
        let v: i64 = match style {
            0 => rng.below(2 * limit as u64 + 1) as i64 - limit,
            1 => rng.next_u64() as u32 as i32 as i64,
            2 => {
                // Square wave: integer-only, so it is the same everywhere.
                if (t / period) % 2 == 0 { amp } else { -amp }
            }
            _ => match rng.below(5) {
                0 => limit,
                1 => -limit,
                2 => i64::from(i32::MAX),
                3 => i64::from(i32::MIN),
                _ => rng.below(2001) as i64 - 1000,
            },
        };
        out.extend_from_slice(&(v as i32).to_le_bytes());
    }
    out
}
