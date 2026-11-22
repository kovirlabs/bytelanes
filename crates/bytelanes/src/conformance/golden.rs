//! The golden case set written to `vectors/`.
//!
//! Inputs are generated from fixed seeds; expected results come from the CPU
//! reference. Known-answer cases (`kat-*`) use published inputs whose
//! outputs are also pinned by hand-written constants in this crate's tests,
//! so the reference is checked against something other than itself.

use super::random::{SplitMix64, column};
use super::{Case, expect_from_cpu};
use crate::abi::zerocopy::{FromBytes, IntoBytes};
use crate::abi::{
    BufId, ChannelCmd, ChannelProgram, CmpOp, ElemType, Extent, FilterSpec, Flags, KeySlot, Op,
    PARAMS_LEN,
};

const IN: BufId = BufId(1);
const OUT: BufId = BufId(2);
const V1: BufId = BufId(16);
const V2: BufId = BufId(17);
const SLOT: KeySlot = 0;

/// NIST SP 800-38A F.5.5 key (CTR-AES256).
pub const NIST_KEY: [u8; 32] = [
    0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
    0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4,
];
/// NIST SP 800-38A F.5.5 initial counter block.
pub const NIST_IV: [u8; 16] = [
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];
/// NIST SP 800-38A F.5.5 plaintext (four blocks).
pub const NIST_PLAIN: [u8; 64] = [
    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
    0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c, 0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
    0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11, 0xe5, 0xfb, 0xc1, 0x19, 0x1a, 0x0a, 0x52, 0xef,
    0xf6, 0x9f, 0x24, 0x45, 0xdf, 0x4f, 0x9b, 0x17, 0xad, 0x2b, 0x41, 0x7b, 0xe6, 0x6c, 0x37, 0x10,
];
/// NIST SP 800-38A F.5.5 ciphertext.
pub const NIST_CIPHER: [u8; 64] = [
    0x60, 0x1e, 0xc3, 0x13, 0x77, 0x57, 0x89, 0xa5, 0xb7, 0xa7, 0xf5, 0x04, 0xbb, 0xf3, 0xd2, 0x28,
    0xf4, 0x43, 0xe3, 0xca, 0x4d, 0x62, 0xb5, 0x9a, 0xca, 0x84, 0xe9, 0x90, 0xca, 0xca, 0xf5, 0xc5,
    0x2b, 0x09, 0x30, 0xda, 0xa2, 0x3d, 0xe9, 0x4c, 0xe8, 0x70, 0x17, 0xba, 0x2d, 0x84, 0x98, 0x8d,
    0xdf, 0xc9, 0xc5, 0x8d, 0xb6, 0x7a, 0xad, 0xa6, 0x13, 0xc2, 0xdd, 0x08, 0x45, 0x79, 0x41, 0xa6,
];

/// CRC-32C known answers: the standard check input and the RFC 3720 B.4
/// iSCSI examples.
pub fn crc_kats() -> Vec<(&'static str, Vec<u8>, u32)> {
    vec![
        ("check-123456789", b"123456789".to_vec(), 0xE306_9283),
        ("rfc3720-zeros", vec![0; 32], 0x8A91_36AA),
        ("rfc3720-ones", vec![0xFF; 32], 0x62A8_AB43),
        ("rfc3720-incrementing", (0..32).collect(), 0x46DD_794E),
        ("rfc3720-decrementing", (0..32).rev().collect(), 0x113F_DB5C),
    ]
}

struct Builder {
    cases: Vec<Case>,
}

impl Builder {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        name: &str,
        description: &str,
        program: ChannelProgram,
        host: Vec<(BufId, Vec<u8>)>,
        vram: Vec<(BufId, u64)>,
        keys: Vec<(KeySlot, [u8; 32])>,
        output_offset: u64,
    ) {
        let expect = expect_from_cpu(&program, &host, &vram, &keys, OUT, output_offset);
        self.cases.push(Case {
            name: name.to_string(),
            description: description.to_string(),
            program,
            host,
            vram,
            keys,
            expect,
        });
    }

    /// One command, `input` in `IN` at offset 0, `out_cap` bytes of `OUT`.
    fn single(
        &mut self,
        name: &str,
        description: &str,
        cmd: impl Fn(Extent, Extent) -> ChannelCmd,
        input: Vec<u8>,
        out_cap: u64,
        keys: Vec<(KeySlot, [u8; 32])>,
    ) {
        let src = Extent::host(IN, 0, input.len() as u64);
        let dst = Extent::host(OUT, 0, out_cap);
        let program = program(&[cmd(src, dst)]);
        let host = vec![(IN, input), (OUT, vec![0; out_cap as usize])];
        self.add(name, description, program, host, Vec::new(), keys, 0);
    }
}

fn program(cmds: &[ChannelCmd]) -> ChannelProgram {
    ChannelProgram::from_commands(cmds).expect("golden program is valid")
}

/// Build a program without validating it, for invalid-program cases.
fn raw_program(cmds: &[ChannelCmd]) -> ChannelProgram {
    let mut p = ChannelProgram::new();
    for c in cmds {
        p.push(*c).expect("at most 16 commands");
    }
    p
}

const SIZES: [usize; 21] = [
    0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 4095, 4096, 4097,
];
const LARGE: usize = 262_144 + 5;

/// The full golden set, in a fixed order.
pub fn golden_cases() -> Vec<Case> {
    let mut b = Builder { cases: Vec::new() };
    let mut rng = SplitMix64::new(0x6279_7465_6c61_6e65);
    checksum_cases(&mut b, &mut rng);
    crypt_cases(&mut b, &mut rng);
    filter_cases(&mut b, &mut rng);
    program_cases(&mut b, &mut rng);
    // Added in M7 with its own seed, so the M1 cases above keep their bytes.
    fft_cases(&mut b, &mut SplitMix64::new(0x6666_745f_6d37));
    b.cases
}

fn checksum_cases(b: &mut Builder, rng: &mut SplitMix64) {
    for (name, input, _) in crc_kats() {
        b.single(
            &format!("checksum/kat-{name}"),
            "CRC-32C known answer",
            ChannelCmd::checksum,
            input,
            4,
            vec![],
        );
    }
    for len in SIZES.iter().copied().chain([LARGE]) {
        b.single(
            &format!("checksum/len-{len}"),
            "CRC-32C of random bytes",
            ChannelCmd::checksum,
            rng.bytes(len),
            4,
            vec![],
        );
    }
    // Input at an odd offset, output at an odd offset in a larger buffer.
    let data = rng.bytes(1000);
    let program = program(&[ChannelCmd::checksum(
        Extent::host(IN, 13, 977),
        Extent::host(OUT, 5, 9),
    )]);
    b.add(
        "checksum/offsets",
        "src and dst at unaligned offsets; dst larger than the 4-byte result",
        program,
        vec![(IN, data), (OUT, vec![0; 16])],
        vec![],
        vec![],
        5,
    );
    b.single(
        "checksum/dst-too-small",
        "3-byte dst: DstTooSmall, bytes = 4",
        ChannelCmd::checksum,
        rng.bytes(10),
        3,
        vec![],
    );
}

fn crypt_cases(b: &mut Builder, rng: &mut SplitMix64) {
    let nist = vec![(SLOT, NIST_KEY)];
    b.single(
        "crypt/kat-nist-f55-encrypt",
        "NIST SP 800-38A F.5.5 CTR-AES256 encrypt",
        |s, d| ChannelCmd::encrypt(s, d, SLOT, NIST_IV),
        NIST_PLAIN.to_vec(),
        64,
        nist.clone(),
    );
    b.single(
        "crypt/kat-nist-f56-decrypt",
        "NIST SP 800-38A F.5.6 CTR-AES256 decrypt",
        |s, d| ChannelCmd::decrypt(s, d, SLOT, NIST_IV),
        NIST_CIPHER.to_vec(),
        64,
        nist,
    );
    for len in SIZES.iter().copied().chain([1000, LARGE]) {
        let key = rng.array::<32>();
        let iv = rng.array::<16>();
        b.single(
            &format!("crypt/decrypt-len-{len}"),
            "AES-256-CTR decrypt of random bytes",
            |s, d| ChannelCmd::decrypt(s, d, SLOT, iv),
            rng.bytes(len),
            len as u64,
            vec![(SLOT, key)],
        );
    }
    let key = rng.array::<32>();
    let iv = rng.array::<16>();
    b.single(
        "crypt/encrypt-len-100",
        "Encrypt is the same transform as Decrypt",
        |s, d| ChannelCmd::encrypt(s, d, SLOT, iv),
        rng.bytes(100),
        100,
        vec![(SLOT, key)],
    );
    // Counter carries: low 64 bits about to wrap (catches 64-bit counters),
    // and the full 128-bit wrap to zero.
    let mut carry64 = [0u8; 16];
    carry64[..8].copy_from_slice(&rng.array::<8>());
    carry64[8..].copy_from_slice(&(u64::MAX - 1).to_be_bytes());
    let mut wrap128 = [0xFF; 16];
    wrap128[15] = 0xFE;
    for (name, iv) in [("carry-64", carry64), ("wrap-128", wrap128)] {
        let key = rng.array::<32>();
        b.single(
            &format!("crypt/counter-{name}"),
            "counter increments across a carry boundary, 5 blocks",
            |s, d| ChannelCmd::decrypt(s, d, SLOT, iv),
            rng.bytes(80),
            80,
            vec![(SLOT, key)],
        );
    }
    let key = rng.array::<32>();
    b.single(
        "crypt/unknown-key",
        "key slot 0 is empty (the case loads slot 9): UnknownKey",
        |s, d| ChannelCmd::decrypt(s, d, SLOT, [0; 16]),
        rng.bytes(32),
        32,
        vec![(9, key)],
    );
    let key = rng.array::<32>();
    b.single(
        "crypt/dst-too-small",
        "dst one byte short: DstTooSmall, bytes = 40",
        |s, d| ChannelCmd::decrypt(s, d, SLOT, [0; 16]),
        rng.bytes(40),
        39,
        vec![(SLOT, key)],
    );
}

fn filter_cases(b: &mut Builder, rng: &mut SplitMix64) {
    for elem in ElemType::ALL {
        let ename = format!("{elem:?}").to_lowercase();
        let col = column(rng, elem, 1000);
        for cmp in CmpOp::ALL {
            let spec = super::random::filter_spec(rng, elem, &col);
            let spec = FilterSpec {
                cmp,
                b: if cmp == CmpOp::Between { spec.b } else { 0 },
                ..spec
            };
            let cname = format!("{cmp:?}").to_lowercase();
            b.single(
                &format!("filter/{ename}-{cname}"),
                "1000 rows mixing random values, repeats and specials",
                |s, d| ChannelCmd::filter(s, d, spec),
                col.clone(),
                4000,
                vec![],
            );
        }
    }

    // Float specials, one row each, against hand-picked operands.
    let f32s: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::NAN,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::MAX,
        f32::MIN,
    ];
    let col32: Vec<u8> = f32s.iter().flat_map(|v| v.to_le_bytes()).collect();
    for (name, spec) in [
        ("eq-zero", FilterSpec::f32(CmpOp::Eq, 0.0, 0.0)),
        ("eq-neg-zero", FilterSpec::f32(CmpOp::Eq, -0.0, 0.0)),
        ("ne-nan", FilterSpec::f32(CmpOp::Ne, f32::NAN, 0.0)),
        ("eq-nan", FilterSpec::f32(CmpOp::Eq, f32::NAN, 0.0)),
        ("lt-zero", FilterSpec::f32(CmpOp::Lt, 0.0, 0.0)),
        (
            "ge-neg-inf",
            FilterSpec::f32(CmpOp::Ge, f32::NEG_INFINITY, 0.0),
        ),
        (
            "between-nan-bound",
            FilterSpec::f32(CmpOp::Between, f32::NAN, 1.0),
        ),
        (
            "between-zero-inf",
            FilterSpec::f32(CmpOp::Between, -0.0, f32::INFINITY),
        ),
    ] {
        b.single(
            &format!("filter/f32-special-{name}"),
            "IEEE 754: NaN matches only Ne; -0.0 equals 0.0",
            |s, d| ChannelCmd::filter(s, d, spec),
            col32.clone(),
            40,
            vec![],
        );
    }
    let f64s: Vec<f64> = vec![
        0.0,
        -0.0,
        f64::NAN,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(1),
        f64::MIN,
    ];
    let col64: Vec<u8> = f64s.iter().flat_map(|v| v.to_le_bytes()).collect();
    for (name, spec) in [
        ("eq-neg-zero", FilterSpec::f64(CmpOp::Eq, -0.0, 0.0)),
        ("ne-nan", FilterSpec::f64(CmpOp::Ne, f64::NAN, 0.0)),
        (
            "le-subnormal",
            FilterSpec::f64(CmpOp::Le, f64::from_bits(1), 0.0),
        ),
        (
            "gt-neg-inf",
            FilterSpec::f64(CmpOp::Gt, f64::NEG_INFINITY, 0.0),
        ),
    ] {
        b.single(
            &format!("filter/f64-special-{name}"),
            "IEEE 754: NaN matches only Ne; -0.0 equals 0.0",
            |s, d| ChannelCmd::filter(s, d, spec),
            col64.clone(),
            32,
            vec![],
        );
    }
    // Integer extremes.
    let i64s: Vec<u8> = [i64::MIN, -1, 0, 1, i64::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    b.single(
        "filter/i64-extremes-lt-zero",
        "signed compare at i64 extremes",
        |s, d| ChannelCmd::filter(s, d, FilterSpec::i64(CmpOp::Lt, 0, 0)),
        i64s,
        20,
        vec![],
    );
    let u64s: Vec<u8> = [0u64, 1, 1 << 63, u64::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    b.single(
        "filter/u64-extremes-ge-high-bit",
        "unsigned compare above 2^63 (not signed)",
        |s, d| ChannelCmd::filter(s, d, FilterSpec::u64(CmpOp::Ge, 1 << 63, 0)),
        u64s,
        16,
        vec![],
    );
    let i32s: Vec<u8> = [i32::MIN, -1, 0, i32::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    b.single(
        "filter/i32-between-negative",
        "signed 32-bit Between with negative bounds",
        |s, d| ChannelCmd::filter(s, d, FilterSpec::i32(CmpOp::Between, i32::MIN, -1)),
        i32s,
        16,
        vec![],
    );

    // Sizes and edges.
    let all = FilterSpec::u32(CmpOp::Ge, 0, 0);
    let none = FilterSpec::u32(CmpOp::Lt, 0, 0);
    b.single(
        "filter/empty",
        "zero rows: 0 bytes, CRC 0",
        |s, d| ChannelCmd::filter(s, d, all),
        vec![],
        4,
        vec![],
    );
    b.single(
        "filter/no-match",
        "no row matches",
        |s, d| ChannelCmd::filter(s, d, none),
        rng.bytes(400),
        400,
        vec![],
    );
    b.single(
        "filter/all-match",
        "every row matches",
        |s, d| ChannelCmd::filter(s, d, all),
        rng.bytes(400),
        400,
        vec![],
    );
    let rows = 65_537;
    let col = column(rng, ElemType::U32, rows);
    let spec = FilterSpec::u32(CmpOp::Lt, 1 << 31, 0);
    b.single(
        "filter/u32-large-lt",
        "65,537 rows, about half match",
        |s, d| ChannelCmd::filter(s, d, spec),
        col,
        rows as u64 * 4,
        vec![],
    );
    b.single(
        "filter/invalid-length",
        "10 bytes is not a whole number of u32s: InvalidLength",
        |s, d| ChannelCmd::filter(s, d, all),
        rng.bytes(10),
        40,
        vec![],
    );
    b.single(
        "filter/dst-too-small",
        "every row of 10 matches, dst holds 9: DstTooSmall, bytes = 40",
        |s, d| ChannelCmd::filter(s, d, all),
        rng.bytes(40),
        36,
        vec![],
    );
}

fn i32s(v: impl IntoIterator<Item = i32>) -> Vec<u8> {
    v.into_iter().flat_map(i32::to_le_bytes).collect()
}

fn fft_cases(b: &mut Builder, rng: &mut SplitMix64) {
    use crate::cpu::fft::{SAMPLE_LIMIT, output_len};
    let one =
        |b: &mut Builder, name: &str, desc: &str, log2n: u8, input: Vec<u8>, frames: usize| {
            b.single(
                &format!("fft/{name}"),
                desc,
                |s, d| ChannelCmd::fft(s, d, log2n),
                input,
                output_len(log2n, frames) as u64,
                vec![],
            );
        };
    for log2n in [1u8, 4, 10] {
        let n = 1usize << log2n;
        let mut x = vec![0; n];
        x[0] = SAMPLE_LIMIT;
        one(
            b,
            &format!("kat-impulse-n{n}"),
            "impulse: every bin is (2^30 / N, 0)",
            log2n,
            i32s(x),
            1,
        );
        one(
            b,
            &format!("kat-dc-n{n}"),
            "constant: bin 0 holds it, the rest are 0",
            log2n,
            i32s(vec![-987_654_321; n]),
            1,
        );
        let alt = (0..n).map(|i| if i % 2 == 0 { 55_555 } else { -55_555 });
        one(
            b,
            &format!("kat-nyquist-n{n}"),
            "alternating: bin N/2 holds it",
            log2n,
            i32s(alt),
            1,
        );
    }
    for log2n in 1..=16u8 {
        let n = 1usize << log2n;
        let frames = if log2n <= 12 { 2 } else { 1 };
        let input = super::random::fft_samples(rng, frames * n);
        one(
            b,
            &format!("random-log2n-{log2n}"),
            "random, saturating, square-wave or extreme samples",
            log2n,
            input,
            frames,
        );
    }
    let n = 1024usize;
    let square = (0..n as i32).map(|t| {
        if (t / 16) % 2 == 0 {
            SAMPLE_LIMIT
        } else {
            -SAMPLE_LIMIT
        }
    });
    one(
        b,
        "square-full-scale",
        "full-scale square wave, odd harmonics, largest butterfly growth",
        10,
        i32s(square),
        1,
    );
    let sat = (0..64).map(|i| match i % 4 {
        0 => i32::MAX,
        1 => i32::MIN,
        2 => SAMPLE_LIMIT + 1,
        _ => -SAMPLE_LIMIT - 1,
    });
    one(
        b,
        "saturation",
        "values beyond ±2^30 are clamped before the transform",
        6,
        i32s(sat),
        1,
    );
    one(b, "empty", "no frames: 0 bytes, CRC 0", 8, vec![], 1);
    b.single(
        "fft/invalid-length",
        "60 bytes is not a whole frame of 16 samples: InvalidLength",
        |s, d| ChannelCmd::fft(s, d, 4),
        vec![1; 60],
        200,
        vec![],
    );
    // Decrypt a frame of samples on the device, transform it, checksum the bins.
    let n = 256u64;
    let key = rng.array::<32>();
    let iv = rng.array::<16>();
    let keys = vec![(SLOT, key)];
    let mut cipher = vec![0; (n * 4) as usize];
    let plain = super::random::fft_samples(rng, n as usize);
    crate::cpu::ops::aes256_ctr(&key, &iv, &plain, &mut cipher);
    let v1 = Extent::vram(V1, 0, n * 4);
    let v2 = Extent::vram(V2, 0, crate::cpu::fft::output_len(8, 1) as u64);
    let p = program(&[
        ChannelCmd::decrypt(Extent::host(IN, 0, n * 4), v1, SLOT, iv).chained(),
        ChannelCmd::fft(v1, v2, 8).chained(),
        ChannelCmd::checksum(v2, Extent::host(OUT, 0, 4)),
    ]);
    b.add(
        "program/decrypt-fft-checksum",
        "Host -> Decrypt -> Vram -> Fft -> Vram -> Checksum -> Host",
        p,
        vec![(IN, cipher), (OUT, vec![0; 4])],
        vec![(V1, n * 4), (V2, 2048)],
        keys.clone(),
        0,
    );

    b.single(
        "fft/dst-too-small",
        "one frame of 8 needs 5 bins (40 bytes); dst holds 32: DstTooSmall",
        |s, d| ChannelCmd::fft(s, d, 3),
        i32s(1..=8),
        32,
        vec![],
    );
}

fn program_cases(b: &mut Builder, rng: &mut SplitMix64) {
    let key = rng.array::<32>();
    let iv = rng.array::<16>();
    let keys = vec![(SLOT, key)];

    // Decrypt a column into Vram, then filter it to host.
    let rows = 512;
    let mut plain = column(rng, ElemType::I32, rows);
    let mut cipher = vec![0; plain.len()];
    crate::cpu::ops::aes256_ctr(&key, &iv, &plain, &mut cipher);
    plain.clear();
    let n = cipher.len() as u64;
    let v = Extent::vram(V1, 0, n);
    let p = program(&[
        ChannelCmd::decrypt(Extent::host(IN, 0, n), v, SLOT, iv).chained(),
        ChannelCmd::filter(
            v,
            Extent::host(OUT, 0, rows as u64 * 4),
            FilterSpec::i32(CmpOp::Gt, 0, 0),
        ),
    ]);
    b.add(
        "program/decrypt-then-filter",
        "Host -> Decrypt -> Vram -> Filter -> Host",
        p,
        vec![(IN, cipher), (OUT, vec![0; rows * 4])],
        vec![(V1, n)],
        keys.clone(),
        0,
    );

    // Three-stage chain: filter -> decrypt? Filter output is indices; use
    // decrypt -> decrypt -> checksum to cover a longer chain.
    let data = rng.bytes(300);
    let v1 = Extent::vram(V1, 0, 300);
    let v2 = Extent::vram(V2, 8, 300);
    let p = program(&[
        ChannelCmd::decrypt(Extent::host(IN, 0, 300), v1, SLOT, iv).chained(),
        ChannelCmd::encrypt(v1, v2, SLOT, iv).chained(),
        ChannelCmd::checksum(v2, Extent::host(OUT, 0, 4)),
    ]);
    b.add(
        "program/three-stage-chain",
        "Decrypt then Encrypt with the same key and IV restores the input; checksum of it",
        p,
        vec![(IN, data), (OUT, vec![0; 4])],
        vec![(V1, 300), (V2, 400)],
        keys.clone(),
        0,
    );

    // Chained input is the previous result, not the whole extent: the Vram
    // extent is 4096 bytes but the filter result is shorter.
    let col = column(rng, ElemType::U32, 64);
    let v = Extent::vram(V1, 0, 4096);
    let p = program(&[
        ChannelCmd::filter(
            Extent::host(IN, 0, 256),
            v,
            FilterSpec::u32(CmpOp::Ne, 0, 0),
        )
        .chained(),
        ChannelCmd::checksum(v, Extent::host(OUT, 0, 4)),
    ]);
    b.add(
        "program/chain-uses-result-length",
        "a chained src is the previous result's bytes, not the full extent",
        p,
        vec![(IN, col), (OUT, vec![0; 4])],
        vec![(V1, 4096)],
        vec![],
        0,
    );

    // Unchained steps: an independent checksum into Vram, then a decrypt to host.
    let data = rng.bytes(64);
    let p = program(&[
        ChannelCmd::checksum(Extent::host(IN, 0, 64), Extent::vram(V1, 0, 4)),
        ChannelCmd::decrypt(Extent::host(IN, 0, 64), Extent::host(OUT, 0, 64), SLOT, iv),
    ]);
    b.add(
        "program/unchained-steps",
        "commands without CHAIN run in order; the completion reports the last",
        p,
        vec![(IN, data), (OUT, vec![0; 64])],
        vec![(V1, 4)],
        keys.clone(),
        0,
    );

    // Failure in the second command reports index 1.
    let p = program(&[
        ChannelCmd::decrypt(Extent::host(IN, 0, 64), Extent::vram(V1, 0, 64), SLOT, iv).chained(),
        ChannelCmd::filter(
            Extent::vram(V1, 0, 64),
            Extent::host(OUT, 0, 4),
            FilterSpec::u32(CmpOp::Ge, 0, 0),
        ),
    ]);
    b.add(
        "program/fail-second-command",
        "second command's dst holds 1 of 16 indices: DstTooSmall at index 1",
        p,
        vec![(IN, rng.bytes(64)), (OUT, vec![0; 4])],
        vec![(V1, 64)],
        keys.clone(),
        0,
    );

    // Error precedence and buffer checks.
    let one = |name: &str,
               desc: &str,
               cmd: ChannelCmd,
               host: Vec<(BufId, Vec<u8>)>,
               keys: Vec<(KeySlot, [u8; 32])>,
               b: &mut Builder| {
        b.add(name, desc, program(&[cmd]), host, vec![], keys, 0);
    };
    one(
        "program/unknown-src-buffer",
        "src names buffer 99: UnknownBuffer",
        ChannelCmd::checksum(Extent::host(BufId(99), 0, 4), Extent::host(OUT, 0, 4)),
        vec![(OUT, vec![0; 4])],
        vec![],
        b,
    );
    one(
        "program/unknown-dst-buffer",
        "dst names buffer 99: UnknownBuffer",
        ChannelCmd::checksum(Extent::host(IN, 0, 4), Extent::host(BufId(99), 0, 4)),
        vec![(IN, vec![1; 4])],
        vec![],
        b,
    );
    one(
        "program/src-out-of-bounds",
        "src reaches one byte past its buffer: OutOfBounds",
        ChannelCmd::checksum(Extent::host(IN, 1, 16), Extent::host(OUT, 0, 4)),
        vec![(IN, vec![1; 16]), (OUT, vec![0; 4])],
        vec![],
        b,
    );
    one(
        "program/dst-out-of-bounds",
        "dst extent reaches past its buffer: OutOfBounds even though the result would fit",
        ChannelCmd::checksum(Extent::host(IN, 0, 16), Extent::host(OUT, 0, 8)),
        vec![(IN, vec![1; 16]), (OUT, vec![0; 4])],
        vec![],
        b,
    );
    one(
        "program/precedence-buffer-before-key",
        "unknown src buffer and empty key slot: UnknownBuffer wins",
        ChannelCmd::decrypt(
            Extent::host(BufId(99), 0, 16),
            Extent::host(OUT, 0, 16),
            5,
            [0; 16],
        ),
        vec![(OUT, vec![0; 16])],
        vec![],
        b,
    );
    one(
        "program/precedence-key-before-size",
        "empty key slot and short dst: UnknownKey wins",
        ChannelCmd::decrypt(Extent::host(IN, 0, 16), Extent::host(OUT, 0, 8), 5, [0; 16]),
        vec![(IN, vec![1; 16]), (OUT, vec![0; 8])],
        vec![],
        b,
    );
    one(
        "program/reserved-op",
        "Compress has no semantics yet: UnsupportedOp",
        ChannelCmd::new(
            Op::Compress,
            Flags::NONE,
            Extent::host(IN, 0, 16),
            Extent::host(OUT, 0, 16),
            [0; PARAMS_LEN],
        ),
        vec![(IN, vec![1; 16]), (OUT, vec![0; 16])],
        vec![],
        b,
    );

    // Invalid programs, submitted unvalidated.
    let inval = |name: &str, desc: &str, p: ChannelProgram, b: &mut Builder| {
        b.add(
            name,
            desc,
            p,
            vec![(IN, vec![1; 64]), (OUT, vec![0; 64])],
            vec![],
            vec![],
            0,
        );
    };
    inval(
        "invalid/overlap",
        "dst overlaps src in the same buffer: InvalidProgram at 0",
        raw_program(&[ChannelCmd::checksum(
            Extent::host(IN, 0, 64),
            Extent::host(IN, 60, 4),
        )]),
        b,
    );
    inval(
        "invalid/intermediate-host-dst",
        "first of two commands writes to Host: InvalidProgram at 0",
        raw_program(&[
            ChannelCmd::checksum(Extent::host(IN, 0, 8), Extent::host(OUT, 0, 4)),
            ChannelCmd::checksum(Extent::host(IN, 8, 8), Extent::host(OUT, 4, 4)),
        ]),
        b,
    );
    inval(
        "invalid/empty",
        "no commands: InvalidProgram at NO_INDEX",
        ChannelProgram::new(),
        b,
    );
    let mut bad = raw_program(&[ChannelCmd::checksum(
        Extent::host(IN, 0, 8),
        Extent::host(OUT, 0, 4),
    )]);
    let mut bytes = bad.as_bytes().to_vec();
    bytes[4] = 1; // program reserved byte
    bad = ChannelProgram::read_from_bytes(&bytes).expect("same length");
    inval(
        "invalid/reserved-byte",
        "non-zero reserved header byte: InvalidProgram at NO_INDEX",
        bad,
        b,
    );
    let mut bytes = raw_program(&[ChannelCmd::checksum(
        Extent::host(IN, 0, 8),
        Extent::host(OUT, 0, 4),
    )])
    .as_bytes()
    .to_vec();
    bytes[8] = 0xEE; // op code low byte of command 0
    let bad = ChannelProgram::read_from_bytes(&bytes).expect("same length");
    inval(
        "invalid/unknown-op",
        "op code 0x00EE: InvalidProgram at 0",
        bad,
        b,
    );
}
