//! Property tests: 10,000 cases per op (8 under miri).
//!
//! Each op is checked against a deliberately naive, independent
//! implementation, and the conformance harness is run over 10,000 seeded
//! random cases per op on the CPU backend.

use aes::Aes256;
use aes::cipher::{BlockCipherEncrypt, KeyInit};
use bytelanes::abi::{
    BufId, ChannelCmd, ChannelProgram, CmpOp, Completion, ElemType, Extent, FilterOutput,
    FilterSpec, Op, PathState, Status, Tag,
};
use bytelanes::conformance::{random_case, run_case};
use bytelanes::cpu::{CpuBackend, ops};
use bytelanes::{Buffers, Busy, ChannelBackend, verify};
use proptest::prelude::*;
use std::cmp::Ordering;

fn cases() -> u32 {
    if cfg!(miri) { 8 } else { 10_000 }
}

fn cfg() -> ProptestConfig {
    ProptestConfig {
        cases: cases(),
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

/// Bit-at-a-time CRC-32C, straight from the definition.
fn naive_crc32c(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x82F6_3B78
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// CTR from the block cipher: encrypt each counter block, XOR.
fn naive_ctr(key: &[u8; 32], iv: &[u8; 16], src: &[u8]) -> Vec<u8> {
    let cipher = Aes256::new(key.into());
    let mut ctr = u128::from_be_bytes(*iv);
    let mut out = Vec::with_capacity(src.len());
    for chunk in src.chunks(16) {
        let mut block = ctr.to_be_bytes().into();
        cipher.encrypt_block(&mut block);
        out.extend(chunk.iter().zip(block.iter()).map(|(a, b)| a ^ b));
        ctr = ctr.wrapping_add(1);
    }
    out
}

#[derive(Clone, Copy)]
enum Num {
    U(u64),
    I(i64),
    F(f64),
}

fn num(elem: ElemType, bits: u64) -> Num {
    match elem {
        ElemType::U32 | ElemType::U64 => Num::U(bits),
        ElemType::I32 => Num::I(i64::from(bits as u32 as i32)),
        ElemType::I64 => Num::I(bits as i64),
        // f32 -> f64 is exact and keeps NaN, zeros and order.
        ElemType::F32 => Num::F(f64::from(f32::from_bits(bits as u32))),
        ElemType::F64 => Num::F(f64::from_bits(bits)),
    }
}

fn ord(x: Num, y: Num) -> Option<Ordering> {
    match (x, y) {
        (Num::U(a), Num::U(b)) => Some(a.cmp(&b)),
        (Num::I(a), Num::I(b)) => Some(a.cmp(&b)),
        (Num::F(a), Num::F(b)) => a.partial_cmp(&b),
        _ => unreachable!(),
    }
}

fn naive_filter(spec: &FilterSpec, src: &[u8]) -> Vec<u32> {
    use Ordering::*;
    let w = spec.elem.width();
    let (a, b) = (num(spec.elem, spec.a), num(spec.elem, spec.b));
    let mut out = Vec::new();
    for i in 0..src.len() / w {
        let mut raw = [0u8; 8];
        raw[..w].copy_from_slice(&src[i * w..(i + 1) * w]);
        let x = num(spec.elem, u64::from_le_bytes(raw));
        let hit = match spec.cmp {
            CmpOp::Eq => ord(x, a) == Some(Equal),
            CmpOp::Ne => ord(x, a) != Some(Equal),
            CmpOp::Lt => ord(x, a) == Some(Less),
            CmpOp::Le => matches!(ord(x, a), Some(Less | Equal)),
            CmpOp::Gt => ord(x, a) == Some(Greater),
            CmpOp::Ge => matches!(ord(x, a), Some(Greater | Equal)),
            CmpOp::Between => {
                matches!(ord(a, x), Some(Less | Equal)) && matches!(ord(x, b), Some(Less | Equal))
            }
        };
        if hit {
            out.push(i as u32);
        }
    }
    out
}

fn elem_strategy() -> impl Strategy<Value = ElemType> {
    prop::sample::select(ElemType::ALL.to_vec())
}

fn cmp_strategy() -> impl Strategy<Value = CmpOp> {
    prop::sample::select(CmpOp::ALL.to_vec())
}

/// Bits biased towards specials so NaN, zeros and extremes come up often.
fn bits_strategy() -> impl Strategy<Value = u64> {
    prop_oneof![
        any::<u64>(),
        prop::sample::select(vec![
            0u64,
            1,
            u64::MAX,
            1 << 63,
            0x8000_0000,
            0x7FC0_0000,
            0xFFC0_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FF8_0000_0000_0000,
            0xFFF0_0000_0000_0000,
            0x7FF0_0000_0000_0000,
        ]),
    ]
}

proptest! {
    #![proptest_config(cfg())]

    #[test]
    fn checksum_matches_naive(data in prop::collection::vec(any::<u8>(), 0..2048)) {
        prop_assert_eq!(ops::checksum(&data), naive_crc32c(&data));
    }

    #[test]
    fn ctr_matches_naive(
        key in any::<[u8; 32]>(),
        iv in prop_oneof![any::<[u8; 16]>(), Just([0xFF; 16]), any::<[u8; 8]>().prop_map(|h| {
            let mut iv = [0xFF; 16];
            iv[..8].copy_from_slice(&h);
            iv
        })],
        data in prop::collection::vec(any::<u8>(), 0..1024),
    ) {
        let mut out = vec![0; data.len()];
        ops::aes256_ctr(&key, &iv, &data, &mut out);
        prop_assert_eq!(&out, &naive_ctr(&key, &iv, &data));
        ops::aes256_ctr_in_place(&key, &iv, &mut out);
        prop_assert_eq!(out, data);
    }

    #[test]
    fn filter_matches_naive(
        elem in elem_strategy(),
        cmp in cmp_strategy(),
        a in bits_strategy(),
        b in bits_strategy(),
        values in prop::collection::vec(bits_strategy(), 0..512),
        pick in any::<prop::sample::Index>(),
    ) {
        let w = elem.width();
        let mask = if w == 4 { u64::from(u32::MAX) } else { u64::MAX };
        // Often compare against a value that is in the column.
        let a = if !values.is_empty() && a % 2 == 0 { values[pick.index(values.len())] } else { a } & mask;
        let spec = FilterSpec { elem, cmp, output: FilterOutput::Indices, a, b: if cmp == CmpOp::Between { b & mask } else { 0 } };
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()[..w].to_vec()).collect();
        let mut got = Vec::new();
        ops::filter(&spec, &src, &mut got).unwrap();
        prop_assert_eq!(got, naive_filter(&spec, &src));
    }

    #[test]
    fn filter_rejects_partial_elements(elem in elem_strategy(), extra in 1usize..8, rows in 0usize..16) {
        let w = elem.width();
        prop_assume!(extra % w != 0);
        let spec = FilterSpec { elem, cmp: CmpOp::Eq, output: FilterOutput::Indices, a: 0, b: 0 };
        let src = vec![0u8; rows * w + extra];
        let mut out = Vec::new();
        prop_assert_eq!(ops::filter(&spec, &src, &mut out), Err(Status::InvalidLength));
        prop_assert!(out.is_empty());
    }
}

fn run_random(op: Op) {
    let backend = CpuBackend::new();
    for seed in 0..u64::from(cases()) {
        let case = random_case(op, seed);
        if let Err(m) = run_case(&backend, &case) {
            panic!("{op:?} seed {seed}: {m}");
        }
    }
}

#[test]
fn harness_random_checksum() {
    run_random(Op::Checksum);
}

#[test]
fn harness_random_decrypt() {
    run_random(Op::Decrypt);
}

#[test]
fn harness_random_encrypt() {
    run_random(Op::Encrypt);
}

#[test]
fn harness_random_filter() {
    run_random(Op::Filter);
}

#[test]
fn random_cases_are_deterministic() {
    for op in Op::DEFINED {
        assert_eq!(random_case(op, 42), random_case(op, 42));
        assert_ne!(random_case(op, 42).program, random_case(op, 43).program);
    }
}

#[test]
fn backend_queue_poll_and_health() {
    let b = CpuBackend::with_queue_depth(2);
    b.put_host(BufId(1), vec![7; 8]).unwrap();
    b.put_host(BufId(2), vec![0; 4]).unwrap();
    let p = ChannelProgram::from_commands(&[ChannelCmd::checksum(
        Extent::host(BufId(1), 0, 8),
        Extent::host(BufId(2), 0, 4),
    )])
    .unwrap();
    assert!(b.capabilities().supports(Op::Checksum));
    assert!(!b.capabilities().supports(Op::Read));
    b.submit(&p, Tag(1)).unwrap();
    b.submit(&p, Tag(2)).unwrap();
    assert_eq!(b.submit(&p, Tag(3)), Err(Busy));
    let mut out = [Completion::error(Tag(0), 0, Status::DeviceError, 0); 4];
    assert_eq!(b.poll(&mut out), 2);
    assert_eq!((out[0].tag(), out[1].tag()), (Tag(1), Tag(2)));
    assert_eq!(b.poll(&mut out), 0);
    let result = b.get_host(BufId(2)).unwrap();
    assert_eq!(result, ops::checksum(&[7; 8]).to_le_bytes());
    verify(&out[0], &result).unwrap();
    assert_eq!(b.health(), PathState::Online);
    b.set_health(PathState::Offline);
    assert_eq!(b.health(), PathState::Offline);
}

#[test]
fn verify_catches_corruption() {
    let data = b"result bytes".to_vec();
    let c = Completion::ok(Tag(1), 0, data.len() as u32, ops::checksum(&data));
    verify(&c, &data).unwrap();
    let mut bad = data.clone();
    bad[3] ^= 1;
    assert_eq!(verify(&c, &bad), Err(Status::CrcMismatch));
    assert_eq!(verify(&c, &data[1..]), Err(Status::CrcMismatch));
    let e = Completion::error(Tag(1), 0, Status::UnknownKey, 0);
    assert_eq!(verify(&e, &[]), Err(Status::UnknownKey));
}

#[test]
fn nvme_extent_is_unsupported_on_cpu() {
    let b = CpuBackend::new();
    b.put_host(BufId(2), vec![0; 4]).unwrap();
    let p = ChannelProgram::from_commands(&[ChannelCmd::checksum(
        Extent::nvme(bytelanes::abi::DevId(0), 0, 1),
        Extent::host(BufId(2), 0, 4),
    )])
    .unwrap();
    assert_eq!(b.run(&p, Tag(0)).status(), Some(Status::UnsupportedExtent));
}

#[test]
fn harness_random_fft() {
    run_random(Op::Fft);
}
