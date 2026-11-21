//! Wire-format tests for the op format.

use bytelanes_abi::zerocopy::{FromBytes, IntoBytes};
use bytelanes_abi::*;
use proptest::prelude::*;

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

const H_IN: BufId = BufId(1);
const H_OUT: BufId = BufId(2);
const V: BufId = BufId(10);

#[test]
fn layout_offsets_are_fixed() {
    let cmd = ChannelCmd::decrypt(
        Extent::host(H_IN, 3, 64),
        Extent::host(H_OUT, 0, 64),
        7,
        [0xAB; 16],
    )
    .chained();
    let b = cmd.as_bytes();
    assert_eq!(b.len(), 104);
    assert_eq!(&b[0..2], &Op::Decrypt.code().to_le_bytes());
    assert_eq!(&b[2..4], &1u16.to_le_bytes());
    assert_eq!(b[8], ExtentKind::Host as u8);
    assert_eq!(&b[16..24], &1u64.to_le_bytes());
    assert_eq!(&b[24..32], &3u64.to_le_bytes());
    assert_eq!(&b[32..40], &64u64.to_le_bytes());
    assert_eq!(b[40], ExtentKind::Host as u8);
    assert_eq!(&b[72..76], &7u32.to_le_bytes());
    assert_eq!(&b[76..88], &[0u8; 12]);
    assert_eq!(&b[88..104], &[0xAB; 16]);

    let c = Completion::ok(Tag(0x0102030405060708), 2, 99, 0xDEADBEEF);
    let b = c.as_bytes();
    assert_eq!(&b[0..8], &0x0102030405060708u64.to_le_bytes());
    assert_eq!(&b[8..10], &0u16.to_le_bytes());
    assert_eq!(&b[10..12], &2u16.to_le_bytes());
    assert_eq!(&b[12..16], &99u32.to_le_bytes());
    assert_eq!(&b[16..20], &0xDEADBEEFu32.to_le_bytes());
}

#[test]
fn program_header_and_roundtrip() {
    let p = ChannelProgram::from_commands(&[ChannelCmd::checksum(
        Extent::host(H_IN, 0, 10),
        Extent::host(H_OUT, 0, 4),
    )])
    .unwrap();
    let b = p.as_bytes();
    assert_eq!(b.len(), 1672);
    assert_eq!(&b[0..2], &FORMAT_VERSION.to_le_bytes());
    assert_eq!(&b[2..4], &1u16.to_le_bytes());
    assert_eq!(ChannelProgram::from_bytes(b).unwrap(), &p);
    assert!(ChannelProgram::from_bytes(&b[1..]).is_err());
}

fn invalid(cmds: &[ChannelCmd]) -> ProgramError {
    ChannelProgram::from_commands(cmds).unwrap_err()
}

#[test]
fn validation_rules() {
    let hin = Extent::host(H_IN, 0, 64);
    let hout = Extent::host(H_OUT, 0, 64);
    let v = Extent::vram(V, 0, 64);

    // Empty program.
    let p = ChannelProgram::new();
    assert_eq!(p.validate().unwrap_err().index, NO_INDEX);

    // Single command host to host is fine.
    ChannelProgram::from_commands(&[ChannelCmd::checksum(hin, hout)]).unwrap();

    // Intermediate dst must be Vram.
    let e = invalid(&[
        ChannelCmd::checksum(hin, hout),
        ChannelCmd::checksum(hin, Extent::host(H_OUT, 64, 4)),
    ]);
    assert_eq!(e.index, 0);

    // Valid chain: decrypt to vram, filter vram to host.
    ChannelProgram::from_commands(&[
        ChannelCmd::decrypt(hin, v, 0, [0; 16]).chained(),
        ChannelCmd::filter(v, hout, FilterSpec::u32(CmpOp::Gt, 5, 0)),
    ])
    .unwrap();

    // Chain whose next src differs.
    let e = invalid(&[
        ChannelCmd::decrypt(hin, v, 0, [0; 16]).chained(),
        ChannelCmd::checksum(hin, hout),
    ]);
    assert_eq!(e.index, 1);

    // CHAIN on the last command.
    let e = invalid(&[ChannelCmd::checksum(hin, hout).chained()]);
    assert_eq!(e.index, 0);

    // dst overlapping src in the same buffer: not idempotent.
    let e = invalid(&[ChannelCmd::decrypt(
        hin,
        Extent::host(H_IN, 32, 64),
        0,
        [0; 16],
    )]);
    assert_eq!(e.reason, "dst overlaps a src of the same program");
    // In-place is rejected too.
    invalid(&[ChannelCmd::decrypt(hin, hin, 0, [0; 16])]);
    // Adjacent, non-overlapping ranges of one buffer are fine.
    ChannelProgram::from_commands(&[ChannelCmd::decrypt(
        hin,
        Extent::host(H_IN, 64, 64),
        0,
        [0; 16],
    )])
    .unwrap();
    // Same id but different kinds never overlap.
    ChannelProgram::from_commands(&[ChannelCmd::checksum(
        Extent::host(V, 0, 8),
        Extent::vram(V, 0, 8),
    )])
    .unwrap();

    // Later command writing a vram range an earlier one read.
    invalid(&[
        ChannelCmd::checksum(v, Extent::vram(BufId(11), 0, 4)),
        ChannelCmd::decrypt(hin, v, 0, [0; 16]),
        ChannelCmd::checksum(Extent::vram(BufId(12), 0, 4), hout),
    ]);

    // None extents for a defined op.
    invalid(&[ChannelCmd::checksum(Extent::NONE, hout)]);

    // Reserved op: zero params is structurally valid; backends report UnsupportedOp.
    ChannelProgram::from_commands(&[ChannelCmd::new(
        Op::Compress,
        Flags::NONE,
        hin,
        hout,
        [0; PARAMS_LEN],
    )])
    .unwrap();

    // Unknown flags.
    invalid(&[ChannelCmd::new(
        Op::Checksum,
        Flags(2),
        hin,
        hout,
        [0; PARAMS_LEN],
    )]);

    // Checksum with params.
    let mut params = [0; PARAMS_LEN];
    params[0] = 1;
    invalid(&[ChannelCmd::new(
        Op::Checksum,
        Flags::NONE,
        hin,
        hout,
        params,
    )]);

    // 16 commands is the limit.
    let mut p = ChannelProgram::new();
    for i in 0..16 {
        p.push(ChannelCmd::checksum(
            hin,
            Extent::host(BufId(100 + i), 0, 4),
        ))
        .unwrap();
    }
    assert!(p.push(ChannelCmd::checksum(hin, hout)).is_err());
}

#[test]
fn extent_decode_rules() {
    assert_eq!(Extent::NONE.decode(), Ok(ExtentRef::None));
    assert!(Extent::host(H_IN, u64::MAX, 1).decode().is_err());
    assert!(Extent::host(H_IN, u64::MAX, 0).decode().is_ok());
    assert_eq!(
        Extent::nvme(DevId(3), 10, 8).decode(),
        Ok(ExtentRef::Nvme(NvmeRange {
            dev: DevId(3),
            lba: 10,
            blocks: 8
        }))
    );
    let mut raw = Extent::host(H_IN, 0, 1).as_bytes().to_vec();
    raw[0] = 9;
    assert!(Extent::read_from_bytes(&raw).unwrap().decode().is_err());
    raw[0] = 1;
    raw[3] = 1;
    assert!(Extent::read_from_bytes(&raw).unwrap().decode().is_err());
    // Nvme block count above u32.
    let mut raw = Extent::nvme(DevId(1), 0, 1).as_bytes().to_vec();
    raw[24..32].copy_from_slice(&(u64::from(u32::MAX) + 1).to_le_bytes());
    assert!(Extent::read_from_bytes(&raw).unwrap().decode().is_err());
}

#[test]
fn filter_params_rules() {
    for &elem in &ElemType::ALL {
        for &cmp in &CmpOp::ALL {
            let spec = FilterSpec {
                elem,
                cmp,
                output: FilterOutput::Indices,
                a: 1,
                b: if cmp == CmpOp::Between { 2 } else { 0 },
            };
            assert_eq!(spec.to_params().decode(), Ok(spec));
        }
    }
    let bad = FilterSpec {
        elem: ElemType::U32,
        cmp: CmpOp::Eq,
        output: FilterOutput::Indices,
        a: 1 << 32,
        b: 0,
    };
    assert!(bad.to_params().decode().is_err());
    let bad = FilterSpec {
        elem: ElemType::U64,
        cmp: CmpOp::Eq,
        output: FilterOutput::Indices,
        a: 0,
        b: 1,
    };
    assert!(bad.to_params().decode().is_err());
    // Constructors drop b for one-operand comparisons.
    assert_eq!(FilterSpec::i32(CmpOp::Lt, -1, 7).b, 0);
    assert_eq!(
        FilterSpec::i32(CmpOp::Between, -1, 7).a,
        u64::from(u32::MAX)
    );
}

#[test]
fn codes_roundtrip() {
    for op in Op::ALL {
        assert_eq!(Op::from_code(op.code()), Some(op));
    }
    assert_eq!(Op::from_code(0), None);
    for code in 0..=10 {
        assert_eq!(Status::from_code(code).unwrap().code(), code);
    }
    assert_eq!(Status::from_code(11), None);
    let caps = Caps::new(
        &Op::DEFINED,
        &[ExtentKind::Host],
        Features::UNIFIED_MEMORY,
        64,
    );
    assert!(caps.supports(Op::Filter) && !caps.supports(Op::Read));
    assert!(caps.supports_extent(ExtentKind::Host) && !caps.supports_extent(ExtentKind::Vram));
    assert!(caps.features().contains(Features::UNIFIED_MEMORY));
    assert_eq!(caps.queue_depth(), 64);
}

proptest! {
    #![proptest_config(cfg())]

    /// Arbitrary bytes never panic the decoder, and anything it accepts
    /// re-encodes to the same bytes.
    #[test]
    fn arbitrary_program_bytes_never_panic(mut bytes in prop::collection::vec(any::<u8>(), 1672), len in 1u16..=16, valid_head in any::<bool>()) {
        if valid_head {
            bytes[0..2].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
            bytes[2..4].copy_from_slice(&len.to_le_bytes());
            bytes[4..8].fill(0);
            bytes[8 + usize::from(len) * 104..].fill(0);
        }
        if let Ok(p) = ChannelProgram::from_bytes(&bytes) {
            prop_assert_eq!(p.as_bytes(), &bytes[..]);
        }
    }

    #[test]
    fn extent_roundtrip(kind in 0u8..4, a in any::<u64>(), b in any::<u64>(), c in any::<u32>()) {
        let e = match kind {
            0 => Extent::NONE,
            1 => Extent::host(BufId(a), b / 2, u64::from(c)),
            2 => Extent::vram(BufId(a), b / 2, u64::from(c)),
            _ => Extent::nvme(DevId(a), b / 2, c),
        };
        let r = e.decode().unwrap();
        prop_assert_eq!(Extent::from_ref(r), e);
        prop_assert_eq!(Extent::read_from_bytes(e.as_bytes()).unwrap(), e);
    }
}
