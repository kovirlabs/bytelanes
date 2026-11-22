//! Large inputs, compared with the CPU reference. They reach the code paths
//! the 4 KiB random cases cannot: more than 65,535 workgroups in one
//! dispatch (a 2D grid), many CRC combine levels, a Filter scan longer than
//! one workgroup, and a large result placed in device memory at an
//! unaligned offset.
//!
//! Hundreds of MB each, so ignored by default:
//! `cargo test -p bytelanes-wgpu --release --test large -- --ignored`

use bytelanes::abi::{BufId, ChannelCmd, ChannelProgram, CmpOp, Extent, FilterSpec};
use bytelanes::conformance::{Case, SplitMix64, expect_from_cpu, run_case};
use bytelanes_wgpu::WgpuBackend;

mod common;

const MIB: usize = 1 << 20;
const IN: BufId = BufId(1);
const OUT: BufId = BufId(2);
const VRAM: BufId = BufId(3);
const KEY: u32 = 9;

fn data(len: usize, seed: u64) -> Vec<u8> {
    SplitMix64::new(seed).bytes(len)
}

fn case(name: &str, cmds: &[ChannelCmd], input: Vec<u8>, out_len: u64, vram: u64) -> Case {
    let program = ChannelProgram::from_commands(cmds).expect("valid program");
    let host = vec![(IN, input), (OUT, vec![0; out_len as usize])];
    let vram = if vram > 0 { vec![(VRAM, vram)] } else { vec![] };
    let keys = vec![(KEY, [0x5A; 32])];
    let expect = expect_from_cpu(&program, &host, &vram, &keys, OUT, 0);
    assert_eq!(
        expect.status,
        bytelanes::abi::Status::Ok,
        "{name}: reference failed"
    );
    Case {
        name: name.into(),
        description: name.into(),
        program,
        host,
        vram,
        keys,
        expect,
    }
}

fn run_all(cases: &[Case]) {
    for a in &common::adapters() {
        let backend = WgpuBackend::new(a).expect("open adapter");
        for c in cases {
            backend.clear();
            let r = run_case(&backend, c);
            eprintln!("{:?} {}: {:?}", a.get_info().backend, c.name, r);
            assert!(r.is_ok(), "{}: {:?}", c.name, r);
        }
    }
}

#[test]
#[ignore = "large: run with --ignored"]
fn checksum_and_crypt_96_mib() {
    let n = 96 * MIB as u64;
    let iv = [0xFF; 16]; // wraps at 2^128 on the first block
    run_all(&[
        case(
            "checksum 96 MiB at offset 1",
            &[ChannelCmd::checksum(
                Extent::host(IN, 1, n - 1),
                Extent::host(OUT, 0, 4),
            )],
            data(n as usize, 1),
            4,
            0,
        ),
        case(
            "decrypt 96 MiB (2D grid)",
            &[ChannelCmd::decrypt(
                Extent::host(IN, 3, n - 3),
                Extent::host(OUT, 0, n),
                KEY,
                iv,
            )],
            data(n as usize, 2),
            n,
            0,
        ),
        // Encrypt into device memory at an unaligned offset, then checksum
        // the chained result.
        case(
            "encrypt 32 MiB to vram offset 5, chained checksum",
            &[
                ChannelCmd::encrypt(
                    Extent::host(IN, 0, 32 * MIB as u64),
                    Extent::vram(VRAM, 5, 32 * MIB as u64),
                    KEY,
                    [7; 16],
                )
                .chained(),
                ChannelCmd::checksum(
                    Extent::vram(VRAM, 5, 32 * MIB as u64),
                    Extent::host(OUT, 0, 4),
                ),
            ],
            data(32 * MIB, 3),
            4,
            32 * MIB as u64 + 16,
        ),
    ]);
}

#[test]
#[ignore = "large: run with --ignored"]
fn filter_70m_rows() {
    let rows = 70_000_000u64;
    // Small values so a predicate matches a predictable share of rows.
    let col: Vec<u8> = data(rows as usize, 4)
        .iter()
        .flat_map(|&b| u32::from(b).to_le_bytes())
        .collect();
    let f64_rows = 8_000_000u64;
    let col64: Vec<u8> = data(f64_rows as usize * 8, 5)
        .as_chunks::<8>()
        .0
        .iter()
        .flat_map(|c| {
            let x = f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])) / 7.0;
            x.to_le_bytes()
        })
        .collect();
    run_all(&[
        case(
            "filter u32 70M rows < 3 (2D grid, long scan)",
            &[ChannelCmd::filter(
                Extent::host(IN, 0, rows * 4),
                Extent::host(OUT, 0, rows * 4),
                FilterSpec::u32(CmpOp::Lt, 3, 0),
            )],
            col,
            rows * 4,
            0,
        ),
        case(
            "filter f64 8M rows between",
            &[ChannelCmd::filter(
                Extent::host(IN, 0, f64_rows * 8),
                Extent::host(OUT, 0, f64_rows * 4),
                FilterSpec::f64(CmpOp::Between, -1000.5, 2.0e8),
            )],
            col64,
            f64_rows * 4,
            0,
        ),
    ]);
}

#[test]
#[ignore = "large: run with --ignored"]
fn fft_large() {
    // Full-scale i32 samples (beyond the +-2^30 clamp) at an unaligned
    // offset; 4096-point frames over 64 MiB need a 2D grid, and
    // 65536-point frames run six global stages after the fused ones.
    let samples = |bytes: usize, seed: u64| data(bytes, seed);
    let mut cases = Vec::new();
    for (log2n, mib, seed) in [(12u8, 64usize, 6u64), (16, 32, 7)] {
        let n = (mib * MIB) as u64;
        let mut input = vec![0u8; 2];
        input.extend(samples(n as usize, seed));
        let out = bytelanes::cpu::fft::output_len(log2n, (n / (4 << log2n)) as usize) as u64;
        cases.push(case(
            &format!("fft log2n {log2n} over {mib} MiB at offset 2"),
            &[ChannelCmd::fft(
                Extent::host(IN, 2, n),
                Extent::host(OUT, 0, out),
                log2n,
            )],
            input,
            out,
            0,
        ));
    }
    run_all(&cases);
}
