//! The programs and generated data each benchmark point runs.
//!
//! All data is generated from fixed seeds (SplitMix64), so every machine
//! measures the same bytes.

use bytelanes::abi::{BufId, ChannelCmd, ChannelProgram, CmpOp, Extent, FilterSpec, KeySlot};
use bytelanes::conformance::SplitMix64;
use bytelanes::cpu::ops;

pub const KEY_SLOT: KeySlot = 1;
pub const KEY: [u8; 32] = [0x4B; 32];

const IN: BufId = BufId(1);
const OUT: BufId = BufId(2);
const AMT: BufId = BufId(3);
const FLAG: BufId = BufId(4);
const ENC: BufId = BufId(5);
const VRAM: BufId = BufId(6);

/// One benchmark point: the programs one repetition submits together,
/// plus the buffers they need.
pub struct Case {
    pub programs: Vec<ChannelProgram>,
    pub host: Vec<(BufId, Vec<u8>)>,
    pub vram: Vec<(BufId, u64)>,
    /// Bytes of input the program processes per run (for throughput).
    pub input_bytes: u64,
    /// Work units per run: bytes for ops, transactions for the commit log.
    pub units: u64,
}

fn program(cmds: &[ChannelCmd]) -> ChannelProgram {
    ChannelProgram::from_commands(cmds).expect("benchmark programs are valid")
}

/// The single-op benchmark ops.
pub const OPS: [&str; 4] = ["checksum", "encrypt", "decrypt", "filter"];

/// Threshold for the u32 Filter: about 1% of uniform values are above it.
const FILTER_U32_GT: u32 = u32::MAX - u32::MAX / 100;

/// `op` over `n` bytes of generated input, host to host.
pub fn op_case(op: &str, n: u64) -> Case {
    let input = SplitMix64::new(0xB0B5 ^ n).bytes(n as usize);
    let iv = [0x11; 16];
    let (cmd, out_len) = match op {
        "checksum" => (
            ChannelCmd::checksum(Extent::host(IN, 0, n), Extent::host(OUT, 0, 4)),
            4,
        ),
        "encrypt" => (
            ChannelCmd::encrypt(
                Extent::host(IN, 0, n),
                Extent::host(OUT, 0, n),
                KEY_SLOT,
                iv,
            ),
            n,
        ),
        "decrypt" => (
            ChannelCmd::decrypt(
                Extent::host(IN, 0, n),
                Extent::host(OUT, 0, n),
                KEY_SLOT,
                iv,
            ),
            n,
        ),
        "filter" => (
            ChannelCmd::filter(
                Extent::host(IN, 0, n),
                Extent::host(OUT, 0, n),
                FilterSpec::u32(CmpOp::Gt, FILTER_U32_GT, 0),
            ),
            n,
        ),
        other => panic!("unknown op {other}"),
    };
    Case {
        programs: vec![program(&[cmd])],
        host: vec![(IN, input), (OUT, vec![0; out_len as usize])],
        vram: vec![],
        input_bytes: n,
        units: n,
    }
}

/// Bytes per generated transaction record.
pub const RECORD: u64 = 256;
/// Fraud rule: amounts above this many cents ($5,000) are flagged.
pub const FLAG_ABOVE_CENTS: i64 = 500_000;

/// `t` transaction records and their amount column.
///
/// Record layout (little-endian): txn id u64, account u64, amount i64
/// (cents), timestamp u64 (µs), merchant u32, currency u16, flags u16, then
/// 216 bytes of payload. Amounts: 99% uniform in $0-$500, 1% uniform in
/// $5,000-$1,005,000, so the fraud rule flags about 1%.
pub fn transactions(t: u64, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = SplitMix64::new(seed);
    let mut records = Vec::with_capacity((t * RECORD) as usize);
    let mut amounts = Vec::with_capacity((t * 8) as usize);
    let t0 = 1_795_000_000_000_000u64;
    for i in 0..t {
        let amount: i64 = if rng.below(100) == 0 {
            FLAG_ABOVE_CENTS + 1 + rng.below(100_000_000) as i64
        } else {
            rng.below(50_000) as i64
        };
        records.extend_from_slice(&(seed.wrapping_mul(1 << 32) + i).to_le_bytes());
        records.extend_from_slice(&rng.below(10_000_000).to_le_bytes());
        records.extend_from_slice(&amount.to_le_bytes());
        records.extend_from_slice(&(t0 + i * 37).to_le_bytes());
        records.extend_from_slice(&(rng.below(200_000) as u32).to_le_bytes());
        records.extend_from_slice(&840u16.to_le_bytes());
        records.extend_from_slice(&0u16.to_le_bytes());
        records.extend_from_slice(&rng.bytes(216));
        amounts.extend_from_slice(&amount.to_le_bytes());
    }
    (records, amounts)
}

fn batch_iv(t: u64) -> [u8; 16] {
    let mut iv = [0u8; 16];
    iv[..8].copy_from_slice(&t.to_be_bytes());
    iv
}

/// Write path for a batch of `t` transactions, as two programs submitted
/// together (the op format keeps every non-final result in device memory,
/// so two host results need two programs):
/// 1. flag rows whose amount breaks the fraud rule (Filter on the amount
///    column, indices back to the host);
/// 2. encrypt the batch's records for the log (host to host). The
///    completion's CRC-32C, computed where the result was produced, is the
///    CRC of the encrypted segment the log would store.
pub fn commit_case(t: u64) -> Case {
    let (records, amounts) = transactions(t, t);
    let n = t * RECORD;
    let flag = ChannelCmd::filter(
        Extent::host(AMT, 0, t * 8),
        Extent::host(FLAG, 0, t * 4),
        FilterSpec::i64(CmpOp::Gt, FLAG_ABOVE_CENTS, 0),
    );
    let seal = ChannelCmd::encrypt(
        Extent::host(IN, 0, n),
        Extent::host(ENC, 0, n),
        KEY_SLOT,
        batch_iv(t),
    );
    Case {
        programs: vec![program(&[flag]), program(&[seal])],
        host: vec![
            (IN, records),
            (AMT, amounts),
            (FLAG, vec![0; (t * 4) as usize]),
            (ENC, vec![0; n as usize]),
        ],
        vram: vec![],
        input_bytes: n + t * 8,
        units: t,
    }
}

/// Read path (audit) for a batch of `t` transactions: decrypt the
/// encrypted amount column into device memory and filter it there, as one
/// chained program. Only the flagged row indices come back.
pub fn audit_case(t: u64) -> Case {
    let (_, amounts) = transactions(t, t);
    let mut enc = vec![0u8; amounts.len()];
    ops::aes256_ctr(&KEY, &batch_iv(t), &amounts, &mut enc);
    let len = t * 8;
    let cmds = [
        ChannelCmd::decrypt(
            Extent::host(ENC, 0, len),
            Extent::vram(VRAM, 0, len),
            KEY_SLOT,
            batch_iv(t),
        )
        .chained(),
        ChannelCmd::filter(
            Extent::vram(VRAM, 0, len),
            Extent::host(FLAG, 0, t * 4),
            FilterSpec::i64(CmpOp::Gt, FLAG_ABOVE_CENTS, 0),
        ),
    ];
    Case {
        programs: vec![program(&cmds)],
        host: vec![(ENC, enc), (FLAG, vec![0; (t * 4) as usize])],
        vram: vec![(VRAM, len)],
        input_bytes: len,
        units: t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_have_the_documented_layout() {
        let (r, a) = transactions(1000, 7);
        assert_eq!(r.len(), 256_000);
        assert_eq!(a.len(), 8000);
        for i in 0..1000 {
            let rec = &r[i * 256..];
            assert_eq!(rec[16..24], a[i * 8..i * 8 + 8], "amount at offset 16");
        }
        let flagged = a
            .as_chunks::<8>()
            .0
            .iter()
            .filter(|&&c| i64::from_le_bytes(c) > FLAG_ABOVE_CENTS)
            .count();
        assert!(
            (3..=30).contains(&flagged),
            "about 1% flagged, got {flagged}"
        );
    }
}
