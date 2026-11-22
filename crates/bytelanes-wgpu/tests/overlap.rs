//! Many programs in flight at once, so the worker overlaps them. Results
//! must equal the CPU reference running the same programs in order, with
//! overlap on and off, including programs that read what the previous
//! program wrote.

mod common;

use bytelanes::abi::{
    BufId, ChannelCmd, ChannelProgram, CmpOp, Completion, Extent, FilterSpec, Status, Tag,
};
use bytelanes::conformance::SplitMix64;
use bytelanes::cpu::CpuBackend;
use bytelanes::{Buffers, ChannelBackend};
use bytelanes_wgpu::WgpuBackend;

const KEY: u32 = 3;

fn prog(cmds: &[ChannelCmd]) -> ChannelProgram {
    ChannelProgram::from_commands(cmds).expect("valid program")
}

/// Submits every program, then polls until all complete; returns the
/// completions in tag order.
fn run_all<B: ChannelBackend>(b: &B, progs: &[ChannelProgram]) -> Vec<Completion> {
    for (i, p) in progs.iter().enumerate() {
        b.submit(p, Tag(i as u64)).expect("queue has room");
    }
    let mut out = vec![Completion::ok(Tag(0), 0, 0, 0); progs.len()];
    let mut slot = [Completion::ok(Tag(0), 0, 0, 0)];
    let mut got = 0;
    let mut order = Vec::new();
    while got < progs.len() {
        if b.poll(&mut slot) == 1 {
            let i = slot[0].tag().0 as usize;
            order.push(i);
            out[i] = slot[0];
            got += 1;
        }
    }
    assert_eq!(
        order,
        (0..progs.len()).collect::<Vec<_>>(),
        "completions arrive in submission order"
    );
    out
}

fn load<B: Buffers>(b: &B, host: &[(BufId, Vec<u8>)]) {
    for (id, data) in host {
        b.put_host(*id, data.clone()).unwrap();
    }
    b.set_key(KEY, [0x77; 32]).unwrap();
}

/// Checks `progs` on the GPU against the CPU reference: completions and
/// every host buffer afterwards.
fn check(gpu: &WgpuBackend, host: &[(BufId, Vec<u8>)], progs: &[ChannelProgram], what: &str) {
    let cpu = CpuBackend::new();
    load(&cpu, host);
    let want = run_all(&cpu, progs);
    gpu.clear();
    load(gpu, host);
    let got = run_all(gpu, progs);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            w.status(),
            Some(Status::Ok),
            "{what}: reference program {i}"
        );
        assert_eq!(g, w, "{what}: program {i}");
    }
    for (id, _) in host {
        assert_eq!(
            gpu.get_host(*id),
            cpu.get_host(*id),
            "{what}: host buffer {id:?}"
        );
    }
}

/// Encrypt A into B, checksum B, decrypt B into C, filter A: each later
/// program reads what the first wrote, so the worker must not overlap them
/// blindly.
#[test]
fn dependent_programs_see_earlier_results() {
    for a in &common::adapters() {
        let gpu = WgpuBackend::new(a).expect("open adapter");
        for overlap in [true, false] {
            gpu.set_overlap(overlap);
            for n in [64u64, 4096, 100_000, 1 << 20, 6 << 20] {
                let (src, mid, crc, back, idx) = (BufId(1), BufId(2), BufId(3), BufId(4), BufId(5));
                let host = vec![
                    (src, SplitMix64::new(n).bytes(n as usize)),
                    (mid, vec![0; n as usize]),
                    (crc, vec![0; 4]),
                    (back, vec![0; n as usize]),
                    (idx, vec![0; n as usize]),
                ];
                let progs = [
                    prog(&[ChannelCmd::encrypt(
                        Extent::host(src, 0, n),
                        Extent::host(mid, 0, n),
                        KEY,
                        [9; 16],
                    )]),
                    prog(&[ChannelCmd::checksum(
                        Extent::host(mid, 0, n),
                        Extent::host(crc, 0, 4),
                    )]),
                    prog(&[ChannelCmd::decrypt(
                        Extent::host(mid, 0, n),
                        Extent::host(back, 0, n),
                        KEY,
                        [9; 16],
                    )]),
                    prog(&[ChannelCmd::filter(
                        Extent::host(back, 0, n),
                        Extent::host(idx, 0, n),
                        FilterSpec::u32(CmpOp::Lt, 1 << 30, 0),
                    )]),
                ];
                check(
                    &gpu,
                    &host,
                    &progs,
                    &format!("{:?} overlap={overlap} n={n}", a.get_info().backend),
                );
                assert_eq!(
                    gpu.get_host(back),
                    Some(host[0].1.clone()),
                    "decrypt undoes encrypt"
                );
            }
        }
    }
}

/// Many independent programs of mixed ops and sizes in flight at once.
#[test]
fn independent_programs_in_flight() {
    for a in &common::adapters() {
        let gpu = WgpuBackend::new(a).expect("open adapter");
        for overlap in [true, false] {
            gpu.set_overlap(overlap);
            let mut rng = SplitMix64::new(42);
            let mut host = Vec::new();
            let mut progs = Vec::new();
            for k in 0..48u64 {
                let n =
                    [16u64, 1000, 65_536, 300_000, 2 << 20][(k % 5) as usize] + 4 * rng.below(8);
                let (src, dst) = (BufId(100 + 2 * k), BufId(101 + 2 * k));
                host.push((src, rng.bytes(n as usize)));
                let cmd = match k % 4 {
                    0 => ChannelCmd::checksum(Extent::host(src, 0, n), Extent::host(dst, 0, 4)),
                    1 => ChannelCmd::encrypt(
                        Extent::host(src, 0, n),
                        Extent::host(dst, 0, n),
                        KEY,
                        [k as u8; 16],
                    ),
                    2 => ChannelCmd::decrypt(
                        Extent::host(src, 3, n - 3),
                        Extent::host(dst, 0, n),
                        KEY,
                        [1; 16],
                    ),
                    _ => ChannelCmd::filter(
                        Extent::host(src, 0, n),
                        Extent::host(dst, 0, n),
                        FilterSpec::u32(CmpOp::Gt, u32::MAX / 2, 0),
                    ),
                };
                host.push((dst, vec![0; n as usize]));
                progs.push(prog(&[cmd]));
            }
            check(
                &gpu,
                &host,
                &progs,
                &format!("{:?} overlap={overlap}", a.get_info().backend),
            );
        }
    }
}
