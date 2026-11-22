//! Encrypt and decrypt whose source and destination are disjoint ranges of
//! one buffer: the CPU backend writes the result in place, so both
//! directions (destination before and after the source) must give the
//! kernel's bytes and leave everything else untouched.

use bytelanes::Buffers;
use bytelanes::abi::{BufId, ChannelCmd, ChannelProgram, Extent, Status, Tag};
use bytelanes::conformance::SplitMix64;
use bytelanes::cpu::{CpuBackend, ops};

#[test]
fn encrypt_within_one_buffer() {
    let key = [5u8; 32];
    let iv = [9u8; 16];
    let mut rng = SplitMix64::new(77);
    for (len, src_off, dst_off) in [
        (1000u64, 0u64, 1000u64),
        (1000, 1000, 0),
        (4097, 3, 5000),
        (4097, 5000, 1),
        (16, 100, 0),
    ] {
        for host in [true, false] {
            let size = (src_off.max(dst_off) + len + 7) as usize;
            let original = rng.bytes(size);
            let cpu = CpuBackend::new();
            cpu.set_key(1, key).unwrap();
            let id = BufId(4);
            let (src, dst, out) = if host {
                cpu.put_host(id, original.clone()).unwrap();
                (
                    Extent::host(id, src_off, len),
                    Extent::host(id, dst_off, len),
                    id,
                )
            } else {
                // Device memory: fill it from a host buffer first.
                cpu.put_host(BufId(1), original.clone()).unwrap();
                cpu.alloc_vram(id, size as u64).unwrap();
                cpu.put_host(BufId(2), vec![0; size]).unwrap();
                let fill = ChannelProgram::from_commands(&[
                    ChannelCmd::encrypt(
                        Extent::host(BufId(1), 0, size as u64),
                        Extent::vram(id, 0, size as u64),
                        1,
                        iv,
                    )
                    .chained(),
                    ChannelCmd::decrypt(
                        Extent::vram(id, 0, size as u64),
                        Extent::host(BufId(2), 0, size as u64),
                        1,
                        iv,
                    ),
                ])
                .unwrap();
                assert_eq!(cpu.run(&fill, Tag(0)).status(), Some(Status::Ok));
                (
                    Extent::vram(id, src_off, len),
                    Extent::vram(id, dst_off, len),
                    id,
                )
            };
            let prog =
                ChannelProgram::from_commands(&[ChannelCmd::encrypt(src, dst, 1, iv)]).unwrap();
            let c = cpu.run(&prog, Tag(1));
            assert_eq!(
                c.status(),
                Some(Status::Ok),
                "len {len} src {src_off} dst {dst_off}"
            );

            // Expected buffer: the device buffer holds encrypt(original)
            // after the fill program; the host buffer holds the original.
            let mut before = original.clone();
            if !host {
                ops::aes256_ctr_in_place(&key, &iv, &mut before);
            }
            let mut want = before.clone();
            let (s, d, n) = (src_off as usize, dst_off as usize, len as usize);
            let mut enc = vec![0u8; n];
            ops::aes256_ctr(&key, &iv, &before[s..s + n], &mut enc);
            want[d..d + n].copy_from_slice(&enc);
            assert_eq!(c.crc(), ops::checksum(&enc), "completion CRC");
            if host {
                assert_eq!(
                    cpu.get_host(out).unwrap(),
                    want,
                    "host len {len} src {src_off} dst {dst_off}"
                );
            } else {
                // Read the device buffer back through a checksum of each
                // range; the whole buffer must match `want`.
                let check = ChannelProgram::from_commands(&[ChannelCmd::checksum(
                    Extent::vram(id, 0, size as u64),
                    Extent::host(BufId(2), 0, 4),
                )])
                .unwrap();
                let c = cpu.run(&check, Tag(2));
                assert_eq!(c.status(), Some(Status::Ok));
                assert_eq!(
                    c.crc(),
                    ops::checksum(&ops::checksum(&want).to_le_bytes()),
                    "vram len {len}"
                );
            }
        }
    }
}
