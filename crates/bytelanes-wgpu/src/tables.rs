//! Constants the shaders need, computed on the host: CRC-32C tables and
//! powers of x, the AES S-box and T-table, and the AES-256 key schedule.
//!
//! Computed rather than typed in, and checked in the tests against the CPU
//! reference in `bytelanes::cpu`.

/// CRC-32C polynomial, bit-reflected.
pub(crate) const CRC_POLY: u32 = 0x82F6_3B78;

/// Byte-at-a-time table for the reflected CRC-32C.
pub(crate) fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for (i, e) in t.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ CRC_POLY
            } else {
                c >> 1
            };
        }
        *e = c;
    }
    t
}

/// `a(x) * b(x) mod P(x)` with bit 31 as x^0 (zlib's `multmodp`).
pub(crate) fn multmodp(a: u32, mut b: u32) -> u32 {
    if a == 0 {
        return 0;
    }
    let mut m = 1u32 << 31;
    let mut p = 0u32;
    loop {
        if a & m != 0 {
            p ^= b;
            if a & (m - 1) == 0 {
                return p;
            }
        }
        m >>= 1;
        b = if b & 1 != 0 {
            (b >> 1) ^ CRC_POLY
        } else {
            b >> 1
        };
    }
}

/// `x^(8 * 2^k) mod P` for `k` in `0..32`: multiplying a CRC register by
/// entry `k` is the same as feeding it `2^k` zero bytes.
pub(crate) fn crc_byte_powers() -> [u32; 32] {
    // Entry k is x^(2^(k+3)) mod P: start from x^8 and square. Do not wrap
    // the index the way zlib does for CRC-32: for CRC-32C, x^(2^n) mod P
    // repeats with period 31, not 32.
    let mut out = [0u32; 32];
    let mut p = 1u32 << 30; // x^1
    for _ in 0..3 {
        p = multmodp(p, p);
    }
    for e in &mut out {
        *e = p;
        p = multmodp(p, p);
    }
    out
}

/// The AES S-box.
pub(crate) fn aes_sbox() -> [u8; 256] {
    let mut s = [0u8; 256];
    let (mut p, mut q) = (1u8, 1u8);
    loop {
        // p *= 3 in GF(2^8)
        p = p ^ (p << 1) ^ if p & 0x80 != 0 { 0x1B } else { 0 };
        // q /= 3
        q ^= q << 1;
        q ^= q << 2;
        q ^= q << 4;
        if q & 0x80 != 0 {
            q ^= 0x09;
        }
        let x = q ^ q.rotate_left(1) ^ q.rotate_left(2) ^ q.rotate_left(3) ^ q.rotate_left(4);
        s[p as usize] = x ^ 0x63;
        if p == 1 {
            break;
        }
    }
    s[0] = 0x63;
    s
}

fn xtime(x: u8) -> u8 {
    (x << 1) ^ if x & 0x80 != 0 { 0x1B } else { 0 }
}

/// `Te0[x] = S[x] * [02, 01, 01, 03]`, most significant byte first. The
/// other three round tables are byte rotations of it.
pub(crate) fn aes_te0(sbox: &[u8; 256]) -> [u32; 256] {
    let mut t = [0u32; 256];
    for (i, e) in t.iter_mut().enumerate() {
        let s = sbox[i];
        let s2 = xtime(s);
        let s3 = s2 ^ s;
        *e = u32::from_be_bytes([s2, s, s, s3]);
    }
    t
}

/// AES-256 encryption key schedule: 60 words, big-endian.
pub(crate) fn aes256_round_keys(key: &[u8; 32], sbox: &[u8; 256]) -> [u32; 60] {
    let sub = |w: u32| {
        let b = w.to_be_bytes();
        u32::from_be_bytes([
            sbox[b[0] as usize],
            sbox[b[1] as usize],
            sbox[b[2] as usize],
            sbox[b[3] as usize],
        ])
    };
    let mut rk = [0u32; 60];
    for (i, &chunk) in key.as_chunks::<4>().0.iter().enumerate() {
        rk[i] = u32::from_be_bytes(chunk);
    }
    let mut rcon = 1u8;
    for i in 8..60 {
        let mut t = rk[i - 1];
        if i % 8 == 0 {
            t = sub(t.rotate_left(8)) ^ (u32::from(rcon) << 24);
            rcon = xtime(rcon);
        } else if i % 8 == 4 {
            t = sub(t);
        }
        rk[i] = rk[i - 8] ^ t;
    }
    rk
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytelanes::conformance::SplitMix64;
    use bytelanes::cpu::ops;

    /// Entry `k` is `x^(8 * 2^k)`: entry 0 is one zero byte through the
    /// table, and each entry is the square of the one before. Entries 29 to
    /// 31 (inputs of 512 MiB and more) once wrapped back to `x^1`, `x^2` and
    /// `x^4`, on the false assumption that `x^(2^n) mod P` repeats with
    /// period 32. For CRC-32C the period is 31.
    #[test]
    fn crc_byte_powers_are_successive_squares() {
        let t = crc_table();
        let one_zero_byte = |r: u32| t[(r & 0xFF) as usize] ^ (r >> 8);
        let pow = crc_byte_powers();
        // x^0 is 1 << 31 in this reflected representation.
        assert_eq!(pow[0], one_zero_byte(1 << 31));
        for k in 1..32 {
            assert_eq!(pow[k], multmodp(pow[k - 1], pow[k - 1]), "entry {k}");
        }
    }

    #[test]
    fn crc_table_matches_reference() {
        let t = crc_table();
        let crc = |data: &[u8]| {
            let mut r = !0u32;
            for &b in data {
                r = t[((r ^ u32::from(b)) & 0xFF) as usize] ^ (r >> 8);
            }
            !r
        };
        assert_eq!(crc(b"123456789"), 0xE306_9283);
        let mut rng = SplitMix64::new(7);
        let data = rng.bytes(1000);
        assert_eq!(crc(&data), ops::checksum(&data));
    }

    /// Runs the shader's algorithm on the host: raw (init 0) CRC per chunk,
    /// pairwise combine, then fold in the initial value and final xor.
    fn chunked_crc(data: &[u8], chunk: usize) -> u32 {
        let t = crc_table();
        let pow = crc_byte_powers();
        let shift = |r: u32, len: usize| {
            let mut x = 1u32 << 31;
            for (k, p) in pow.iter().enumerate() {
                if (len >> k) & 1 != 0 {
                    x = multmodp(*p, x);
                }
            }
            multmodp(x, r)
        };
        let mut nodes: Vec<(u32, usize)> = data
            .chunks(chunk)
            .map(|c| {
                let mut r = 0u32;
                for &b in c {
                    r = t[((r ^ u32::from(b)) & 0xFF) as usize] ^ (r >> 8);
                }
                (r, c.len())
            })
            .collect();
        while nodes.len() > 1 {
            nodes = nodes
                .chunks(2)
                .map(|p| match p {
                    [l, r] => (shift(l.0, r.1) ^ r.0, l.1 + r.1),
                    [l] => *l,
                    _ => unreachable!(),
                })
                .collect();
        }
        let r0 = nodes.first().map_or(0, |n| n.0);
        !(shift(!0, data.len()) ^ r0)
    }

    #[test]
    fn chunked_crc_matches_reference() {
        let mut rng = SplitMix64::new(11);
        for len in [0usize, 1, 3, 63, 64, 65, 255, 256, 257, 1000, 4096, 70_001] {
            let data = rng.bytes(len);
            for chunk in [1, 7, 64, 256] {
                assert_eq!(
                    chunked_crc(&data, chunk),
                    ops::checksum(&data),
                    "len {len} chunk {chunk}"
                );
            }
        }
    }

    #[test]
    fn sbox_known_values() {
        let s = aes_sbox();
        assert_eq!(
            (s[0x00], s[0x01], s[0x53], s[0xFF]),
            (0x63, 0x7C, 0xED, 0x16)
        );
        let mut seen = [false; 256];
        for &v in &s {
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "S-box is a permutation");
    }

    /// FIPS-197 Appendix A.3 key expansion for a 256-bit key.
    #[test]
    fn key_schedule_fips197() {
        let key: [u8; 32] = [
            0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d,
            0x77, 0x81, 0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3,
            0x09, 0x14, 0xdf, 0xf4,
        ];
        let rk = aes256_round_keys(&key, &aes_sbox());
        assert_eq!(rk[8], 0x9ba3_5411);
        assert_eq!(rk[9], 0x8e69_25af);
        assert_eq!(rk[59], 0x706c_631e);
    }
}
