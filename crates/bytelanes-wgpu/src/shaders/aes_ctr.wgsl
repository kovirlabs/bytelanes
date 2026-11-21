// AES-256-CTR. One thread per 16-byte block: counter = IV + block index
// (128-bit big-endian, wrapping), encrypted with AES-256, xored with the
// input. Round keys and tables come from the host. State words are
// big-endian columns, as in the reference "fst" implementation.

struct Params {
    off: u32,      // input byte offset in `src`
    n: u32,        // input length in bytes
    blocks: u32,   // ceil(n / 16)
    _pad: u32,
    iv: vec4<u32>, // IV as four big-endian words, most significant first
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> te0: array<u32, 256>;
@group(0) @binding(2) var<storage, read> sbox: array<u32, 256>;
@group(0) @binding(3) var<storage, read> rk: array<u32, 60>;
@group(0) @binding(4) var<storage, read_write> dst: array<u32>;
@group(0) @binding(5) var<uniform> p: Params;

var<workgroup> t0: array<u32, 256>;
var<workgroup> sb: array<u32, 256>;

fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

fn te(i: u32, rot: u32) -> u32 {
    if (rot == 0u) {
        return t0[i & 0xffu];
    }
    return rotr(t0[i & 0xffu], rot);
}

fn bswap(x: u32) -> u32 {
    return ((x & 0xffu) << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u);
}

fn encrypt(c: vec4<u32>) -> vec4<u32> {
    var s0 = c.x ^ rk[0];
    var s1 = c.y ^ rk[1];
    var s2 = c.z ^ rk[2];
    var s3 = c.w ^ rk[3];
    for (var r = 1u; r < 14u; r++) {
        let k = 4u * r;
        let n0 = te(s0 >> 24u, 0u) ^ te(s1 >> 16u, 8u) ^ te(s2 >> 8u, 16u) ^ te(s3, 24u) ^ rk[k];
        let n1 = te(s1 >> 24u, 0u) ^ te(s2 >> 16u, 8u) ^ te(s3 >> 8u, 16u) ^ te(s0, 24u) ^ rk[k + 1u];
        let n2 = te(s2 >> 24u, 0u) ^ te(s3 >> 16u, 8u) ^ te(s0 >> 8u, 16u) ^ te(s1, 24u) ^ rk[k + 2u];
        let n3 = te(s3 >> 24u, 0u) ^ te(s0 >> 16u, 8u) ^ te(s1 >> 8u, 16u) ^ te(s2, 24u) ^ rk[k + 3u];
        s0 = n0;
        s1 = n1;
        s2 = n2;
        s3 = n3;
    }
    let o0 = (sb[s0 >> 24u] << 24u) ^ (sb[(s1 >> 16u) & 0xffu] << 16u) ^ (sb[(s2 >> 8u) & 0xffu] << 8u) ^ sb[s3 & 0xffu] ^ rk[56];
    let o1 = (sb[s1 >> 24u] << 24u) ^ (sb[(s2 >> 16u) & 0xffu] << 16u) ^ (sb[(s3 >> 8u) & 0xffu] << 8u) ^ sb[s0 & 0xffu] ^ rk[57];
    let o2 = (sb[s2 >> 24u] << 24u) ^ (sb[(s3 >> 16u) & 0xffu] << 16u) ^ (sb[(s0 >> 8u) & 0xffu] << 8u) ^ sb[s1 & 0xffu] ^ rk[58];
    let o3 = (sb[s3 >> 24u] << 24u) ^ (sb[(s0 >> 16u) & 0xffu] << 16u) ^ (sb[(s1 >> 8u) & 0xffu] << 8u) ^ sb[s2 & 0xffu] ^ rk[59];
    return vec4<u32>(o0, o1, o2, o3);
}

// IV + j, 128-bit big-endian, wrapping at 2^128.
fn counter(j: u32) -> vec4<u32> {
    let w3 = p.iv.w + j;
    var carry = select(0u, 1u, w3 < p.iv.w);
    let w2 = p.iv.z + carry;
    carry = select(0u, 1u, carry == 1u && w2 == 0u);
    let w1 = p.iv.y + carry;
    carry = select(0u, 1u, carry == 1u && w1 == 0u);
    let w0 = p.iv.x + carry;
    return vec4<u32>(w0, w1, w2, w3);
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    for (var i = li; i < 256u; i += 64u) {
        t0[i] = te0[i];
        sb[i] = sbox[i];
    }
    workgroupBarrier();
    let j = group_index(wid, nwg) * 64u + li;
    if (j >= p.blocks) {
        return;
    }
    let ks = encrypt(counter(j));
    let base = p.off + 16u * j;
    // Bytes past `n` are written too; the host never reads them.
    dst[4u * j] = load_word(base) ^ bswap(ks.x);
    dst[4u * j + 1u] = load_word(base + 4u) ^ bswap(ks.y);
    dst[4u * j + 2u] = load_word(base + 8u) ^ bswap(ks.z);
    dst[4u * j + 3u] = load_word(base + 12u) ^ bswap(ks.w);
}
