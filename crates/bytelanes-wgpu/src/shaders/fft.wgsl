// Fixed-point FFT, bit for bit as specified on `Op::Fft`:
//
//   a[i] = (sat(x[bitrev(i)]), 0)                  sat clamps to [-2^30, 2^30]
//   for s in 1..=log2n: m = 2^s, h = m/2, stride = 65536/m
//     for each butterfly (k, j):
//       (wr, wi) = W[j * stride]; u = a[k+j]; b = a[k+j+h]
//       tr = (wr*br - wi*bi + 2^29) >> 30           in i64, arithmetic shift
//       ti = (wr*bi + wi*br + 2^29) >> 30
//       a[k+j]   = ((ur + tr) >> 1, (ui + ti) >> 1)
//       a[k+j+h] = ((ur - tr) >> 1, (ui - ti) >> 1)
//   write a[0..=N/2]
//
// WGSL has no 64-bit integers: products and sums are (lo, hi) pairs of u32.
// `>> 30` of the 64-bit sum keeps bits 30..62, which the spec guarantees fit
// in i32. The halving adds may exceed i32 before the shift, so they use
// floor((a + b) / 2) = (a >> 1) + (b >> 1) + (a & b & 1) and
// floor((a - b) / 2) = (a >> 1) - (b >> 1) - (~a & b & 1), which are exact.
//
// Entry points, one dispatch each:
//   fft_load:    one thread per element: bit-reverse, clamp, imaginary part 0
//   fft_stage:   one thread per butterfly of stage `s`, every frame at once
//   fft_local:   stages 1..=p.local in workgroup memory, a tile of TILE
//             elements per workgroup (tiles never straddle a frame)
//   fft_extract: one thread per output bin: bins 0..=N/2 of every frame

struct Params {
    off: u32,      // input byte offset in `src`
    log2n: u32,    // N = 2^log2n samples per frame
    frames: u32,
    s: u32,        // stage (stage); stages fused (local)
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> tw: array<vec2<i32>, 32768>;
@group(0) @binding(2) var<storage, read_write> a: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<i32>>;
@group(0) @binding(4) var<uniform> p: Params;

const LIMIT: i32 = 1073741824; // 2^30
const TILE: u32 = 1024u;
const LOG2_TILE: u32 = 10u;

var<workgroup> tile: array<vec2<i32>, 1024>;

// Unsigned 32 x 32 -> 64 multiply as (lo, hi), through 16-bit limbs.
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (p00 & 0xffffu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

// Signed 32 x 32 -> 64 multiply (two's complement), as (lo, hi).
fn mul_i32(a: i32, b: i32) -> vec2<u32> {
    let ua = bitcast<u32>(a);
    let ub = bitcast<u32>(b);
    let u = mul_u32(ua, ub);
    let hi = u.y - select(0u, ub, a < 0) - select(0u, ua, b < 0);
    return vec2<u32>(u.x, hi);
}

fn add64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let lo = x.x + y.x;
    return vec2<u32>(lo, x.y + y.y + select(0u, 1u, lo < x.x));
}

fn sub64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(x.x - y.x, x.y - y.y - select(0u, 1u, x.x < y.x));
}

// (v + 2^29) >> 30 for a 64-bit v whose result fits in i32.
fn round_q30(v: vec2<u32>) -> i32 {
    let r = add64(v, vec2<u32>(0x20000000u, 0u));
    return bitcast<i32>((r.x >> 30u) | (r.y << 2u));
}

fn half_sum(a: i32, b: i32) -> i32 {
    return (a >> 1u) + (b >> 1u) + (a & b & 1);
}

fn half_diff(a: i32, b: i32) -> i32 {
    return (a >> 1u) - (b >> 1u) - (~a & b & 1);
}

// The butterfly for twiddle index `t` on `u` (top) and `b` (bottom):
// returns (new top, new bottom).
fn butterfly(t: u32, u: vec2<i32>, b: vec2<i32>) -> array<vec2<i32>, 2> {
    let w = tw[t];
    let tr = round_q30(sub64(mul_i32(w.x, b.x), mul_i32(w.y, b.y)));
    let ti = round_q30(add64(mul_i32(w.x, b.y), mul_i32(w.y, b.x)));
    return array<vec2<i32>, 2>(
        vec2<i32>(half_sum(u.x, tr), half_sum(u.y, ti)),
        vec2<i32>(half_diff(u.x, tr), half_diff(u.y, ti)),
    );
}

@compute @workgroup_size(256)
fn fft_load(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = group_index(wid, nwg) * 256u + li;
    let n = 1u << p.log2n;
    if (g >= p.frames * n) {
        return;
    }
    let frame = g >> p.log2n;
    let i = g & (n - 1u);
    let srci = (frame << p.log2n) + (reverseBits(i) >> (32u - p.log2n));
    let x = clamp(bitcast<i32>(load_word(p.off + srci * 4u)), -LIMIT, LIMIT);
    a[g] = vec2<i32>(x, 0);
}

@compute @workgroup_size(256)
fn fft_stage(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = group_index(wid, nwg) * 256u + li;
    let half_n = 1u << (p.log2n - 1u);
    if (g >= p.frames * half_n) {
        return;
    }
    let frame = g >> (p.log2n - 1u);
    let q = g & (half_n - 1u);        // butterfly within the frame
    let h = 1u << (p.s - 1u);
    let j = q & (h - 1u);
    let k = (q >> (p.s - 1u)) << p.s;
    let top = (frame << p.log2n) + k + j;
    let r = butterfly(j << (16u - p.s), a[top], a[top + h]);
    a[top] = r[0];
    a[top + h] = r[1];
}

// Stages 1..=p.s inside one tile of min(N, TILE) elements, in workgroup
// memory. 256 threads; each does tile/2 / 256 butterflies per stage.
@compute @workgroup_size(256)
fn fft_local(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let size = 1u << min(p.log2n, LOG2_TILE);
    let tiles = (p.frames << p.log2n) / size;
    let t = group_index(wid, nwg);
    if (t >= tiles) {
        return; // uniform across the workgroup
    }
    let base = t * size;
    for (var i = li; i < size; i += 256u) {
        tile[i] = a[base + i];
    }
    workgroupBarrier();
    for (var s = 1u; s <= p.s; s++) {
        let h = 1u << (s - 1u);
        for (var q = li; q < size / 2u; q += 256u) {
            let j = q & (h - 1u);
            let top = ((q >> (s - 1u)) << s) + j;
            let r = butterfly(j << (16u - s), tile[top], tile[top + h]);
            tile[top] = r[0];
            tile[top + h] = r[1];
        }
        workgroupBarrier();
    }
    for (var i = li; i < size; i += 256u) {
        a[base + i] = tile[i];
    }
}

@compute @workgroup_size(256)
fn fft_extract(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = group_index(wid, nwg) * 256u + li;
    let bins = (1u << (p.log2n - 1u)) + 1u;
    if (g >= p.frames * bins) {
        return;
    }
    let frame = g / bins;
    let i = g - frame * bins;
    out[g] = a[(frame << p.log2n) + i];
}
