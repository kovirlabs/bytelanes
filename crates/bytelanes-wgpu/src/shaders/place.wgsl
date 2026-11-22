// Writes `n` bytes from the start of `src` into `dst` at byte offset
// `dst_off`, leaving the surrounding bytes of `dst` unchanged. One thread
// owns one destination word, so unaligned edges need no atomics.

struct Params {
    dst_off: u32,
    n: u32,
    first_word: u32, // dst_off / 4
    words: u32,      // destination words touched
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = group_index(wid, nwg) * 64u + li;
    if (g >= p.words) {
        return;
    }
    let w = p.first_word + g;
    var v = dst[w];
    for (var k = 0u; k < 4u; k++) {
        let b = 4u * w + k;
        if (b >= p.dst_off && b < p.dst_off + p.n) {
            let s = 8u * k;
            v = (v & ~(0xffu << s)) | (load_byte(b - p.dst_off) << s);
        }
    }
    dst[w] = v;
}
