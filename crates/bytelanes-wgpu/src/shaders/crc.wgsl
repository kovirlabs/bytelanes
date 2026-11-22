// CRC-32C (Castagnoli), bit-reflected, init and final xor 0xFFFFFFFF.
//
// `chunks`: each thread computes the raw CRC (register starting at 0, no
// final xor) of one chunk of the input, a word at a time with slicing-by-4
// tables, then any tail bytes with the byte table.
// `combine`: pairs of (raw CRC, length) merge: r(A||B) = r(A) * x^(8|B|) ^ r(B).
// `finalize`: crc(M) = ~(0xFFFFFFFF * x^(8n) ^ r(M)), written as one word.

struct Params {
    off: u32,     // input byte offset in `src`
    n: u32,       // input length in bytes
    chunk: u32,   // bytes per chunk
    count: u32,   // chunks (chunks), input nodes (combine)
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
// Slicing-by-4 tables: [0..256) is the byte table, [256k..256k+256) is
// table k.
@group(0) @binding(1) var<storage, read> table: array<u32, 1024>;
@group(0) @binding(2) var<storage, read> powers: array<u32, 32>;
@group(0) @binding(3) var<storage, read> nodes_in: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> nodes_out: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> p: Params;

const POLY: u32 = 0x82f63b78u;

var<workgroup> tbl: array<u32, 1024>;

// a(x) * b(x) mod P(x), bit 31 is x^0.
fn multmodp(a: u32, b_in: u32) -> u32 {
    if (a == 0u) {
        return 0u;
    }
    var b = b_in;
    var m = 0x80000000u;
    var r = 0u;
    loop {
        if ((a & m) != 0u) {
            r ^= b;
            if ((a & (m - 1u)) == 0u) {
                break;
            }
        }
        m >>= 1u;
        b = select(b >> 1u, (b >> 1u) ^ POLY, (b & 1u) != 0u);
    }
    return r;
}

// r * x^(8 * len) mod P: the register after `len` more zero bytes.
fn shift(r: u32, len: u32) -> u32 {
    var x = 0x80000000u;
    for (var k = 0u; k < 32u; k++) {
        if (((len >> k) & 1u) != 0u) {
            x = multmodp(powers[k], x);
        }
    }
    return multmodp(x, r);
}

@compute @workgroup_size(64)
fn chunks(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    for (var i = li; i < 1024u; i += 64u) {
        tbl[i] = table[i];
    }
    workgroupBarrier();
    let g = group_index(wid, nwg) * 64u + li;
    if (g >= p.count) {
        return;
    }
    let start = g * p.chunk;
    let end = min(start + p.chunk, p.n);
    var r = 0u;
    var i = start;
    // Whole words: load_word handles any byte alignment of the input.
    for (; i + 4u <= end; i += 4u) {
        r ^= load_word(p.off + i);
        r = tbl[768u + (r & 0xffu)] ^ tbl[512u + ((r >> 8u) & 0xffu)]
            ^ tbl[256u + ((r >> 16u) & 0xffu)] ^ tbl[r >> 24u];
    }
    for (; i < end; i++) {
        r = tbl[(r ^ load_byte(p.off + i)) & 0xffu] ^ (r >> 8u);
    }
    nodes_out[g] = vec2<u32>(r, end - start);
}

@compute @workgroup_size(64)
fn combine(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = group_index(wid, nwg) * 64u + li;
    let pairs = (p.count + 1u) / 2u;
    if (g >= pairs) {
        return;
    }
    let l = nodes_in[2u * g];
    if (2u * g + 1u >= p.count) {
        nodes_out[g] = l;
        return;
    }
    let r = nodes_in[2u * g + 1u];
    nodes_out[g] = vec2<u32>(shift(l.x, r.y) ^ r.x, l.y + r.y);
}

@compute @workgroup_size(1)
fn finalize() {
    var r0 = 0u;
    if (p.count > 0u) {
        r0 = nodes_in[0].x;
    }
    let crc = ~(shift(0xffffffffu, p.n) ^ r0);
    nodes_out[0] = vec2<u32>(crc, 4u);
}
