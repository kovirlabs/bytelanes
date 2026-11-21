// Filter: ascending indices of rows that match a predicate.
//
// Every element becomes a 64-bit ordered key (hi, lo) plus a NaN flag,
// using integer operations only, so the result does not depend on how a
// GPU treats NaN, signed zero or subnormal floats:
//   u32 (0, x)          i32 (0, x ^ 2^31)          f32 (0, key32(x))
//   u64 (hi, lo)        i64 (hi ^ 2^31, lo)        f64 key64(hi, lo)
// Float keys: -0.0 becomes +0.0; negative values have all bits flipped;
// non-negative values get the sign bit set. Unsigned order of keys is then
// numeric order. Comparisons with a NaN are false, except Ne, which is true.
//
// Three passes over tiles of TILE rows (one workgroup per tile):
// `count` counts matches per tile, `scan` turns counts into offsets (one
// workgroup), `scatter` writes indices at those offsets in row order.

const THREADS: u32 = 256u;
const PER_THREAD: u32 = 4u;
const TILE: u32 = 1024u; // THREADS * PER_THREAD

struct Params {
    off: u32,    // input byte offset in `src`
    rows: u32,
    width: u32,  // 4 or 8 bytes
    kind: u32,   // 0 unsigned, 1 signed, 2 float
    cmp: u32,    // 0 Eq, 1 Ne, 2 Lt, 3 Le, 4 Gt, 5 Ge, 6 Between
    tiles: u32,
    a_nan: u32,
    b_nan: u32,
    a: vec2<u32>, // key (hi, lo)
    b: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> sh: array<u32, 256>;

struct Key {
    k: vec2<u32>,
    nan: bool,
}

fn key_of(row: u32) -> Key {
    let at = p.off + row * p.width;
    var lo = load_word(at);
    var hi = 0u;
    if (p.width == 8u) {
        hi = load_word(at + 4u);
    }
    var nan = false;
    if (p.kind == 1u) {
        if (p.width == 8u) {
            hi ^= 0x80000000u;
        } else {
            lo ^= 0x80000000u;
        }
    } else if (p.kind == 2u) {
        if (p.width == 4u) {
            let mag = lo & 0x7fffffffu;
            nan = mag > 0x7f800000u;
            if (mag == 0u) {
                lo = 0u;
            }
            lo = select(lo | 0x80000000u, ~lo, (lo & 0x80000000u) != 0u);
        } else {
            let mag = hi & 0x7fffffffu;
            nan = mag > 0x7ff00000u || (mag == 0x7ff00000u && lo != 0u);
            if (mag == 0u && lo == 0u) {
                hi = 0u;
            }
            if ((hi & 0x80000000u) != 0u) {
                hi = ~hi;
                lo = ~lo;
            } else {
                hi |= 0x80000000u;
            }
        }
    }
    return Key(vec2<u32>(hi, lo), nan);
}

fn less(x: vec2<u32>, y: vec2<u32>) -> bool {
    return x.x < y.x || (x.x == y.x && x.y < y.y);
}

fn matches(row: u32) -> bool {
    let e = key_of(row);
    let x = e.k;
    let any_nan = e.nan || p.a_nan != 0u;
    let eq = x.x == p.a.x && x.y == p.a.y;
    switch p.cmp {
        case 0u: { return !any_nan && eq; }
        case 1u: { return any_nan || !eq; }
        case 2u: { return !any_nan && less(x, p.a); }
        case 3u: { return !any_nan && !less(p.a, x); }
        case 4u: { return !any_nan && less(p.a, x); }
        case 5u: { return !any_nan && !less(x, p.a); }
        default: {
            return !any_nan && p.b_nan == 0u && !less(x, p.a) && !less(p.b, x);
        }
    }
}

// Matches among this thread's rows of tile `tile`, as a bit mask.
fn thread_mask(tile: u32, li: u32) -> u32 {
    var mask = 0u;
    let first = tile * TILE + li * PER_THREAD;
    for (var k = 0u; k < PER_THREAD; k++) {
        let row = first + k;
        if (row < p.rows && matches(row)) {
            mask |= 1u << k;
        }
    }
    return mask;
}

// Inclusive scan of sh[0..256] in place. Call from all threads.
fn scan_shared(li: u32) {
    for (var d = 1u; d < THREADS; d <<= 1u) {
        var v = 0u;
        if (li >= d) {
            v = sh[li - d];
        }
        workgroupBarrier();
        sh[li] += v;
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn count(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tile = group_index(wid, nwg);
    if (tile >= p.tiles) {
        return;
    }
    sh[li] = countOneBits(thread_mask(tile, li));
    workgroupBarrier();
    for (var s = THREADS / 2u; s > 0u; s >>= 1u) {
        if (li < s) {
            sh[li] += sh[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        counts[tile] = sh[0];
    }
}

// One workgroup: counts[0..tiles] becomes exclusive offsets, counts[tiles]
// the total.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) li: u32) {
    var carry = 0u;
    for (var base = 0u; base < p.tiles; base += THREADS) {
        let i = base + li;
        var v = 0u;
        if (i < p.tiles) {
            v = counts[i];
        }
        sh[li] = v;
        workgroupBarrier();
        scan_shared(li);
        if (i < p.tiles) {
            counts[i] = carry + sh[li] - v;
        }
        carry += sh[THREADS - 1u];
        workgroupBarrier();
    }
    if (li == 0u) {
        counts[p.tiles] = carry;
    }
}

@compute @workgroup_size(256)
fn scatter(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tile = group_index(wid, nwg);
    if (tile >= p.tiles) {
        return;
    }
    let mask = thread_mask(tile, li);
    let c = countOneBits(mask);
    sh[li] = c;
    workgroupBarrier();
    scan_shared(li);
    var at = counts[tile] + sh[li] - c;
    let first = tile * TILE + li * PER_THREAD;
    for (var k = 0u; k < PER_THREAD; k++) {
        if ((mask & (1u << k)) != 0u) {
            out[at] = first + k;
            at += 1u;
        }
    }
}
