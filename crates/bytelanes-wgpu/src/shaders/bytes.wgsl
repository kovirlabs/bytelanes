// Byte access to a storage buffer of u32 words named `src`, little-endian.
// Appended to shaders that declare `src`. Inputs may start at any byte
// offset; buffers are padded so a word read past the end stays in bounds.

fn load_byte(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

// The four bytes starting at byte index `i`, as a little-endian word.
fn load_word(i: u32) -> u32 {
    let s = (i & 3u) * 8u;
    let w0 = src[i >> 2u];
    if (s == 0u) {
        return w0;
    }
    let w1 = src[(i >> 2u) + 1u];
    return (w0 >> s) | (w1 << (32u - s));
}

// Flattened workgroup index for 2D dispatches (more than 65535 groups).
fn group_index(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.y * nwg.x + wid.x;
}
