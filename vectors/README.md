# vectors

Golden inputs and expected outputs per op, written by the CPU reference
backend in `bytelanes`. Every backend must reproduce them exactly.

Frozen after M1: changes only through a pull request that passes on all
three machines. Cases are only ever added; an existing case's bytes never
change (M7 added `fft/` with its own seed for this reason). Regenerate with `cargo run -p bytelanes --example gen_vectors`
only as part of such a pull request; the test
`committed_vectors_match_generator` fails if this directory and
`bytelanes::conformance::golden_cases()` disagree.

## Groups

| Directory | Covers |
| --- | --- |
| `checksum/` | CRC-32C known answers (check value, RFC 3720 B.4) and random inputs from 0 bytes to 256 KiB |
| `crypt/` | AES-256-CTR: NIST SP 800-38A F.5.5 and F.5.6, random inputs, counter carry at 2^64 and wrap at 2^128, `Encrypt` |
| `filter/` | Every element type and comparison, float specials (NaN, -0.0, infinities, subnormals), integer extremes, empty and large columns |
| `fft/` | Fixed-point FFT (added in M7): impulse, constant and Nyquist known answers, random frames for every `log2n` from 1 to 16, full-scale square wave, saturation, length errors. Also holds `twiddles.bin`, the normative Q30 twiddle table (32,768 pairs of little-endian `i32`) |
| `program/` | Chains through `Vram`, unchained steps, failures in a later command, buffer and bounds errors, error precedence, reserved ops |
| `invalid/` | Programs that fail validation, submitted as raw bytes |

## Case layout

One directory per case:

```text
<group>/<case>/
  case.json       buffers, keys and the expected completion
  program.bin     the ChannelProgram wire bytes (1672)
  host-<id>.bin   initial contents of host buffer <id>
  key-<slot>.bin  a 32-byte AES-256 key (test keys only)
  expected.bin    the result bytes, present only when the status is Ok
```

`case.json` fields:

| Field | Meaning |
| --- | --- |
| `format` | Case file format, `1` |
| `host` | Host buffers to create: `buf` id and either `file` or `zeroed` (that many zero bytes) |
| `vram` | Device buffers to allocate, zero-filled: `buf` id and `len` |
| `keys` | Key slots to load: `slot` and `file` |
| `expect.status_code`, `expect.cmd_index`, `expect.bytes`, `expect.crc` | The completion fields the backend must report (`status` is the readable name) |
| `expect.output_buf`, `expect.output_offset`, `expect.output` | Where the result lands and the file holding its bytes |

To replay: create the buffers and keys, submit `program.bin` unchanged (some
programs are invalid on purpose), compare the completion's status, command
index, byte count and CRC, and on `Ok` compare the result bytes and check the
CRC against them. `bytelanes::conformance::run_case` does all of this for any
`ChannelBackend + Buffers`.

All data is generated from fixed seeds or taken from published test vectors.
