# Changelog

## Unreleased

### bytelanes-abi (M7, contract change)

- `Op::Fft` (code 10) and `FftParams`: a fixed-point FFT of real `i32` frames
  of `2^log2n` samples (`log2n` 1 to 16) into half-spectrum `(i32, i32)`
  bins. The algorithm, including every rounding step, is normative, so GPU
  backends reproduce it byte for byte; WGSL does not guarantee exact float
  arithmetic, so a float FFT could not be.

### bytelanes

- `cpu::fft`: the reference FFT and its Q30 twiddle table, built from a
  Taylor series with plain `f64` arithmetic so it is identical on every
  platform.
- `CpuBackend` runs `Fft`; `random_case` covers it.

### vectors

- 32 new cases: 30 under `fft/`, `program/decrypt-fft-checksum` and
  `invalid/fft-log2n-17`. Adds
  `fft/twiddles.bin`. The 139 M1 cases are unchanged.

### bytelanes-wgpu (M3, agent B)

- `WgpuBackend`: a `ChannelBackend` and `Buffers` implementation on wgpu 30
  that reproduces the CPU reference byte for byte. Checksum (CRC-32C, chunk
  CRCs combined on the GPU), Encrypt and Decrypt (AES-256-CTR), Filter (all
  six element types through integer order keys; count, scan and scatter
  passes). Every result's CRC-32C is computed on the GPU. Programs run in
  order on a worker thread; `submit` and `poll` do not wait for the GPU.
- `BufferStrategy` trait and `CopyingStrategy`, the seam for a
  unified-memory strategy.
- `hardware_adapters()`, `adapters()`, and the `adapters` example.
- Tests: all golden vectors and 10,000 random cases per op on every hardware
  adapter; large-input tests (`--ignored`).
- `bytelanes-wgpu` needs Rust 1.87 (wgpu 30); the other crates stay on 1.85.
- Until a WGSL FFT exists, `WgpuBackend` does not list `Fft` in its
  capabilities and completes it with `UnsupportedOp`; its golden-vector test
  skips cases that use ops the backend does not list.

## 0.0.1 (unreleased)

First version, published to hold the crate names. No GPU backend and no
performance figures yet.

### bytelanes-abi

- The op format, version 1: `ChannelProgram` (up to 16 commands), `ChannelCmd`,
  `Extent` (`Host`, `Vram`, `Nvme`), `CryptParams`, `FilterParams`,
  `Completion`, `Status`, `Caps` and `PathState`. Fixed-size, little-endian,
  alignment-1 wire structs via `zerocopy`; `no_std`; no `unsafe`.
- Defined op semantics: `Checksum` (CRC-32C), `Encrypt` and `Decrypt`
  (AES-256-CTR, 128-bit big-endian counter) and `Filter` (fixed-width column
  compare, ascending `u32` row indices). Other ops are reserved names.
- Program validation: intermediate results stay in `Vram`, chain rules, and
  no `dst` may overlap a `src`, so every valid program is idempotent.
- A fixed order of per-command error checks, so backends agree on status.

### bytelanes

- `ChannelBackend` and `Buffers` traits, and `verify` for result CRCs.
- `cpu::ops`: the CPU reference, using `crc32c`, `aes` and `ctr`.
- `cpu::CpuBackend`: runs whole channel programs on the reference.
- `conformance`: golden-vector loading, `run_case` for any backend, and
  seeded `random_case` generation for 10,000-case runs per op.

### vectors

- 139 golden cases across checksum, crypt, filter, multi-command programs and
  invalid programs, including NIST SP 800-38A and RFC 3720 known answers.
