# Changelog

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
