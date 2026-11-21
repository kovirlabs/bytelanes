# bytelanes

Rust library of bulk data operations (checksum, filter, decrypt; later
FFT and parity) that run on any GPU through wgpu. Every op has a CPU
reference and a measured input size below which the CPU is used.

This repo owns the **op format**, which is the contract between all four
repos.

Read `../paralanes/docs/PLAN.md` and `../paralanes/docs/SPEC.md` before any
change. The plan sets milestone order and which machine does what. The spec is
the technical source of truth and uses the real crate names; the earlier
`cf-` placeholder names appear in the table below for reference. Where plan and spec disagree, the
plan is current; note the conflict in your status file.

> DRAFT. Items marked TBD are open decisions for Evan.

## Repo map

| Repo | Role | Depends on | Licence |
| --- | --- | --- | --- |
| `paralanes` | Spec, plan, atlas, network join, appliance image, all status files | All three below | GPL-3.0-only |
| `bytelanes` | Bulk data ops on any GPU through wgpu, each with a CPU reference. Owns the op format, which is the contract between all repos | Nothing above it | Apache-2.0 |
| `stagelanes` | Topology-aware stage runtime for Windows and Linux: pinning, rings, pinned memory, stage placement | Nothing above it | Apache-2.0 |
| `diagsteno` | The Diagnostic Stenographer: strain gauge and accelerometer signal processing, the demo workload | `bytelanes`, `stagelanes` | GPL-3.0-only |

Sibling repos are expected as checkouts next to each other (`../paralanes`,
`../bytelanes`, `../stagelanes`, `../diagsteno`). Use path dependencies during
development; switch to published versions only when Evan says so.

## Crates

| Crate | Spec name | Kind | Owns | `unsafe` |
| --- | --- | --- | --- | --- |
| `bytelanes-abi` | `cf-abi` | `no_std` lib | Op format: channel program wire format, status codes, zero-copy structs | no |
| `bytelanes` | `cf-backend` | lib | `ChannelBackend` trait and the CPU reference backend (`bytelanes::cpu`) | no |
| `bytelanes-wgpu` | `cf-backend-wgpu` | lib | WGSL compute shaders through wgpu; `BufferStrategy` trait with copying and unified-memory implementations | no |
| `bytelanes-bench` | `cf-bench` | bin | GPU versus CPU benchmark per op across input sizes; feeds the atlas | no |
| `vectors/` | `vectors/` | data | Golden inputs and expected outputs per op | n/a |

## Milestones in this repo

| # | Milestone | Agent |
| --- | --- | --- |
| M1 | Op format and CPU reference: Checksum, Filter, Decrypt; golden vectors; 0.0.1 prepared | A (Dell) |
| M3 | GPU ops on Windows with the Radeon RX 570 | B (Ryzen) |
| M4 | GPU ops on Metal, plus a unified-memory buffer strategy | C (MacBook) |
| M5 | Benchmark runs for the atlas (atlas lives in `paralanes`). Release 1: `bytelanes` 0.1 | B, C |
| M7 | FFT op, built for `diagsteno` | A builds; B and C check |

## Which agent are you

Three agents work in parallel, one per machine. Identify your machine first
(`uname -sm`, `lscpu` or `sysctl -a | grep machdep.cpu`, `systeminfo` on
Windows, `vulkaninfo --summary`) and act only as that agent. If the milestone
you were given is not assigned to your machine in
`../paralanes/docs/PLAN.md`, stop and say so.

| Agent | Machine | OS |
| --- | --- | --- |
| A | Dell PowerEdge, 2x Xeon Silver (10 cores each), no GPU | Linux guest on Proxmox, cores pinned, both NUMA nodes exposed |
| B | Desktop, Ryzen 7 (model TBD) on AM5, Radeon RX 570 (4 or 8 GB, TBD) | Windows |
| C | MacBook Pro, M5 Pro, 24 GB unified memory | macOS, AArch64 |

Ownership inside this repo: agent B owns the ops and WGSL shaders in
`bytelanes-wgpu`. Agent C owns the unified-memory `BufferStrategy` module in
`bytelanes-wgpu` and does not edit shaders. On the Ryzen, GPU benchmarks (M3,
M5) must not run at the same time as `stagelanes` pipeline benchmarks (M2, M6).

## Hard rules

- **Agents never merge, tag or publish.** Prepare the pull request, changelog
  and version bump; Evan does the rest. Crates carry `publish = false` until
  Evan removes it.
- **One milestone per session.** Stop at the gate. Do not start the next
  milestone unprompted.
- **No unmeasured numbers.** A performance figure appears in a README, the
  atlas or a status file only with the run file that produced it. Every run
  file records machine, OS, kernel or build number, GPU driver version, wgpu
  backend, Rust version and the exact command.
- **A negative result is a result.** Publish it plainly.
- **Data rule.** Use generated data or data Evan confirms he owns. Never
  commit machine data from an employer or customer.
- **No `#[cfg(target_os)]` or `#[cfg(target_arch)]`** outside a module named
  `platform`.
- **Look up current docs** before using wgpu, tokio, io_uring or any
  networking crate API. Do not write them from memory.
- **Do not write NVIDIA or CUDA code.** That path is deferred until NVIDIA
  hardware exists.
- **The op format is the contract.** After M1, `bytelanes-abi` and `vectors/`
  change only through a pull request that passes on all three machines. If
  you need a change, write it under Requests in your status file and stop
  that task.
- **Every op has a CPU reference** in `bytelanes::cpu`. GPU output matches it
  byte for byte: replay `vectors/` plus 10,000 property-test cases per op.
- **Results are verified.** Every result leaving a GPU carries a CRC computed
  on the GPU that the receiver checks.
- `bytelanes-abi` stays `#![no_std]`.
- **No `unsafe`** in this repo; every crate has `#![forbid(unsafe_code)]`.
- **CPU reference speed comes from existing crates.** CRC32C uses `crc32c`;
  AES-256-CTR uses RustCrypto's `aes` and `ctr`. They pick the hardware
  instructions at runtime, so the atlas compares the GPU against a fast CPU.
  Do not depend on `stagelanes`; look up the crates' current docs first.

## Status files

All status files live in the umbrella repo: write
`../paralanes/docs/status/M<n>.md` (shape defined in `../paralanes/CLAUDE.md`),
whichever repo the work was in, and commit it on your milestone branch there.

At the start of every session: rebase your branch on `main`, then read every
file in `../paralanes/docs/status/`.

## Branches

- `main` is Evan's. Never push to it.
- Work on `m<n>-<short-topic>` (for example `m3-rx570-ops`). One branch per
  milestone per repo.
- Small commits, one logical change each. Run fmt, clippy and tests before
  each commit.

## Commands

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo +nightly miri test -p bytelanes-abi -p bytelanes
```

Benchmark output goes to `../paralanes/atlas/runs/<machine>/<YYYY-MM-DD>-<op>.json`;
do not edit run files after they are committed.

## Release rule

Agents prepare releases (changelog, README, version bump in a pull request).
Only Evan merges, tags and publishes.

## Open decisions (TBD)

- GitHub owner for the four repos
- Ryzen 7 model and RX 570 memory size, for the atlas
