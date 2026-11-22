# bytelanes

**Status: pre-alpha.** The op format and the CPU reference exist (0.0.1).
The wgpu GPU backend reproduces the CPU reference on the Radeon RX 7800 XT
(Vulkan and DX12). Nothing is measured yet: there are no speed figures.

Rust library of bulk data operations (checksum, filter, decrypt; later FFT and parity) that run on any GPU through wgpu. Every op has a CPU reference and a measured input size below which the CPU is used.

Part of the paralanes project: `paralanes` (umbrella: spec, plan, atlas,
network join, appliance image), `bytelanes` (GPU data ops), `stagelanes`
(stage placement runtime) and `diagsteno` (strain and vibration processing).

## Crates

| Crate | What it holds |
| --- | --- |
| `bytelanes-abi` | The op format: channel programs, commands, extents, completions, status codes. `no_std` |
| `bytelanes` | The `ChannelBackend` trait, the CPU reference backend (`bytelanes::cpu`) and the conformance harness |
| `bytelanes-wgpu` | GPU backend through wgpu: WGSL shaders for every op, a `BufferStrategy` seam for how bytes reach the GPU |
| `bytelanes-bench` | GPU versus CPU benchmark (not built yet) |

`vectors/` holds the golden vectors every backend must reproduce byte for byte;
see [`vectors/README.md`](vectors/README.md).

```rust
use bytelanes::abi::{BufId, ChannelCmd, ChannelProgram, Extent, Tag};
use bytelanes::cpu::CpuBackend;
use bytelanes::Buffers;

let cpu = CpuBackend::new();
cpu.put_host(BufId(1), b"123456789".to_vec()).unwrap();
cpu.put_host(BufId(2), vec![0; 4]).unwrap();
let prog = ChannelProgram::from_commands(&[ChannelCmd::checksum(
    Extent::host(BufId(1), 0, 9),
    Extent::host(BufId(2), 0, 4),
)])
.unwrap();
let done = cpu.run(&prog, Tag(1));
let result = cpu.get_host(BufId(2)).unwrap();
bytelanes::verify(&done, &result).unwrap();
assert_eq!(result, 0xE306_9283u32.to_le_bytes());
```

## GPU backend

```text
cargo run -p bytelanes-wgpu --example adapters          # GPUs found, with limits
cargo test -p bytelanes-wgpu --release                   # golden vectors + 10,000 cases per op, every GPU
cargo test -p bytelanes-wgpu --release --test large -- --ignored   # inputs of hundreds of MB
WGPU_BACKEND=dx12 cargo test -p bytelanes-wgpu --release # one API only
```

Status checks run on the host in the CPU reference's order. All data work,
including the CRC-32C of every result, runs on the GPU. Filter compares
floats through integer keys, so NaN, signed zero and subnormals follow IEEE 754
whatever the GPU's float hardware does, and 64-bit types work on GPUs
without 64-bit integer or float support. On a machine with no GPU the GPU
tests print `SKIPPED` and pass.

Checked so far on one GPU: AMD Radeon RX 7800 XT, Windows, Vulkan and DX12.
Metal (MacBook) is M4.

### GPU CI

GitHub's hosted runners have no GPU, so CI's `gpu` job runs on a
self-hosted runner: the Ryzen desktop with the RX 7800 XT (labels
`self-hosted, Windows, X64, gpu`). It runs the conformance tests on every
pull request from this repo, and the large-input tests on pushes to `main`
and manual runs. It sets `BYTELANES_REQUIRE_BACKENDS=vulkan,dx12`, so a
missing driver or adapter fails the job instead of skipping it.

The runner lives in `C:\Users\gress\actions-runner\bytelanes` and starts at
logon from the scheduled task `GitHub runner bytelanes-gpu` (log:
`runner.log` there). Jobs wait while the machine is off or logged out.
Stop the runner during pipeline or GPU benchmarks on that machine, so a CI
job can't disturb a measurement:

```powershell
Stop-ScheduledTask 'GitHub runner bytelanes-gpu'; Stop-Process -Name Runner.Listener,Runner.Worker -ErrorAction SilentlyContinue
Start-ScheduledTask 'GitHub runner bytelanes-gpu'   # afterwards
```

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
