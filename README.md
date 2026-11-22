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
| `bytelanes-bench` | GPU versus CPU benchmark per op across input sizes, plus a commit-log workload; writes atlas run files |

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
cargo run --release -p bytelanes-wgpu --example strategy_cost -- --out run.json   # copy vs no-copy cost
```

Status checks run on the host in the CPU reference's order. All data work,
including the CRC-32C of every result, runs on the GPU. Filter compares
floats through integer keys, so NaN, signed zero and subnormals follow IEEE 754
whatever the GPU's float hardware does, and 64-bit types work on GPUs
without 64-bit integer or float support. On a machine with no GPU the GPU
tests print `SKIPPED` and pass.

Device buffers and transfer staging are pooled and reused across programs.
Consecutive programs overlap: the next program's GPU work is submitted
before the previous program's result is copied out. A program that reads
host bytes the previous one writes waits for it, so results never change.
`WgpuBackend::set_overlap(false)` turns overlap off.

Checked so far on two GPUs: AMD Radeon RX 7800 XT (Windows, Vulkan and
DX12) and Apple M5 Pro (macOS on AArch64, Metal). The fixed-point FFT is
checked on the RX 7800 XT only so far; Metal ran before the GPU FFT existed.

### Buffer strategies

How bytes reach the GPU and come back is a `BufferStrategy`, chosen with
`WgpuBackend::with_strategy`:

- `CopyingStrategy` (the default) stages every transfer: a CPU copy into a
  staging buffer and a GPU copy on the way in, a GPU copy into a mappable
  buffer and a CPU copy on the way out. A discrete GPU on PCIe needs this.
- `UnifiedMemoryStrategy` is for GPUs that share memory with the CPU, such
  as Apple silicon through Metal. It writes inputs straight into the GPU's
  buffer and reads results straight out of it, so the staging buffers and
  GPU copies go away. One CPU copy per direction remains, because host
  buffers are ordinary `Vec<u8>`s; removing it would need wgpu's `unsafe`
  `create_buffer_from_hal`, and this crate forbids `unsafe`.
  `UnifiedMemoryStrategy::for_adapter` refuses discrete GPUs.

```text
let strategy = UnifiedMemoryStrategy::for_adapter(&adapter)?;
let gpu = WgpuBackend::with_strategy(&adapter, Box::new(strategy), DEFAULT_QUEUE_DEPTH)?;
```

The conformance and large-input tests run every strategy that applies to
each adapter. What each strategy costs is measured, not stated here: see
the M4 run file in the `paralanes` atlas.

### GPU CI

GitHub's hosted runners have no GPU, so CI's `gpu` job runs on a
self-hosted runner: the Ryzen desktop with the RX 7800 XT (labels
`self-hosted, Windows, X64, gpu`). It runs the conformance tests on every
pull request from this repo, and the large-input tests on pushes to `main`
and manual runs. It sets `BYTELANES_REQUIRE_BACKENDS=vulkan,dx12`, so a
missing driver or adapter fails the job instead of skipping it.

The runner lives in `C:\Users\gress\actions-runner\bytelanes` and starts at
logon from the shortcut `GitHub runner bytelanes-gpu` in the user's Startup
folder, which runs `start-runner.cmd` (log: `runner.log` there). Jobs wait
while the machine is off or logged out. Stop the runner during pipeline or
GPU benchmarks on that machine, so a CI job can't disturb a measurement:

```powershell
Stop-Process -Name Runner.Listener,Runner.Worker -ErrorAction SilentlyContinue
Start-Process C:\Users\gress\actions-runner\bytelanes\start-runner.cmd -WindowStyle Minimized   # afterwards
```

## Benchmarks

`bytelanes-bench` measures the CPU reference and every hardware GPU adapter
through the same `ChannelBackend` API, so GPU times include every transfer.
Every completion must match the CPU reference, or the run aborts. It
records latency percentiles and the CPU time spent on the work, and writes
one run file per op to `../paralanes/atlas/runs/<machine>/`.

```text
cargo run --release -p bytelanes-bench -- ops       --machine ryzen --host-link "PCIe 3.0 x16" [--note ...]
cargo run --release -p bytelanes-bench -- commitlog --machine ryzen --host-link "PCIe 3.0 x16"
cargo run --release -p bytelanes-bench -- stream    --machine ryzen --host-link "PCIe 3.0 x16"
cargo run --release -p bytelanes-bench -- ops ... --ops filter --sizes 16M,24M,32M,64M   # explicit sizes
```

`stream` measures throughput: 16 commit-log batches in flight per
repetition, the GPU with program overlap off and on, and the CPU reference on
1, half and all logical CPUs.

- `ops` runs checksum, encrypt, decrypt, filter and FFT (4,096-sample frames) from 64 B to 256 MiB.
- `commitlog` is a transactional workload: batches of generated 256-byte
  payment records, from 1 to about 1M per batch. It has two paths:
  - **Write:** a fraud-rule filter, plus encrypting the records for the log.
  - **Read:** decrypt an encrypted column and filter it on the device, as
    one chained program.
- Results and crossover sizes are in the paralanes atlas
  (`atlas/crossover.md`). No figures appear here without their run file.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
