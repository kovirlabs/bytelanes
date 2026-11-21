# bytelanes

**Status: pre-alpha.** The op format and the CPU reference exist (0.0.1).
No GPU backend exists yet and nothing is measured.

Rust library of bulk data operations (checksum, filter, decrypt; later FFT and parity) that run on any GPU through wgpu. Every op has a CPU reference and a measured input size below which the CPU is used.

Part of the paralanes project: `paralanes` (umbrella: spec, plan, atlas,
network join, appliance image), `bytelanes` (GPU data ops), `stagelanes`
(stage placement runtime) and `diagsteno` (strain and vibration processing).

## Crates

| Crate | What it holds |
| --- | --- |
| `bytelanes-abi` | The op format: channel programs, commands, extents, completions, status codes. `no_std` |
| `bytelanes` | The `ChannelBackend` trait, the CPU reference backend (`bytelanes::cpu`) and the conformance harness |
| `bytelanes-wgpu` | GPU backend through wgpu (not built yet) |
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

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
