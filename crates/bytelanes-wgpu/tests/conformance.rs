//! R7 on real GPUs: every golden vector, then 10,000 seeded random cases per
//! op, on every hardware adapter wgpu finds (Vulkan and DX12 on Windows,
//! Metal on macOS). `WGPU_BACKEND=vulkan` or `=dx12` narrows the set.
//! `BYTELANES_CASES=<n>` changes the number of random cases per op.
//!
//! Machines without a GPU (hosted CI) find no adapter and skip, saying so.
//! `BYTELANES_REQUIRE_BACKENDS=vulkan,dx12` turns a missing adapter into a
//! failure (the GPU runner sets it).

mod common;

use std::time::Instant;

use bytelanes::abi::Op;
use bytelanes::conformance::{self, run_case};
use bytelanes_wgpu::WgpuBackend;

fn backends() -> Vec<WgpuBackend> {
    common::adapters()
        .iter()
        .map(|a| WgpuBackend::new(a).expect("open adapter"))
        .collect()
}

fn describe(b: &WgpuBackend) -> String {
    let i = b.adapter_info();
    format!(
        "{} | {:?} | driver {} {} | {:?}",
        i.name, i.backend, i.driver, i.driver_info, i.device_type
    )
}

#[test]
fn golden_vectors() {
    let cases = conformance::load_vectors(conformance::vectors_dir()).expect("load vectors");
    for backend in backends() {
        let start = Instant::now();
        // Each case starts from an empty backend, as with a fresh CpuBackend.
        let failures: Vec<String> = cases
            .iter()
            .filter_map(|c| {
                backend.clear();
                run_case(&backend, c)
                    .err()
                    .map(|e| format!("{}: {e}", c.name))
            })
            .collect();
        eprintln!(
            "{}: {} golden cases, {} failed, {:.1} s",
            describe(&backend),
            cases.len(),
            failures.len(),
            start.elapsed().as_secs_f64()
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

#[test]
fn random_cases() {
    let n: u64 = std::env::var("BYTELANES_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    for backend in backends() {
        for op in [Op::Checksum, Op::Encrypt, Op::Decrypt, Op::Filter] {
            let start = Instant::now();
            let mut failures = Vec::new();
            for seed in 0..n {
                let case = conformance::random_case(op, seed);
                if let Err(e) = run_case(&backend, &case) {
                    failures.push(format!("{}: {e}", case.name));
                    if failures.len() >= 20 {
                        break;
                    }
                }
            }
            eprintln!(
                "{}: {op:?} seeds 0..{n}, {} failed, {:.1} s",
                describe(&backend),
                failures.len(),
                start.elapsed().as_secs_f64()
            );
            assert!(failures.is_empty(), "{}", failures.join("\n"));
        }
    }
}
