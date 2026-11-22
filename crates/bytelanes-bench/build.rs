//! Records the compiler and wgpu versions in the binary, for run files.

use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=BENCH_RUSTC={version}");

    // The wgpu version actually locked, from the workspace Cargo.lock.
    let lock = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default())
        .join("../../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = std::fs::read_to_string(&lock).unwrap_or_default();
    let mut wgpu = "unknown".to_owned();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line == "name = \"wgpu\"" {
            if let Some(v) = lines.next().and_then(|l| l.strip_prefix("version = ")) {
                wgpu = v.trim_matches('"').to_owned();
            }
            break;
        }
    }
    println!("cargo:rustc-env=BENCH_WGPU={wgpu}");
    println!("cargo:rerun-if-changed=build.rs");
}
