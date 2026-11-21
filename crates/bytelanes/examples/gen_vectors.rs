//! Regenerate the golden vectors in `vectors/` from the CPU reference.
//!
//! `cargo run -p bytelanes --example gen_vectors`
//!
//! After M1 the vectors are frozen: run this only as part of a contract pull
//! request that passes on all three machines.

#![forbid(unsafe_code)]

use std::fs;

use bytelanes::conformance::{golden_cases, vectors_dir, write_vectors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = vectors_dir();
    // Every subdirectory of vectors/ is generated; README.md is not.
    for entry in fs::read_dir(&dir)? {
        let path = entry?.path();
        if path.is_dir() {
            fs::remove_dir_all(&path)?;
        }
    }
    let cases = golden_cases();
    write_vectors(&dir, &cases)?;
    println!("wrote {} cases to {}", cases.len(), dir.display());
    Ok(())
}
