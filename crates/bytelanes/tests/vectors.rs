//! Replay the committed golden vectors on the CPU reference backend.

use bytelanes::abi::Status;
use bytelanes::conformance::{golden_cases, load_vectors, run_case, vectors_dir};
use bytelanes::cpu::CpuBackend;

#[test]
#[cfg_attr(miri, ignore = "reads files; covered natively")]
fn committed_vectors_replay_on_cpu() {
    let cases = load_vectors(vectors_dir()).expect("vectors load");
    assert!(cases.len() > 100, "only {} cases", cases.len());
    for case in &cases {
        let backend = CpuBackend::new();
        if let Err(m) = run_case(&backend, case) {
            panic!("{}: {m}", case.name);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "reads files; covered natively")]
fn committed_vectors_match_generator() {
    let mut disk = load_vectors(vectors_dir()).expect("vectors load");
    let mut built = golden_cases();
    disk.sort_by(|a, b| a.name.cmp(&b.name));
    built.sort_by(|a, b| a.name.cmp(&b.name));
    let disk_names: Vec<_> = disk.iter().map(|c| &c.name).collect();
    let built_names: Vec<_> = built.iter().map(|c| &c.name).collect();
    assert_eq!(
        disk_names, built_names,
        "vectors/ and golden_cases() list different cases"
    );
    for (d, b) in disk.iter().zip(&built) {
        assert!(
            d == b,
            "{} differs between vectors/ and golden_cases()",
            d.name
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "builds ~200 cases with up to 256 KiB each")]
fn golden_set_covers_every_status_a_cpu_can_report() {
    let cases = golden_cases();
    for s in [
        Status::Ok,
        Status::InvalidProgram,
        Status::UnsupportedOp,
        Status::UnknownBuffer,
        Status::OutOfBounds,
        Status::DstTooSmall,
        Status::InvalidLength,
        Status::UnknownKey,
    ] {
        assert!(
            cases.iter().any(|c| c.expect.status == s),
            "no case for {s:?}"
        );
    }
    let mut names: Vec<_> = cases.iter().map(|c| c.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), cases.len(), "duplicate case names");
}
