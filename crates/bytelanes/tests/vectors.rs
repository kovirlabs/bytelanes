//! Replay the committed golden vectors on the CPU reference backend.

use bytelanes::abi::Status;
use bytelanes::conformance::{golden_cases, load_vectors, run_case, vectors_dir};
use bytelanes::cpu::CpuBackend;

#[test]
#[cfg_attr(miri, ignore = "reads files; covered natively")]
fn committed_vectors_replay_on_cpu() {
    let cases = load_vectors(vectors_dir()).expect("vectors load");
    assert!(cases.len() >= 171, "only {} cases", cases.len());
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

#[test]
#[cfg_attr(miri, ignore = "builds the full golden set")]
fn cases_report_their_ops_for_capability_filtering() {
    use bytelanes::ChannelBackend;
    use bytelanes::abi::{Caps, ExtentKind, Features, Op};
    let cases = golden_cases();
    let cpu = CpuBackend::new().capabilities();
    assert!(
        cases.iter().all(|c| c.runs_on(&cpu)),
        "the CPU reference runs every case"
    );
    let no_fft = Caps::new(
        &[Op::Checksum, Op::Encrypt, Op::Decrypt, Op::Filter],
        &[ExtentKind::Host, ExtentKind::Vram],
        Features::NONE,
        1,
    );
    let skipped: Vec<_> = cases.iter().filter(|c| !c.runs_on(&no_fft)).collect();
    // The 30 fft/ cases, the decrypt-fft-checksum chain and the invalid
    // log2n case use Fft.
    assert_eq!(skipped.len(), 32);
    assert!(skipped.iter().any(|c| c.name == "invalid/fft-log2n-17"));
    assert!(skipped.iter().all(|c| c.ops().contains(&Op::Fft)));
}
