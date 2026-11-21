//! Known-answer tests: the CPU reference against published values.

use bytelanes::conformance::golden;
use bytelanes::cpu::ops;

#[test]
fn crc32c_known_answers() {
    for (name, input, want) in golden::crc_kats() {
        assert_eq!(ops::checksum(&input), want, "{name}");
    }
    assert_eq!(ops::checksum(&[]), 0);
}

#[test]
fn aes256_ctr_nist_sp800_38a_f55() {
    let mut out = [0u8; 64];
    ops::aes256_ctr(
        &golden::NIST_KEY,
        &golden::NIST_IV,
        &golden::NIST_PLAIN,
        &mut out,
    );
    assert_eq!(out, golden::NIST_CIPHER);
    ops::aes256_ctr_in_place(&golden::NIST_KEY, &golden::NIST_IV, &mut out);
    assert_eq!(out, golden::NIST_PLAIN);
}

#[test]
#[cfg_attr(miri, ignore = "builds the full golden set")]
fn golden_kat_cases_carry_the_published_outputs() {
    let cases = bytelanes::conformance::golden_cases();
    let get = |n: &str| {
        cases
            .iter()
            .find(|c| c.name == n)
            .unwrap_or_else(|| panic!("{n}"))
    };
    assert_eq!(
        get("crypt/kat-nist-f55-encrypt").expect.output,
        golden::NIST_CIPHER
    );
    assert_eq!(
        get("crypt/kat-nist-f56-decrypt").expect.output,
        golden::NIST_PLAIN
    );
    assert_eq!(
        get("checksum/kat-check-123456789").expect.output,
        0xE306_9283u32.to_le_bytes()
    );
}
