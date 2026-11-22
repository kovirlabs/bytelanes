//! FFT reference tests: known answers, an independent recursive
//! implementation, bounds against an f64 DFT, and the twiddle table.

use bytelanes::cpu::fft::{self, SAMPLE_LIMIT, TWIDDLES, fft_frame, twiddles};
use proptest::prelude::*;

fn cases() -> u32 {
    if cfg!(miri) { 4 } else { 10_000 }
}

fn cfg() -> ProptestConfig {
    ProptestConfig {
        cases: cases(),
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

fn run(log2n: u8, x: &[i32]) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    fft_frame(log2n, x, &mut out);
    out
}

/// The same algorithm written recursively (split into even and odd halves),
/// from the definition rather than the iterative loop. Its per-element
/// arithmetic must agree exactly with the reference.
fn recursive(x: &[i64], w_step: usize) -> Vec<(i64, i64)> {
    let n = x.len();
    if n == 1 {
        return vec![(x[0], 0)];
    }
    let even: Vec<i64> = x.iter().step_by(2).copied().collect();
    let odd: Vec<i64> = x.iter().skip(1).step_by(2).copied().collect();
    let e = recursive(&even, w_step * 2);
    let o = recursive(&odd, w_step * 2);
    let w = twiddles();
    let mut out = vec![(0, 0); n];
    for k in 0..n / 2 {
        let (wr, wi) = w[k * w_step];
        let (wr, wi) = (i64::from(wr), i64::from(wi));
        let (br, bi) = o[k];
        let tr = (wr * br - wi * bi + (1 << 29)) >> 30;
        let ti = (wr * bi + wi * br + (1 << 29)) >> 30;
        let (ur, ui) = e[k];
        out[k] = ((ur + tr) >> 1, (ui + ti) >> 1);
        out[k + n / 2] = ((ur - tr) >> 1, (ui - ti) >> 1);
    }
    out
}

fn recursive_fft(log2n: u8, x: &[i32]) -> Vec<(i32, i32)> {
    let xs: Vec<i64> = x
        .iter()
        .map(|&v| i64::from(v.clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT)))
        .collect();
    let full = recursive(&xs, 65536 >> log2n);
    full[..=(1usize << log2n) / 2]
        .iter()
        .map(|&(r, i)| (r as i32, i as i32))
        .collect()
}

/// f64 DFT of the clamped input, divided by N.
fn dft(x: &[i32]) -> Vec<(f64, f64)> {
    let n = x.len();
    (0..=n / 2)
        .map(|k| {
            let mut re = 0.0;
            let mut im = 0.0;
            for (t, &v) in x.iter().enumerate() {
                let v = f64::from(v.clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT));
                let ang = -std::f64::consts::TAU * ((k * t) % n) as f64 / n as f64;
                re += v * ang.cos();
                im += v * ang.sin();
            }
            (re / n as f64, im / n as f64)
        })
        .collect()
}

#[test]
fn twiddle_table_matches_committed_bytes() {
    let path = bytelanes::conformance::vectors_dir().join("fft/twiddles.bin");
    if cfg!(miri) {
        return;
    }
    let committed = std::fs::read(&path).expect("vectors/fft/twiddles.bin");
    assert_eq!(
        committed,
        fft::twiddle_bytes(),
        "twiddle table differs on this platform"
    );
}

#[test]
#[cfg_attr(miri, ignore = "32,768 libm calls")]
fn twiddles_are_within_one_lsb_of_libm() {
    let w = twiddles();
    assert_eq!(w.len(), TWIDDLES);
    assert_eq!(w[0], (1 << 30, 0));
    assert_eq!(w[16384], (0, -(1 << 30)));
    for (t, &(c, s)) in w.iter().enumerate() {
        let th = std::f64::consts::TAU * t as f64 / 65536.0;
        let ec = (f64::from(c) - th.cos() * 2f64.powi(30)).abs();
        let es = (f64::from(s) + th.sin() * 2f64.powi(30)).abs();
        assert!(ec <= 0.5 + 1e-6 && es <= 0.5 + 1e-6, "t={t}: {c},{s}");
    }
}

#[test]
fn impulse_gives_flat_spectrum() {
    for log2n in 1..=12u8 {
        let n = 1usize << log2n;
        let mut x = vec![0; n];
        x[0] = SAMPLE_LIMIT;
        let want = SAMPLE_LIMIT >> log2n;
        assert!(
            run(log2n, &x).iter().all(|&b| b == (want, 0)),
            "log2n={log2n}"
        );
    }
}

#[test]
fn dc_and_nyquist_are_exact() {
    for log2n in 1..=12u8 {
        let n = 1usize << log2n;
        let c = 123_456_789;
        let out = run(log2n, &vec![c; n]);
        assert_eq!(out[0], (c, 0));
        assert!(out[1..].iter().all(|&b| b == (0, 0)), "dc log2n={log2n}");
        let alt: Vec<i32> = (0..n).map(|i| if i % 2 == 0 { c } else { -c }).collect();
        let out = run(log2n, &alt);
        assert_eq!(out[n / 2], (c, 0), "nyquist log2n={log2n}");
        assert!(out[..n / 2].iter().all(|&b| b == (0, 0)));
    }
}

#[test]
fn inputs_saturate_at_2_pow_30() {
    let x = vec![i32::MAX, i32::MIN, 5, -5];
    let y = vec![SAMPLE_LIMIT, -SAMPLE_LIMIT, 5, -5];
    assert_eq!(run(2, &x), run(2, &y));
}

#[test]
#[cfg_attr(miri, ignore = "large frames")]
fn extreme_inputs_stay_in_range() {
    // Worst cases for growth: all at the limit, alternating, and a tone at
    // full scale. Debug builds panic on any i32 overflow in narrowing.
    for log2n in [1u8, 2, 5, 10, 16] {
        let n = 1usize << log2n;
        for x in [
            vec![SAMPLE_LIMIT; n],
            vec![-SAMPLE_LIMIT; n],
            (0..n)
                .map(|i| {
                    if i % 2 == 0 {
                        SAMPLE_LIMIT
                    } else {
                        -SAMPLE_LIMIT
                    }
                })
                .collect::<Vec<_>>(),
            (0..n)
                .map(|i| {
                    if (i / 3) % 2 == 0 {
                        SAMPLE_LIMIT
                    } else {
                        -SAMPLE_LIMIT
                    }
                })
                .collect(),
        ] {
            let _ = run(log2n, &x);
        }
    }
}

#[test]
fn tone_lands_in_its_bin() {
    let log2n = 10u8;
    let n = 1usize << log2n;
    let a = 1 << 29;
    let k = 37;
    let x: Vec<i32> = (0..n)
        .map(|t| {
            (f64::from(a) * (std::f64::consts::TAU * (k * t) as f64 / n as f64).cos()).round()
                as i32
        })
        .collect();
    let out = run(log2n, &x);
    // A cosine of amplitude A puts A/2 in bin k (DFT / N).
    let (re, im) = out[k];
    assert!((i64::from(re) - i64::from(a / 2)).abs() <= 8, "re={re}");
    assert!(im.abs() <= 8, "im={im}");
}

#[test]
fn bytes_wrapper_checks_lengths() {
    let mut out = Vec::new();
    assert_eq!(
        fft::fft(3, &[0; 31], &mut out),
        Err(bytelanes::abi::Status::InvalidLength)
    );
    assert_eq!(
        fft::fft(0, &[0; 4], &mut out),
        Err(bytelanes::abi::Status::InvalidLength)
    );
    assert_eq!(
        fft::fft(17, &[0; 4], &mut out),
        Err(bytelanes::abi::Status::InvalidLength)
    );
    // Two frames of 8 samples -> two frames of 5 bins.
    fft::fft(3, &[0; 64], &mut out).unwrap();
    assert_eq!(out.len(), 2 * 5 * 8);
    let mut empty = Vec::new();
    fft::fft(3, &[], &mut empty).unwrap();
    assert!(empty.is_empty());
}

fn frame_strategy() -> impl Strategy<Value = (u8, Vec<i32>)> {
    (1u8..=10).prop_flat_map(|log2n| {
        let n = 1usize << log2n;
        let sample = prop_oneof![
            4 => -SAMPLE_LIMIT..=SAMPLE_LIMIT,
            1 => any::<i32>(),
            1 => Just(SAMPLE_LIMIT),
            1 => Just(-SAMPLE_LIMIT),
            2 => -1000i32..1000,
        ];
        (Just(log2n), prop::collection::vec(sample, n))
    })
}

/// Worst observed error against the f64 DFT is checked against this bound
/// (in output LSB). Rounding adds at most about 1 LSB per stage before the
/// halving shrinks it, plus twiddle quantisation.
const ERROR_BOUND: f64 = 4.0;

proptest! {
    #![proptest_config(cfg())]

    #[test]
    fn iterative_equals_recursive((log2n, x) in frame_strategy()) {
        prop_assert_eq!(run(log2n, &x), recursive_fft(log2n, &x));
    }

    #[test]
    fn close_to_f64_dft((log2n, x) in (1u8..=8).prop_flat_map(|l| (Just(l), prop::collection::vec(-SAMPLE_LIMIT..=SAMPLE_LIMIT, 1usize << l)))) {
        let got = run(log2n, &x);
        let want = dft(&x);
        for (k, (&(r, i), &(wr, wi))) in got.iter().zip(&want).enumerate() {
            let err = (f64::from(r) - wr).abs().max((f64::from(i) - wi).abs());
            prop_assert!(err <= ERROR_BOUND, "log2n={} bin {}: got ({}, {}), want ({:.2}, {:.2})", log2n, k, r, i, wr, wi);
        }
    }
}
