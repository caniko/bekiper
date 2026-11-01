//! Property tests: every sweep kernel against `select_bouts_with_gap_bound`.
//! Seeded xorshift loops (no proptest in the workspace lockfile).
use bekiper_math::sweep::*;
use bekiper_math::temporal::{DurationComparison, select_bouts_with_gap_bound};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// CFR or timestamped clock closure data, mirroring `BoutTiming::duration`.
#[derive(Clone)]
enum Clock {
    Cfr(f64),
    Vfr(Vec<f64>),
}
impl Clock {
    fn random(rng: &mut Rng, n: usize) -> Self {
        if rng.below(2) == 0 {
            Clock::Cfr([25.0, 30.0, 29.97, 4.0][rng.below(4)])
        } else {
            let mut t = 5.0;
            let mut b = vec![t];
            for _ in 0..n {
                t += 0.02 + 0.03 * rng.unit();
                b.push(t);
            }
            Clock::Vfr(b)
        }
    }
    fn duration(&self, a: usize, b: usize) -> f64 {
        match self {
            Clock::Cfr(fps) => (b - a) as f64 / fps,
            Clock::Vfr(bd) => bd[b] - bd[a],
        }
    }
}

fn random_rule(rng: &mut Rng) -> BoutRule {
    let pick = |rng: &mut Rng, xs: &[f64]| xs[rng.below(xs.len())];
    BoutRule {
        minimum: pick(rng, &[0.0, 0.1, 0.25, 0.5, 1.0, 0.08, 0.2]),
        comparison: if rng.below(2) == 0 {
            DurationComparison::AtLeast
        } else {
            DurationComparison::GreaterThan
        },
        maximum_gap: pick(rng, &[0.0, 0.04, 0.08, 0.2, 0.5, 1.0]),
        gap_density: pick(rng, &[0.0, 0.0, 0.0, 0.3]),
        strict_gap: rng.below(2) == 0,
    }
}

fn legacy_mask(mask: &[bool], clock: &Clock, rule: BoutRule) -> Vec<bool> {
    select_bouts_with_gap_bound(
        mask,
        |a, b| clock.duration(a, b),
        rule.minimum,
        rule.comparison,
        rule.maximum_gap,
        rule.gap_density,
        rule.strict_gap,
    )
    .mask
}

fn random_mask(rng: &mut Rng, n: usize) -> Vec<bool> {
    let p = [0.1, 0.3, 0.5, 0.8][rng.below(4)];
    (0..n).map(|_| rng.unit() < p).collect()
}

#[test]
fn select_runs_and_qualification_match_legacy() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..3000 {
        let n = 1 + rng.below(60);
        let mask = random_mask(&mut rng, n);
        let clock = Clock::random(&mut rng, n);
        let rule = random_rule(&mut rng);
        let runs = bekiper_math::temporal::state_spans(&mask, true);
        let spans = select_runs(&runs, |a, b| clock.duration(a, b), rule);
        let legacy = select_bouts_with_gap_bound(
            &mask,
            |a, b| clock.duration(a, b),
            rule.minimum,
            rule.comparison,
            rule.maximum_gap,
            rule.gap_density,
            rule.strict_gap,
        );
        assert_eq!(spans_to_mask(&spans, n), legacy.mask);
        let accepted: Vec<_> = legacy.bouts.iter().filter(|b| b.accepted).collect();
        assert_eq!(accepted.len(), spans.len());
        for (bout, span) in accepted.iter().zip(&spans) {
            assert_eq!(
                qualification_index(
                    span,
                    |a, b| clock.duration(a, b),
                    rule.minimum,
                    rule.comparison
                ),
                bout.qualification_index
            );
        }
    }
}

fn check_family(rng: &mut Rng, noisy_legacy: bool) {
    let n = 1 + rng.below(80);
    let clock = Clock::random(rng, n);
    let rule = random_rule(rng);
    // Quantised critical values force ties and frames sitting exactly on thresholds.
    let critical: Vec<f64> = (0..n)
        .map(|_| {
            if rng.below(12) == 0 {
                f64::NAN
            } else {
                rng.below(10) as f64 * 0.01
            }
        })
        .collect();
    let thresholds: Vec<f64> = (0..1 + rng.below(6))
        .map(|_| rng.below(11) as f64 * 0.01 - 0.005 * rng.below(2) as f64)
        .collect();
    // Legacy predicate: exact comparison, or one that disagrees with `c <= t`
    // for frames inside the float band (including non-monotone answers).
    let flip: Vec<bool> = (0..n * thresholds.len())
        .map(|_| rng.below(3) == 0)
        .collect();
    let nk = thresholds.len();
    let legacy_pred = |i: usize, k: usize| {
        let exact = critical[i] <= thresholds[k];
        let near = (critical[i] - thresholds[k]).abs() <= 1e-12 * (1.0 + thresholds[k].abs());
        if noisy_legacy && near && critical[i].is_finite() {
            flip[i * nk + k]
        } else {
            exact
        }
    };
    let got = sweep_family(
        &critical,
        &thresholds,
        &legacy_pred,
        |a, b| clock.duration(a, b),
        rule,
        true,
    );
    let bits = threshold_bitsets(&critical, &thresholds, &legacy_pred);
    for k in 0..nk {
        let mask: Vec<bool> = (0..n).map(|i| legacy_pred(i, k)).collect();
        let want = legacy_mask(&mask, &clock, rule);
        assert_eq!(spans_to_mask(&got[k].spans, n), want, "k={k}");
        assert_eq!(got[k].accepted_samples, want.iter().filter(|&&v| v).count());
        assert_eq!(bits[k].to_bools(), mask);
        assert_eq!(
            bits[k].runs(),
            bekiper_math::temporal::state_spans(&mask, true)
        );
    }
}

#[test]
fn sweep_family_matches_legacy_exact_predicate() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    for _ in 0..4000 {
        check_family(&mut rng, false);
    }
}

#[test]
fn sweep_family_matches_legacy_inside_float_band_including_non_monotone() {
    let mut rng = Rng(0x2545F4914F6CDD1D);
    for _ in 0..4000 {
        check_family(&mut rng, true);
    }
}

#[test]
fn fixed_mask_special_cases_match_legacy() {
    let mut rng = Rng(0xA0761D6478BD642F);
    for _ in 0..1500 {
        let n = 1 + rng.below(120);
        let clock = Clock::random(&mut rng, n);
        let mask = random_mask(&mut rng, n);
        let runs = bekiper_math::temporal::state_spans(&mask, true);
        let variants: Vec<BoutRule> = (0..1 + rng.below(8))
            .map(|_| random_rule(&mut rng))
            .collect();
        let got = gap_duration_totals(&runs, |a, b| clock.duration(a, b), &variants);
        for (rule, totals) in variants.iter().zip(&got) {
            let want = legacy_mask(&mask, &clock, *rule);
            let count = bekiper_math::temporal::state_spans(&want, true).len();
            // Accepted spans can touch only through zero-width gaps, which
            // never exist between maximal runs of the legacy mask.
            assert_eq!(totals.samples, want.iter().filter(|&&v| v).count());
            assert_eq!(totals.count, count);
        }
        let dwell: Vec<_> = variants.iter().map(|r| (r.minimum, r.comparison)).collect();
        let got = min_duration_totals(&runs, |a, b| clock.duration(a, b), &dwell);
        for (rule, totals) in variants.iter().zip(&got) {
            let rule = BoutRule {
                maximum_gap: 0.0,
                ..*rule
            };
            let want = legacy_mask(&mask, &clock, rule);
            assert_eq!(totals.samples, want.iter().filter(|&&v| v).count());
            assert_eq!(
                totals.count,
                bekiper_math::temporal::state_spans(&want, true).len()
            );
        }
    }
}

#[test]
fn bitset_ops_and_runs() {
    let mut rng = Rng(77);
    for _ in 0..500 {
        let n = rng.below(200);
        let (a, b) = (random_mask(&mut rng, n), random_mask(&mut rng, n));
        let (x, y) = (BitSet::from_bools(&a), BitSet::from_bools(&b));
        let and: Vec<bool> = a.iter().zip(&b).map(|(p, q)| *p && *q).collect();
        let or: Vec<bool> = a.iter().zip(&b).map(|(p, q)| *p || *q).collect();
        let not: Vec<bool> = a.iter().map(|p| !p).collect();
        assert_eq!(x.and(&y).to_bools(), and);
        assert_eq!(x.or(&y).to_bools(), or);
        assert_eq!(x.not().to_bools(), not);
        assert_eq!(x.not().count_ones(), not.iter().filter(|&&v| v).count());
        assert_eq!(x.runs(), bekiper_math::temporal::state_spans(&a, true));
    }
}

#[test]
fn bitset_and_not_len_and_spanless_sweep() {
    let (a, b) = (
        BitSet::from_bools(&[true, true, false]),
        BitSet::from_bools(&[true, false, false]),
    );
    assert_eq!(a.and_not(&b).to_bools(), [false, true, false]);
    assert_eq!(
        (a.len(), a.is_empty(), BitSet::new(0).is_empty()),
        (3, false, true)
    );
    let rule = BoutRule {
        minimum: 0.0,
        comparison: DurationComparison::AtLeast,
        maximum_gap: 0.0,
        gap_density: 0.0,
        strict_gap: false,
    };
    let got = sweep_family(
        &[0.0, 1.0],
        &[0.5],
        |i, _| i == 0,
        |a, b| (b - a) as f64,
        rule,
        false,
    );
    assert_eq!((got[0].accepted_count, got[0].spans.len()), (1, 0));
}

/// `cargo test --release -p bekiper-math --test sweep -- --ignored --nocapture`
#[test]
#[ignore = "micro-benchmark"]
fn bench_sweep_vs_legacy_on_9000_frames() {
    use std::time::Instant;
    let mut rng = Rng(42);
    let n = 9000;
    let clock = Clock::Cfr(30.0);
    let critical: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
    let rule = BoutRule {
        minimum: 0.25,
        comparison: DurationComparison::AtLeast,
        maximum_gap: 1.0,
        gap_density: 0.0,
        strict_gap: false,
    };
    for k in [1usize, 4, 16] {
        let thresholds: Vec<f64> = (0..k).map(|i| 0.3 + 0.4 * i as f64 / k as f64).collect();
        let t = Instant::now();
        let masks: Vec<Vec<bool>> = thresholds
            .iter()
            .map(|&t| {
                let mask: Vec<bool> = critical.iter().map(|&c| c <= t).collect();
                legacy_mask(&mask, &clock, rule)
            })
            .collect();
        let legacy = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let snaps = sweep_family(
            &critical,
            &thresholds,
            |i, k| critical[i] <= thresholds[k],
            |a, b| clock.duration(a, b),
            rule,
            false,
        );
        let sweep = t.elapsed().as_secs_f64();
        for (snap, mask) in snaps.iter().zip(&masks) {
            assert_eq!(snap.accepted_samples, mask.iter().filter(|&&v| v).count());
        }
        println!("sweep K={k:<2} legacy {legacy:.6}s  sweep {sweep:.6}s");
    }
}
