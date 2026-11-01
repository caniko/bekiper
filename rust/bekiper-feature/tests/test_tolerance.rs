use polars::prelude::*;

use bekiper_core::video::FrameTiming;
use bekiper_feature::tolerance::{
    BoutTiming, DurationComparison, DwellOverrides, TemporalBoutOps, TemporalPolicy,
};

#[test]
fn select_temporal_bouts_validates_column_type_and_length() {
    let df = df! { "active" => &[true, true, false], "x" => &[1.0, 2.0, 3.0] }.unwrap();
    let policy = TemporalPolicy::default();
    let timing = BoutTiming::cfr(1.0, 3).unwrap();
    let selection = df
        .select_temporal_bouts("active", &policy, &timing)
        .unwrap();
    assert_eq!(selection.mask, vec![true, true, false]);
    assert!(
        df.select_temporal_bouts("missing", &policy, &timing)
            .is_err()
    );
    assert!(df.select_temporal_bouts("x", &policy, &timing).is_err());
    let short = BoutTiming::cfr(1.0, 2).unwrap();
    let err = df
        .select_temporal_bouts("active", &policy, &short)
        .unwrap_err();
    assert!(err.to_string().contains("lengths differ"));
}

#[test]
fn filter_series_treats_null_as_absent() {
    let raw =
        BooleanChunked::from_iter_options("m".into(), [Some(true), None, Some(true)].into_iter());
    let policy = TemporalPolicy {
        max_gap_seconds: 1.0,
        ..Default::default()
    };
    let selection = policy
        .filter_series(&raw, &BoutTiming::cfr(1.0, 3).unwrap())
        .unwrap();
    assert_eq!(selection.raw_spans, vec![0..1, 2..3]);
    // A one-second gap is bridged at exactly the inclusive maximum.
    assert_eq!(selection.mask, vec![true, true, true]);
}

#[test]
fn duration_comparison_at_exact_minimum() {
    let timing = BoutTiming::cfr(4.0, 4).unwrap();
    let at_least = TemporalPolicy {
        min_duration_seconds: 1.0,
        ..Default::default()
    };
    let greater = TemporalPolicy {
        duration_comparison: DurationComparison::GreaterThan,
        ..at_least
    };
    assert!(at_least.filter(&[true; 4], &timing).unwrap().bouts[0].accepted);
    let rejected = greater.filter(&[true; 4], &timing).unwrap();
    assert!(!rejected.bouts[0].accepted);
    assert_eq!(rejected.mask, vec![false; 4]);
    assert_eq!(rejected.post_merge_mask, vec![true; 4]);
}

#[test]
fn policy_validation_rejects_invalid_limits() {
    let ok = TemporalPolicy::ymaze_default();
    assert_eq!(ok.min_duration_seconds, 1.0);
    assert!(ok.validate().is_ok());
    for policy in [
        TemporalPolicy {
            max_gap_seconds: -1.0,
            ..ok
        },
        TemporalPolicy {
            max_gap_seconds: f64::INFINITY,
            ..ok
        },
        TemporalPolicy {
            gap_density_threshold: 1.5,
            ..ok
        },
        TemporalPolicy {
            gap_density_threshold: f64::NAN,
            ..ok
        },
    ] {
        assert!(policy.validate().is_err(), "{policy:?}");
        assert!(
            policy
                .filter(&[], &BoutTiming::cfr(1.0, 0).unwrap())
                .is_err()
        );
    }
    assert!(
        TemporalPolicy {
            gap_density_threshold: 1.0,
            ..ok
        }
        .validate()
        .is_ok()
    );
    assert!(ok.validate_uninterrupted().is_ok());
    assert!(
        TemporalPolicy {
            max_gap_seconds: 0.5,
            ..ok
        }
        .validate_uninterrupted()
        .unwrap_err()
        .contains("uninterrupted")
    );
    assert!(
        TemporalPolicy {
            min_duration_seconds: -0.1,
            ..ok
        }
        .validate_uninterrupted()
        .is_err()
    );
}

#[test]
fn dwell_overrides_resolve_explicit_before_env() {
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        move |key: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    };
    let none = DwellOverrides::from_env(env(&[])).unwrap();
    assert!(none.is_empty());
    let from_env = DwellOverrides::from_env(env(&[
        ("BEKIPER_YMAZE_DWELL_SECONDS", "0.5"),
        ("BEKIPER_YMAZE_DWELL_COMPARISON", "greater_than"),
    ]))
    .unwrap();
    assert_eq!(from_env.seconds, Some(0.5));
    assert_eq!(from_env.comparison, Some(DurationComparison::GreaterThan));
    assert!(!from_env.is_empty());
    assert!(DwellOverrides::from_env(env(&[("BEKIPER_YMAZE_DWELL_SECONDS", "x")])).is_err());
    assert!(DwellOverrides::from_env(env(&[("BEKIPER_YMAZE_DWELL_SECONDS", "-1")])).is_err());
    assert!(
        DwellOverrides::from_env(env(&[("BEKIPER_YMAZE_DWELL_COMPARISON", "longer")])).is_err()
    );
    // Explicit values win, so invalid lower-priority env values are ignored.
    let explicit = DwellOverrides {
        seconds: Some(2.0),
        comparison: Some(DurationComparison::AtLeast),
    }
    .with_env(env(&[
        ("BEKIPER_YMAZE_DWELL_SECONDS", "x"),
        ("BEKIPER_YMAZE_DWELL_COMPARISON", "longer"),
    ]))
    .unwrap();
    assert_eq!(explicit.seconds, Some(2.0));
    let policy = explicit.apply(TemporalPolicy::ymaze_default());
    assert_eq!(policy.min_duration_seconds, 2.0);
    let merged = from_env.merged(DwellOverrides {
        seconds: Some(3.0),
        comparison: None,
    });
    assert_eq!(merged.seconds, Some(3.0));
    assert_eq!(merged.comparison, Some(DurationComparison::GreaterThan));
}

#[test]
fn bout_timing_constructors_reject_invalid_clocks() {
    assert!(BoutTiming::cfr(0.0, 1).is_err());
    assert!(BoutTiming::cfr(f64::NAN, 1).is_err());
    assert!(BoutTiming::timestamps(&[0.0, f64::NAN], 1.0).is_err());
    let empty = BoutTiming::cfr(30.0, 0).unwrap();
    assert!(empty.is_empty());
    assert!(!BoutTiming::cfr(30.0, 1).unwrap().is_empty());

    let cfr = BoutTiming::frames(&FrameTiming::cfr(4.0), 8).unwrap();
    assert_eq!(cfr.len(), 8);
    assert_eq!(cfr.duration(0, 8), 2.0);
    let vfr = FrameTiming {
        fps: 2.0,
        timestamps: vec![0.0, 1.0, 1.25],
        is_vfr: true,
    };
    let timed = BoutTiming::frames(&vfr, 3).unwrap();
    // The last frame gets nominal 1/FPS support.
    assert_eq!(timed.boundary(3), 1.75);
    assert!(BoutTiming::frames(&vfr, 2).is_err());
    assert!(
        BoutTiming::frames(
            &FrameTiming {
                fps: 0.0,
                ..vfr.clone()
            },
            3
        )
        .is_err()
    );
}

#[test]
fn bout_timing_slicing_and_windows() {
    let timed = BoutTiming::timestamps(&[0.0, 1.0, 3.0, 4.0], 6.0).unwrap();
    let trimmed = timed.trimmed(1);
    assert_eq!(trimmed.len(), 3);
    assert_eq!(trimmed.boundary(0), 1.0);
    let sliced = timed.sliced(1, 3);
    assert_eq!(sliced.len(), 2);
    assert_eq!(sliced.duration(0, 2), 3.0);
    let cfr = BoutTiming::cfr(2.0, 10).unwrap().sliced(2, 6);
    assert_eq!(cfr.len(), 4);
    assert_eq!(cfr.boundary(0), 1.0);
    assert_eq!(cfr.mask_seconds(&[true, false, true, true]), 1.5);

    // Window end falls inside sample 2 ([3, 4)): its support is clipped at 3.5 s.
    let window = timed.window(1, 3.5);
    assert_eq!(window.len(), 2);
    assert_eq!(window.boundary(0), 1.0);
    assert_eq!(window.boundary(2), 3.5);
    assert_eq!(window.mask_seconds(&[false, true]), 0.5);
    // A window ending exactly on a sample start excludes that sample.
    let exact = timed.window(0, 3.0);
    assert_eq!(exact.len(), 2);
    assert_eq!(exact.boundary(2), 3.0);
    // A window past the recording keeps the final boundary.
    let whole = timed.window(0, 100.0);
    assert_eq!(whole.len(), 4);
    assert_eq!(whole.boundary(4), 6.0);
    let cfr_window = BoutTiming::cfr(2.0, 4).unwrap().window(0, 1.25);
    assert_eq!(cfr_window.len(), 3);
    assert_eq!(cfr_window.mask_seconds(&[true, true, true]), 1.25);
}
