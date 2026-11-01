use bekiper_math::temporal::{DurationComparison, select_bouts};

#[test]
fn duration_boundaries_and_original_crossing_are_explicit() {
    let raw = [false, true, true, false, true, true, true];
    let measure = |start, end| (end - start) as f64 / 2.0;
    let inclusive = select_bouts(&raw, measure, 1.0, DurationComparison::AtLeast, 0.0, 0.0);
    assert_eq!(inclusive.mask, raw);
    assert_eq!(inclusive.bouts[0].start, 1);
    assert_eq!(inclusive.bouts[0].qualification_index, Some(2));
    let strict = select_bouts(
        &raw,
        measure,
        1.0,
        DurationComparison::GreaterThan,
        0.0,
        0.0,
    );
    assert_eq!(strict.mask, [false, false, false, false, true, true, true]);
    assert_eq!(strict.bouts[0].qualification_index, None);
    assert_eq!(strict.bouts[1].qualification_index, Some(6));
}

#[test]
fn merge_precedes_duration_and_leading_trailing_gaps_are_preserved() {
    let raw = [false, true, false, true, false];
    let selected = select_bouts(
        &raw,
        |a, b| (b - a) as f64,
        3.0,
        DurationComparison::AtLeast,
        1.0,
        0.0,
    );
    assert_eq!(selected.mask, [false, true, true, true, false]);
    assert_eq!(selected.raw_spans, [1..2, 3..4]);
    assert_eq!(selected.bouts.len(), 1);
    let density = select_bouts(
        &raw,
        |a, b| (b - a) as f64,
        3.0,
        DurationComparison::AtLeast,
        1.0,
        0.1,
    );
    assert_eq!(density.mask, [false; 5]);
}
