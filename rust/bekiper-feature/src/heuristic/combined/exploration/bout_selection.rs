#[cfg(test)]
fn select_exploration_bouts(
    raw: &BooleanChunked,
    timing: &BoutTiming,
    minimum: f64,
    gap: f64,
    density: f64,
) -> bekiper_math::temporal::TemporalSelection {
    select_exploration_bouts_bounded(raw, timing, minimum, (gap, false), density)
}

fn select_exploration_bouts_bounded(
    raw: &BooleanChunked,
    timing: &BoutTiming,
    minimum: f64,
    gap: (f64, bool),
    density: f64,
) -> bekiper_math::temporal::TemporalSelection {
    let values: Vec<bool> = raw.into_iter().map(|v| v.unwrap_or(false)).collect();
    bekiper_math::temporal::select_bouts_with_gap_bound(
        &values,
        |start, end| timing.duration(start, end),
        minimum,
        bekiper_math::temporal::DurationComparison::AtLeast,
        gap.0,
        density,
        gap.1,
    )
}

/// Another object's resolved candidate interval is an assignment, not a gap.
/// Split at these barriers before applying the shared merge/duration kernel.
fn select_exploration_bouts_for_column(
    df: &DataFrame,
    raw_col: &str,
    raw: &BooleanChunked,
    timing: &BoutTiming,
    minimum: f64,
    gap: (f64, bool),
    density: f64,
) -> bekiper_math::temporal::TemporalSelection {
    let mut barriers = vec![false; raw.len()];
    if raw_col.starts_with("exploring_") && raw_col.ends_with("_raw") {
        for column in df.columns() {
            let name = column.name().as_str();
            if name != raw_col
                && name.starts_with("exploring_")
                && name.ends_with("_raw")
                && let Ok(other) = column.bool()
            {
                for (barrier, value) in barriers.iter_mut().zip(other) {
                    *barrier |= value.unwrap_or(false);
                }
            }
        }
    }
    if !barriers.iter().any(|&value| value) {
        return select_exploration_bouts_bounded(raw, timing, minimum, gap, density);
    }
    let mut result = bekiper_math::temporal::TemporalSelection {
        mask: vec![false; raw.len()],
        post_merge_mask: vec![false; raw.len()],
        raw_spans: Vec::new(),
        bouts: Vec::new(),
    };
    for span in bekiper_math::temporal::state_spans(&barriers, false) {
        let selected = select_exploration_bouts_bounded(
            &raw.slice(span.start as i64, span.len()),
            &timing.sliced(span.start, span.end),
            minimum,
            gap,
            density,
        );
        result.mask[span.clone()].copy_from_slice(&selected.mask);
        result.post_merge_mask[span.clone()].copy_from_slice(&selected.post_merge_mask);
        result.raw_spans.extend(
            selected
                .raw_spans
                .into_iter()
                .map(|raw| raw.start + span.start..raw.end + span.start),
        );
        result
            .bouts
            .extend(selected.bouts.into_iter().map(|mut bout| {
                bout.start += span.start;
                bout.end += span.start;
                bout.qualification_index = bout.qualification_index.map(|index| index + span.start);
                bout
            }));
    }
    result
}
