fn bout_filter_diagnostics(
    raw: &BooleanChunked,
    final_toleranced: Option<&BooleanChunked>,
    selected: bekiper_math::temporal::TemporalSelection,
    filters: BoutFilterSpec<'_>,
) -> BoutFilterDiagnostics {
    let Some(final_toleranced) = final_toleranced else {
        return BoutFilterDiagnostics::default();
    };

    let (_, mut diagnostics) = filter_selected_bouts(raw, selected, filters);
    diagnostics.final_valid_bouts = count_bouts(final_toleranced);
    diagnostics.per_bout_median_nose_speed.clear();
    diagnostics.per_bout_median_head_angle_deg.clear();
    collect_final_bout_medians(&mut diagnostics, final_toleranced, &filters);
    diagnostics
}

fn any_true_in_range(mask: Option<&BooleanChunked>, start: usize, end: usize) -> bool {
    let Some(mask) = mask else {
        return false;
    };
    (start..end).any(|idx| mask.get(idx).unwrap_or(false))
}

fn collect_final_bout_medians(
    diagnostics: &mut BoutFilterDiagnostics,
    final_toleranced: &BooleanChunked,
    filters: &BoutFilterSpec<'_>,
) {
    for (start, end) in true_ranges(final_toleranced) {
        // NaN when the underlying diagnostic column is unavailable
        // (e.g. PerFrame filter mode without those columns); per-bout
        // medians stay aligned with `final_valid_bouts` count.
        diagnostics
            .per_bout_median_nose_speed
            .push(median_for_range(filters.nose_speed_mps, start, end).unwrap_or(f64::NAN));
        diagnostics
            .per_bout_median_head_angle_deg
            .push(median_for_range(filters.angle_deg, start, end).unwrap_or(f64::NAN));
    }
}

fn true_ranges(mask: &BooleanChunked) -> Vec<(usize, usize)> {
    bekiper_math::discrete::true_series_spans(mask)
        .iter()
        .map(|span| (span.start, span.end))
        .collect()
}

fn median_for_range(values: Option<&Float64Chunked>, start: usize, end: usize) -> Option<f64> {
    let values = values?;
    let mut finite_values: Vec<f64> = (start..end)
        .filter_map(|idx| values.get(idx))
        .filter(|value| value.is_finite())
        .collect();
    if finite_values.is_empty() {
        return None;
    }

    finite_values.sort_by(f64::total_cmp);
    let mid = finite_values.len() / 2;
    if finite_values.len().is_multiple_of(2) {
        Some((finite_values[mid - 1] + finite_values[mid]) / 2.0)
    } else {
        Some(finite_values[mid])
    }
}
