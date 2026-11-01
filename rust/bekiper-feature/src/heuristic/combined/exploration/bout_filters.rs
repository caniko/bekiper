fn filter_selected_bouts(
    raw: &BooleanChunked,
    selected: bekiper_math::temporal::TemporalSelection,
    filters: BoutFilterSpec<'_>,
) -> (BooleanChunked, BoutFilterDiagnostics) {
    let mut diagnostics = BoutFilterDiagnostics {
        candidate_bouts_pre_merge: count_bouts(raw),
        ..Default::default()
    };
    let post_merge = BooleanChunked::from_slice(raw.name().clone(), &selected.post_merge_mask);
    diagnostics.bouts_post_merge = count_bouts(&post_merge);

    let mut toleranced = BooleanChunked::from_slice(raw.name().clone(), &selected.mask);
    let bouts_post_duration = count_bouts(&toleranced);
    diagnostics.removed_by_duration = diagnostics
        .bouts_post_merge
        .saturating_sub(bouts_post_duration);

    if filters.speed_mode != BoutFilterMode::PerBoutMedian
        && filters.angle_mode != BoutFilterMode::PerBoutMedian
    {
        diagnostics.final_valid_bouts = bouts_post_duration;
        collect_final_bout_medians(&mut diagnostics, &toleranced, &filters);
        return (toleranced, diagnostics);
    }

    let mut values: Vec<bool> = toleranced
        .into_iter()
        .map(|value| value.unwrap_or(false))
        .collect();
    for span in bekiper_math::temporal::state_spans(&values, true) {
        let (start, end) = (span.start, span.end);

        let speed_median = median_for_range(filters.nose_speed_mps, start, end);
        let angle_median = median_for_range(filters.angle_deg, start, end);
        let speed_ok = filters.speed_mode != BoutFilterMode::PerBoutMedian
            || speed_median.is_some_and(|median| {
                if filters.strict_boundaries {
                    median < filters.max_nose_speed
                } else {
                    median <= filters.max_nose_speed
                }
            });
        let angle_median_ok = filters.angle_mode != BoutFilterMode::PerBoutMedian
            || angle_median.is_some_and(|median| median <= filters.max_angle_deg);
        let angle_ok = filters.angle_mode != BoutFilterMode::PerBoutMedian
            || accepts_bout_angle(
                angle_median,
                filters.max_angle_deg,
                any_true_in_range(filters.nose_contact, start, end),
            );
        let contact_rescue = !angle_median_ok && angle_ok;

        if !(speed_ok && angle_ok) {
            if !speed_ok {
                diagnostics.removed_by_speed += 1;
            } else {
                diagnostics.removed_by_angle += 1;
            }
            for value in values.iter_mut().take(end).skip(start) {
                *value = false;
            }
        } else if contact_rescue {
            diagnostics.bouts_rescued_by_contact_override += 1;
        }
    }

    toleranced = BooleanChunked::from_slice(toleranced.name().clone(), &values);
    diagnostics.final_valid_bouts = count_bouts(&toleranced);
    collect_final_bout_medians(&mut diagnostics, &toleranced, &filters);
    (toleranced, diagnostics)
}
