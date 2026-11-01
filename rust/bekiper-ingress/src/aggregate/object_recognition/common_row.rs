fn common_row(trial: &TrialResult, detail: &NortResult) -> IndexMap<String, CellValue> {
    let mut row = IndexMap::new();
    row.insert("animal".into(), pluck_trial_metadata(trial, "animal"));
    row.insert("cohort".into(), pluck_trial_metadata(trial, "cohort"));
    row.insert("gene".into(), pluck_trial_metadata(trial, "gene"));
    row.insert("sex".into(), pluck_trial_metadata(trial, "sex"));
    row.insert("age".into(), pluck_trial_metadata(trial, "age"));
    row.insert("fps".into(), CellValue::Float(trial.fps));
    if let Some(policy) = &detail.scoring_provenance {
        row.insert(
            "scoring_provenance".into(),
            CellValue::String(policy.to_string()),
        );
    }
    row.insert(
        "protocol".into(),
        CellValue::String(detail.protocol.to_string()),
    );
    row.insert(
        "total_duration_s".into(),
        CellValue::Float(detail.total_duration_seconds),
    );
    row.insert(
        "duration_within_expected".into(),
        CellValue::Bool(bekiper_feature::zone::duration_within_expected(
            detail.total_duration_seconds,
        )),
    );
    row.insert(
        "total_distance_m".into(),
        CellValue::Float(detail.total_distance_meters),
    );
    row.insert(
        "total_mean_speed".into(),
        CellValue::Float(detail.total_mean_speed),
    );
    row.insert(
        "freezing_s".into(),
        CellValue::Float(detail.freezing_seconds),
    );
    row.insert(
        "latency_to_center".into(),
        detail
            .latency_to_center
            .map(CellValue::Float)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "median_speed_observing".into(),
        detail
            .median_speed_observing
            .map(CellValue::Float)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "median_speed_not_observing".into(),
        detail
            .median_speed_not_observing
            .map(CellValue::Float)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "first_approach_frame".into(),
        detail
            .first_approach_frame
            .map(|value| CellValue::Int(value as i64))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "first_object_approach_time_s".into(),
        detail
            .first_object_approach_time_s
            .or_else(|| {
                detail
                    .first_approach_frame
                    .map(|frame| frame as f64 / trial.fps)
            })
            .map(CellValue::Float)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "analysis_window_s".into(),
        CellValue::Float(detail.analysis_window_seconds),
    );
    row.insert(
        "window_mode".into(),
        if detail.window_mode == NortWindowMode::Unknown {
            CellValue::Null
        } else {
            CellValue::String(detail.window_mode.as_str().to_string())
        },
    );
    row.insert(
        "window_start_frame".into(),
        detail
            .window_start_frame
            .map(|value| CellValue::Int(value as i64))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "window_end_frame_exclusive".into(),
        detail
            .window_end_frame_exclusive
            .map(|value| CellValue::Int(value as i64))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "side_bias".into(),
        detail
            .side_bias
            .map(CellValue::Float)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_bias".into(),
        detail
            .object_bias
            .map(CellValue::Float)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_A_label".into(),
        detail
            .object_labels
            .first()
            .map(|value| CellValue::String(value.clone()))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_B_label".into(),
        detail
            .object_labels
            .get(1)
            .map(|value| CellValue::String(value.clone()))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_A_placement".into(),
        trial
            .session_context
            .get("object_a")
            .map(CellValue::from_json)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_B_placement".into(),
        trial
            .session_context
            .get("object_b")
            .map(CellValue::from_json)
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_A_exploration_s".into(),
        detail
            .exploration_times
            .first()
            .map(|value| CellValue::Float(*value))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_B_exploration_s".into(),
        detail
            .exploration_times
            .get(1)
            .map(|value| CellValue::Float(*value))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_A_bouts".into(),
        detail
            .bout_counts
            .first()
            .map(|value| CellValue::Int(*value as i64))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "object_B_bouts".into(),
        detail
            .bout_counts
            .get(1)
            .map(|value| CellValue::Int(*value as i64))
            .unwrap_or(CellValue::Null),
    );
    row.insert(
        "experiment".into(),
        CellValue::String(trial.experiment.clone()),
    );
    let excluded = if matches!(
        detail.protocol,
        ObjectRecognitionProtocol::NortTraining | ObjectRecognitionProtocol::NortNovelty
    ) {
        false
    } else {
        detail.excluded
    };
    row.insert("excluded".into(), CellValue::Bool(excluded));
    row.insert(
        "passes_15s_exploration_threshold".into(),
        CellValue::Bool(detail.passes_15s_exploration_threshold),
    );
    row
}

fn insert_observation_totals(
    row: &mut IndexMap<String, CellValue>,
    totals: Option<&ObservationTotals>,
) {
    let Some(totals) = totals else {
        return;
    };
    row.insert(
        "total_observation_s".into(),
        CellValue::Float(totals.total_seconds),
    );
    // `passes_15s_exploration_threshold` stays the trial's own flag, which
    // uses the configured `min_total_exploration` rather than a literal.
    row.insert(
        "total_exploration_s".into(),
        CellValue::Float(totals.total_seconds),
    );
    row.insert(
        "total_observation_bouts".into(),
        CellValue::Int(totals.total_bouts as i64),
    );
    for (label, seconds) in totals
        .object_labels
        .iter()
        .zip(totals.observation_seconds.iter())
    {
        row.insert(format!("{label}_observation_s"), CellValue::Float(*seconds));
        row.insert(format!("{label}_exploration_s"), CellValue::Float(*seconds));
    }
    for (label, bouts) in totals
        .object_labels
        .iter()
        .zip(totals.observation_bouts.iter())
    {
        row.insert(
            format!("{label}_observation_bouts"),
            CellValue::Int(*bouts as i64),
        );
        row.insert(format!("{label}_bouts"), CellValue::Int(*bouts as i64));
    }
}

fn derive_observation_totals(detail: &NortResult) -> Option<ObservationTotals> {
    let mut object_labels = Vec::new();
    let mut observation_seconds = Vec::new();
    let mut observation_bouts = Vec::new();

    for ((label, seconds), bouts) in detail
        .object_labels
        .iter()
        .zip(detail.exploration_times.iter())
        .zip(detail.bout_counts.iter())
    {
        if matches!(label.as_str(), "arena" | "center_zone") {
            continue;
        }
        object_labels.push(label.clone());
        observation_seconds.push(*seconds);
        observation_bouts.push(*bouts);
    }

    if object_labels.is_empty() {
        return None;
    }

    Some(ObservationTotals {
        total_seconds: observation_seconds.iter().sum(),
        total_bouts: observation_bouts.iter().sum(),
        object_labels,
        observation_seconds,
        observation_bouts,
    })
}

fn derive_pairwise_metrics(totals: &ObservationTotals) -> Vec<PairwiseObservationMetric> {
    let mut pairs = Vec::new();
    for lhs in 0..totals.object_labels.len() {
        for rhs in (lhs + 1)..totals.object_labels.len() {
            pairs.push(PairwiseObservationMetric {
                lhs_label: totals.object_labels[lhs].clone(),
                rhs_label: totals.object_labels[rhs].clone(),
                absolute_delta_seconds: (totals.observation_seconds[lhs]
                    - totals.observation_seconds[rhs])
                    .abs(),
            });
        }
    }
    pairs
}

fn insert_pairwise_metrics(
    row: &mut IndexMap<String, CellValue>,
    pairs: &[PairwiseObservationMetric],
) {
    for pair in pairs {
        row.insert(
            format!("pair_{}_{}_delta_s", pair.lhs_label, pair.rhs_label),
            CellValue::Float(pair.absolute_delta_seconds),
        );
        row.insert(
            format!("pair_{}_{}_delta_s", pair.rhs_label, pair.lhs_label),
            CellValue::Float(pair.absolute_delta_seconds),
        );
    }
}

fn insert_qualia_rollups(
    row: &mut IndexMap<String, CellValue>,
    rollups: &[QualiaDiagnosticRollup],
) {
    for rollup in rollups {
        let label = &rollup.label;
        row.insert(
            format!("{label}_body_center_ear_s"),
            CellValue::Float(rollup.body_center_ear_seconds),
        );
        row.insert(
            format!("{label}_body_tail_base_s"),
            CellValue::Float(rollup.body_tail_base_seconds),
        );
        row.insert(
            format!("{label}_body_total_s"),
            CellValue::Float(rollup.body_total_seconds),
        );
        row.insert(
            format!("{label}_olfaction_nose_s"),
            CellValue::Float(rollup.olfaction_nose_seconds),
        );
        row.insert(
            format!("{label}_olfaction_ray_s"),
            CellValue::Float(rollup.olfaction_ray_seconds),
        );
        row.insert(
            format!("{label}_olfaction_total_s"),
            CellValue::Float(rollup.olfaction_total_seconds),
        );
        row.insert(
            format!("{label}_whisker_left_proximity_s"),
            CellValue::Float(rollup.whisker_left_proximity_seconds),
        );
        row.insert(
            format!("{label}_whisker_left_fov_s"),
            CellValue::Float(rollup.whisker_left_fov_seconds),
        );
        row.insert(
            format!("{label}_whisker_right_proximity_s"),
            CellValue::Float(rollup.whisker_right_proximity_seconds),
        );
        row.insert(
            format!("{label}_whisker_right_fov_s"),
            CellValue::Float(rollup.whisker_right_fov_seconds),
        );
        row.insert(
            format!("{label}_whisker_total_s"),
            CellValue::Float(rollup.whisker_total_seconds),
        );
    }
}

fn insert_zone_metrics(
    row: &mut IndexMap<String, CellValue>,
    zone_metrics: Option<&ObjectRecognitionZoneMetrics>,
) {
    let Some(zone_metrics) = zone_metrics else {
        return;
    };

    insert_zone_motion_summary(row, "all", zone_metrics.all_motion_summary.as_ref());
    insert_zone_summary(row, "center", zone_metrics.center_summary.as_ref());
    insert_zone_motion_summary(row, "center", zone_metrics.center_motion_summary.as_ref());
    insert_zone_summary(row, "periphery", zone_metrics.periphery_summary.as_ref());
    insert_zone_motion_summary(
        row,
        "periphery",
        zone_metrics.periphery_motion_summary.as_ref(),
    );

    for quadrant in &zone_metrics.quadrant_summaries {
        let name = quadrant_name(&quadrant.zone_name);
        insert_zone_summary(row, &name, Some(quadrant));
    }
    for quadrant in &zone_metrics.quadrant_motion_summaries {
        let name = quadrant_name(&quadrant.zone_name);
        insert_zone_motion_summary(row, &name, Some(quadrant));
    }
}

fn insert_zone_summary(
    row: &mut IndexMap<String, CellValue>,
    prefix: &str,
    summary: Option<&bekiper_feature::zone::ZoneSummary>,
) {
    let Some(summary) = summary else {
        return;
    };
    row.insert(
        format!("{prefix}_duration_s"),
        CellValue::Float(summary.duration_seconds),
    );
    row.insert(
        format!("{prefix}_entries"),
        CellValue::Int(summary.entries as i64),
    );
    row.insert(
        format!("{prefix}_distance_m"),
        CellValue::Float(summary.distance_meters),
    );
    row.insert(
        format!("{prefix}_mean_speed"),
        CellValue::Float(summary.mean_speed),
    );
}

fn insert_zone_motion_summary(
    row: &mut IndexMap<String, CellValue>,
    prefix: &str,
    summary: Option<&bekiper_behaviour::motion_detail::ZoneMotionSummary>,
) {
    let Some(summary) = summary else {
        return;
    };
    row.insert(
        format!("{prefix}_motion_distance_m"),
        CellValue::Float(summary.total_displacement_meters),
    );
    row.insert(
        format!("{prefix}_motion_median_speed"),
        CellValue::Float(summary.median_speed_mps),
    );
    row.insert(
        format!("{prefix}_motion_median_acceleration"),
        CellValue::Float(summary.median_acceleration_mps2),
    );
    row.insert(
        format!("{prefix}_motion_freezing_s"),
        CellValue::Float(summary.freezing_seconds),
    );
}

fn ensure_static_columns(row: &mut IndexMap<String, CellValue>, columns: &[String]) {
    for column in columns {
        row.entry(column.clone()).or_insert(CellValue::Null);
    }
}

fn quadrant_name(zone_name: &str) -> String {
    let Some(raw) = zone_name.strip_prefix("q_") else {
        return zone_name.to_string();
    };
    let mut parts = raw.split('_');
    let row = parts
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value + 1)
        .unwrap_or(1);
    let col = parts
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value + 1)
        .unwrap_or(1);
    format!("quadrant_{row}_{col}")
}
