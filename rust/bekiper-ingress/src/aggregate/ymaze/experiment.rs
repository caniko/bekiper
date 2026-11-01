impl Experiment for YMazeExperiment {
    fn id(&self) -> ExperimentId {
        ExperimentId("ymaze")
    }

    fn sheet_key(&self, trial: &TrialResult) -> Option<SheetKey> {
        // Variant tags come from the same role/recipe as the routing key.
        Some(legacy_routing_key_for_trial(trial).map_or(
            SheetKey {
                experiment: ExperimentId("y_maze"),
                variant: None,
            },
            |key| sheet_key_from_routing_key(&key),
        ))
    }

    fn static_columns(&self) -> Vec<String> {
        YMAZE_STATIC_COLUMNS.iter().map(|s| s.to_string()).collect()
    }

    fn static_columns_for_trials(&self, _trials: &[&TrialResult]) -> Vec<String> {
        YMAZE_STATIC_COLUMNS.iter().map(|s| s.to_string()).collect()
    }

    fn dynamic_columns(&self, _trial: &TrialResult) -> Vec<String> {
        Vec::new()
    }

    fn extract_row(&self, trial: &TrialResult) -> IndexMap<String, CellValue> {
        let mut row: IndexMap<String, CellValue> =
            IndexMap::with_capacity(YMAZE_STATIC_COLUMNS.len());

        row.insert("animal".into(), pluck_trial_metadata(trial, "animal"));
        row.insert("cohort".into(), pluck_trial_metadata(trial, "cohort"));
        row.insert("gene".into(), pluck_trial_metadata(trial, "gene"));
        row.insert("sex".into(), pluck_trial_metadata(trial, "sex"));
        row.insert("age".into(), pluck_trial_metadata(trial, "age"));
        row.insert("fps".into(), CellValue::Float(trial.fps));

        let d = &trial.detailed;
        row.insert("proximal_threshold".into(), pluck(d, "proximal_threshold"));
        row.insert("start_mode".into(), pluck(d, "start_mode"));
        row.insert(
            "zone_max_gap_seconds".into(),
            pluck(d, "zone_max_gap_seconds"),
        );
        row.insert(
            "arm_membership_hysteresis".into(),
            pluck(d, "arm_membership_hysteresis"),
        );
        row.insert(
            "pose_gap_fill_seconds".into(),
            pluck(d, "pose_gap_fill_seconds"),
        );
        // CellValue::from_json drops objects. Preserve structured evidence as
        // JSON text in the aggregate cell rather than silently exporting null.
        let scoring_provenance = match d.get("scoring_provenance") {
            Some(value @ Value::Object(_)) => CellValue::String(value.to_string()),
            _ => pluck(d, "scoring_provenance"),
        };
        row.insert("scoring_provenance".into(), scoring_provenance);
        row.insert(
            "total_duration_s".into(),
            pluck(d, "total_duration_seconds"),
        );
        row.insert(
            "analysis_duration_s".into(),
            pluck(d, "analysis_duration_seconds"),
        );
        row.insert(
            "duration_within_expected".into(),
            d.get("analysis_duration_seconds")
                .and_then(Value::as_f64)
                .map_or(CellValue::Null, |s| {
                    CellValue::Bool(bekiper_feature::zone::duration_within_expected(s))
                }),
        );
        row.insert("total_distance_m".into(), pluck(d, "total_distance_meters"));
        row.insert("total_mean_speed".into(), pluck(d, "total_mean_speed"));

        let empty: &[Value] = &[];
        let zone_summaries = d
            .get("zone_summaries")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(empty);
        let area_motion_summaries = d
            .get("area_motion_summaries")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(empty);

        let find_zone = |name: &str| -> Option<&Value> {
            zone_summaries.iter().find(|z| {
                z.get("zone_name")
                    .and_then(Value::as_str)
                    .map(|s| s == name)
                    .unwrap_or(false)
            })
        };
        let find_motion = |name: &str| -> Option<&Value> {
            area_motion_summaries.iter().find(|z| {
                z.get("zone_name")
                    .and_then(Value::as_str)
                    .map(|s| s == name)
                    .unwrap_or(false)
            })
        };

        for label in YMAZE_ARM_LABELS {
            insert_zone_columns(
                &mut row,
                &format!("arm_{label}"),
                // Zone entries (criteria #25); `raw_arm_*_entries` is the
                // crossing-sequence count written below.
                &format!("arm_{label}_entries"),
                &format!("in_arm_{label}"),
                &find_zone,
                &find_motion,
            );
        }
        insert_zone_columns(
            &mut row,
            "center",
            "center_entries",
            "in_center",
            &find_zone,
            &find_motion,
        );

        let null = Value::Null;
        let analysis = d.get("analysis").unwrap_or(&null);
        // `valid_total_arm_entries` comes directly from the exported
        // `arm_sequence`, which is the principal instantaneous or configured comparison sequence for
        // threshold-scoped SA scoring. Per-arm `valid_*` counts still come
        // from tallying that same sequence.
        let arm_seq_counts = arm_sequence_counts(analysis.get("arm_sequence"));
        let raw_arm_seq_counts = arm_sequence_counts(analysis.get("raw_arm_sequence"));
        let valid_total = analysis
            .get("arm_sequence")
            .and_then(Value::as_array)
            .map(|values| values.len() as i64);
        let raw_total = analysis
            .get("raw_arm_sequence")
            .and_then(Value::as_array)
            .map(|values| values.len() as i64);
        let total_triads = valid_total.map(|total| total.saturating_sub(2).max(0));
        let correct_alternations = pluck(analysis, "correct_alternations");
        let incorrect_alternations = pluck(analysis, "incorrect_alternations");
        let neutral_triads = pluck(analysis, "neutral_triads");
        let usable_triads = match (&correct_alternations, &incorrect_alternations) {
            (CellValue::Int(correct), CellValue::Int(incorrect)) => {
                CellValue::Int(correct + incorrect)
            }
            _ => CellValue::Null,
        };
        let usable_triads_count = match &usable_triads {
            CellValue::Int(value) => Some(*value),
            _ => None,
        };
        let entry_difference = match (raw_total, valid_total) {
            (Some(raw_total), Some(valid_total)) => CellValue::Int(raw_total - valid_total),
            _ => CellValue::Null,
        };
        let spontaneous_alternation_pct = match (&correct_alternations, &incorrect_alternations) {
            (CellValue::Int(correct), CellValue::Int(incorrect)) if correct + incorrect > 0 => {
                pluck(analysis, "spontaneous_alternation_pct")
            }
            (CellValue::Int(_), CellValue::Int(_)) => CellValue::Null,
            _ => pluck(analysis, "spontaneous_alternation_pct"),
        };
        let spontaneous_alternation_pct_standard = match (&correct_alternations, total_triads) {
            (CellValue::Int(correct), Some(total_triads)) if total_triads > 0 => {
                CellValue::Float(100.0 * (*correct as f64) / (total_triads as f64))
            }
            (CellValue::Int(_), Some(_)) => CellValue::Null,
            _ => CellValue::Null,
        };
        let neutral_triads_pct = match (&neutral_triads, total_triads) {
            (CellValue::Int(neutral), Some(total_triads)) if total_triads > 0 => {
                CellValue::Float(100.0 * (*neutral as f64) / (total_triads as f64))
            }
            (CellValue::Int(_), Some(_)) => CellValue::Null,
            _ => CellValue::Null,
        };
        let (valid_a, valid_b, valid_c) = match arm_seq_counts {
            Some(counts) => (
                CellValue::Int(counts.a as i64),
                CellValue::Int(counts.b as i64),
                CellValue::Int(counts.c as i64),
            ),
            None => (CellValue::Null, CellValue::Null, CellValue::Null),
        };
        let (raw_a, raw_b, raw_c) = match raw_arm_seq_counts {
            Some(counts) => (
                CellValue::Int(counts.a as i64),
                CellValue::Int(counts.b as i64),
                CellValue::Int(counts.c as i64),
            ),
            None => (CellValue::Null, CellValue::Null, CellValue::Null),
        };

        row.insert("raw_arm_A_entries".into(), raw_a);
        row.insert("raw_arm_B_entries".into(), raw_b);
        row.insert("raw_arm_C_entries".into(), raw_c);

        row.insert(
            "zone_entry_sequence".into(),
            sequence_cell(analysis.get("zone_sequence")),
        );
        row.insert(
            "raw_arm_entry_sequence".into(),
            sequence_cell(analysis.get("raw_arm_sequence")),
        );
        row.insert(
            "valid_arm_entry_sequence".into(),
            sequence_cell(analysis.get("arm_sequence")),
        );
        row.insert(
            "first_arm_entered".into(),
            pluck(analysis, "first_arm_entered"),
        );
        row.insert(
            "last_arm_entered".into(),
            pluck(analysis, "last_arm_entered"),
        );
        row.insert("valid_arm_A_entries".into(), valid_a);
        row.insert("valid_arm_B_entries".into(), valid_b);
        row.insert("valid_arm_C_entries".into(), valid_c);
        row.insert(
            "valid_total_arm_entries".into(),
            valid_total.map(CellValue::Int).unwrap_or(CellValue::Null),
        );
        row.insert(
            "raw_total_arm_entries".into(),
            raw_total.map(CellValue::Int).unwrap_or(CellValue::Null),
        );
        row.insert(
            "raw_to_scored_entry_difference".into(),
            entry_difference.clone(),
        );
        row.insert("removed_short_entries".into(), entry_difference);
        row.insert("dwell_rejected_entries".into(), dwell_rejected_entries(d));
        row.insert(
            "total_triads".into(),
            total_triads.map(CellValue::Int).unwrap_or(CellValue::Null),
        );
        row.insert("correct_alternations".into(), correct_alternations);
        row.insert("incorrect_alternations".into(), incorrect_alternations);
        row.insert("usable_triads".into(), usable_triads);
        row.insert("neutral_triads".into(), neutral_triads);
        row.insert("neutral_triads_pct".into(), neutral_triads_pct);
        row.insert(
            "spontaneous_alternation_pct".into(),
            spontaneous_alternation_pct,
        );
        row.insert(
            "spontaneous_alternation_pct_standard".into(),
            spontaneous_alternation_pct_standard,
        );

        // Parallel raw SA view: independent of `apply_dwell_filter`, always
        // scored from the pre-mitigation arm-threshold crossings.
        let raw_correct_alternations = pluck(analysis, "raw_correct_alternations");
        let raw_incorrect_alternations = pluck(analysis, "raw_incorrect_alternations");
        let raw_neutral_triads = pluck(analysis, "raw_neutral_triads");
        let raw_total_triads = raw_total.map(|total| total.saturating_sub(2).max(0));
        let raw_spontaneous_alternation_pct =
            match (&raw_correct_alternations, &raw_incorrect_alternations) {
                (CellValue::Int(correct), CellValue::Int(incorrect)) if correct + incorrect > 0 => {
                    pluck(analysis, "raw_spontaneous_alternation_pct")
                }
                (CellValue::Int(_), CellValue::Int(_)) => CellValue::Null,
                _ => pluck(analysis, "raw_spontaneous_alternation_pct"),
            };
        let raw_spontaneous_alternation_pct_standard =
            match (&raw_correct_alternations, raw_total_triads) {
                (CellValue::Int(correct), Some(total_triads)) if total_triads > 0 => {
                    CellValue::Float(100.0 * (*correct as f64) / (total_triads as f64))
                }
                (CellValue::Int(_), Some(_)) => CellValue::Null,
                _ => CellValue::Null,
            };
        row.insert("raw_correct_alternations".into(), raw_correct_alternations);
        row.insert(
            "raw_incorrect_alternations".into(),
            raw_incorrect_alternations,
        );
        row.insert("raw_neutral_triads".into(), raw_neutral_triads);
        row.insert(
            "raw_spontaneous_alternation_pct".into(),
            raw_spontaneous_alternation_pct,
        );
        row.insert(
            "raw_spontaneous_alternation_pct_standard".into(),
            raw_spontaneous_alternation_pct_standard,
        );

        row.insert("arm_preference".into(), pluck(analysis, "arm_preference"));

        let permutation_distribution = analysis.get("permutation_distribution").unwrap_or(&null);
        // Correct-triad distribution is equivalent to these permutation counts:
        // both count only A/B/C sliding windows with three distinct arms. Both
        // names remain exposed in serialized detail for downstream consumers.
        for (column, key) in [
            ("perm_ABC", "abc"),
            ("perm_ACB", "acb"),
            ("perm_BAC", "bac"),
            ("perm_BCA", "bca"),
            ("perm_CAB", "cab"),
            ("perm_CBA", "cba"),
        ] {
            row.insert(column.into(), pluck(permutation_distribution, key));
        }
        row.insert(
            "permutation_sum".into(),
            pluck(analysis, "sum_of_permutation_distribution"),
        );

        let incorrect_distribution = analysis
            .get("incorrect_triad_distribution")
            .unwrap_or(&null);
        let incorrect_keys = [
            "aab", "aac", "aba", "aca", "bab", "bba", "bbc", "bcb", "cac", "cbc", "cca", "ccb",
        ];
        for key in incorrect_keys {
            row.insert(
                format!("incorrect_{}", key.to_ascii_uppercase()),
                pluck(incorrect_distribution, key),
            );
        }
        row.insert(
            "incorrect_sum".into(),
            sum_distribution(incorrect_distribution, &incorrect_keys),
        );

        let neutral_distribution = analysis.get("neutral_triad_distribution").unwrap_or(&null);
        let neutral_keys = [
            "aaa", "abb", "acc", "baa", "bbb", "bcc", "caa", "cbb", "ccc",
        ];
        for key in neutral_keys {
            row.insert(
                format!("neutral_{}", key.to_ascii_uppercase()),
                pluck(neutral_distribution, key),
            );
        }
        row.insert(
            "neutral_sum".into(),
            sum_distribution(neutral_distribution, &neutral_keys),
        );

        row.insert("seconds_in_arms".into(), pluck(d, "seconds_in_arms"));
        row.insert(
            "seconds_in_arms_and_center".into(),
            pluck(d, "seconds_in_arms_and_center"),
        );
        row.insert("freezing_s".into(), pluck(d, "freezing_seconds"));
        row.insert(
            "low_valid_entries_flag".into(),
            valid_total
                .map(|valid_total| CellValue::Bool(valid_total < 8))
                .unwrap_or(CellValue::Null),
        );
        row.insert(
            "low_usable_triads_flag".into(),
            usable_triads_count
                .map(|usable_triads| CellValue::Bool(usable_triads < 5))
                .unwrap_or(CellValue::Null),
        );
        row.insert(
            "experiment".into(),
            CellValue::String(trial.experiment.clone()),
        );
        row.insert("timepoint".into(), ymaze_timepoint(&trial.experiment));
        row.insert("excluded".into(), pluck(d, "excluded"));

        row
    }

    fn animal_overview_columns(&self) -> &'static [&'static str] {
        YMAZE_ANIMAL_OVERVIEW_COLUMNS
    }
}
