/// Join against explicit canonical session keys, with a threshold-scoped
/// one-to-many allowance only for principal Y-maze 25/45/65 inputs.
fn attach_native_trials(
    metadata: &[Record],
    trials: &[TrialResult],
    plan: &Record,
) -> Result<Vec<Record>> {
    let mut measurements: BTreeMap<inputs::SessionKey, Vec<Record>> = BTreeMap::new();
    let mut comparisons: BTreeMap<inputs::SessionKey, Vec<Record>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let indexed: BTreeMap<_, _> = metadata
        .iter()
        .map(|row| native(inputs::session_key(row)).map(|key| (key, row)))
        .collect::<Result<_>>()?;
    for trial in trials {
        let assay = match trial.task_type {
            TaskType::OpenField => "oft",
            TaskType::YMaze => "ymaze",
            TaskType::ObjectRecognition => "nort",
            TaskType::Cheeseboard => continue,
        };
        let nort_detail = if assay == "nort" {
            // The aggregate extractor assumes this typed payload; validate it
            // before calling the extractor rather than exposing its panic path.
            let detail = native(serde_json::from_value::<
                bekiper_behaviour::object_recognition::NortResult,
            >(trial.detailed.clone()))?;
            if !matches!(
                detail.protocol,
                bekiper_core::types::ObjectRecognitionProtocol::NortTraining
                    | bekiper_core::types::ObjectRecognitionProtocol::NortNovelty
            ) {
                continue;
            }
            require(
                detail.object_labels.len() == detail.exploration_times.len()
                    && detail.object_labels.len() == detail.bout_counts.len(),
                "native NORT object arrays have different lengths",
            )?;
            let distinct: BTreeSet<_> = detail.object_labels.iter().collect();
            require(
                distinct.len() == detail.object_labels.len()
                    && detail
                        .object_labels
                        .iter()
                        .all(|label| !label.trim().is_empty()),
                "native NORT object identities must be distinct and nonempty",
            )?;
            Some(detail)
        } else {
            None
        };
        let mut raw: serde_json::Map<String, Value> = experiment_for_trial(trial)
            .extract_row(trial)
            .into_iter()
            .map(|(name, value)| cell_value(value).map(|value| (name, value)))
            .collect::<Result<serde_json::Map<_, _>>>()?;
        raw.retain(|_, value| !value.is_null());
        // Biological identities in this extractor are display fields. Canonical
        // biology is validated from the supplied session metadata instead.
        for name in [
            "animal",
            "sex",
            "age",
            "gene",
            "cohort",
            "excluded",
            "timepoint",
            "phase",
        ] {
            raw.remove(name);
        }
        if let Some(value) = trial.detailed.get("excluded") {
            raw.insert("upstream_excluded".into(), value.clone());
        }
        raw.insert("experiment".into(), trial.experiment.clone().into());
        raw.insert(
            "recording_id".into(),
            trial
                .experiment
                .split("__")
                .next()
                .unwrap_or(&trial.experiment)
                .into(),
        );
        raw.insert("analysis_id".into(), trial.experiment.clone().into());
        raw.insert("assay".into(), assay.into());
        raw.insert(
            "native_session_context".into(),
            trial.session_context.clone(),
        );
        // NORT's total duration is the observed selected window, while
        // analysis_window_s is the nominal requested duration. A truncated
        // recording must not acquire a longer validation window.
        if assay == "nort" {
            if let Some(duration) = raw.remove("total_duration_s") {
                raw.insert("analysis_duration_s".into(), duration);
            }
        } else if assay == "oft" && !trial.detailed["window"].is_null() {
            raw.remove("total_duration_s");
            let window = &trial.detailed["window"];
            raw.insert("analysis_start_s".into(), window["start_seconds"].clone());
            if window["policy"] == "first_finite_centroid_half_open_v1"
                && trial.detailed["total_duration_seconds"]
                    .as_f64()
                    .is_some_and(|d| d > 0.0)
            {
                raw.insert("first_detection_s".into(), window["start_seconds"].clone());
            }
            raw.insert(
                "native_window_policy".into(),
                trial.detailed["window"].clone(),
            );
        } else if (assay == "ymaze" || raw.contains_key("analysis_duration_s"))
            && let Some(duration) = raw.remove("total_duration_s")
        {
            raw.insert("source_video_duration_s".into(), duration);
        }
        if let Some(context) = trial.session_context.as_object() {
            for (field, value) in context {
                // Legacy trial biology is evidence, not the cohort authority.
                if matches!(
                    field.as_str(),
                    "group"
                        | "sex"
                        | "treatment"
                        | "genotype"
                        | "hcar1"
                        | "five_xfad"
                        | "age_treatment_window"
                ) {
                    continue;
                }
                if !matches!(field.as_str(), "assay" | "phase" | "timepoint") {
                    require(
                        raw.get(field).is_none_or(|native| native == value),
                        format!("native session context conflicts with fresh measurement: {field}"),
                    )?;
                }
                raw.insert(field.clone(), value.clone());
            }
        }
        for field in [
            "group",
            "sex",
            "treatment",
            "hcar1",
            "five_xfad",
            "age_treatment_window",
        ] {
            if let Some(value) = trial
                .animal_metadata
                .get(field)
                .filter(|value| !value.is_null())
            {
                raw.insert(field.into(), value.clone());
            }
        }
        if let Some(animal) = trial
            .animal_metadata
            .get("animal")
            .filter(|value| !value.is_null())
            && !raw.contains_key("animal_id")
            && !raw.contains_key("animal")
        {
            raw.insert("animal".into(), animal.clone());
        }
        if let Some(detail) = &nort_detail {
            raw.insert("native_object_measurements".into(), json!(detail.object_labels.iter()
                .zip(&detail.exploration_times).zip(&detail.bout_counts)
                .map(|((object_id, seconds), bouts)| json!({"object_id":object_id,"exploration_s":seconds,"bouts":bouts}))
                .collect::<Vec<_>>()));
        }
        for field in [
            "scoring_provenance",
            "effective_parameters",
            "native_state_distributions",
            "qc_flags",
            "missing_reasons",
            "disagreements",
            "candidate_segments",
            "coordinate_samples",
            "coordinate_sample_end_s",
            "geometry",
            "center_intervals",
            "first_detection_s",
            "analysis_start_s",
            "synthetic_observation",
        ] {
            if let Some(value) = trial.detailed.get(field) {
                raw.insert(field.into(), value.clone());
            }
        }
        if assay == "ymaze" {
            // The extractor retains the historical signed count difference and
            // separately exports arm-only, evidence-backed dwell rejections.
            for (source, target) in [
                ("arm_sequence", "valid_arm_entry_sequence"),
                ("raw_arm_sequence", "raw_arm_entry_sequence"),
            ] {
                if let Some(value) = trial.detailed["analysis"].get(source) {
                    raw.insert(target.into(), value.clone());
                }
            }
        }
        let mut row = native(inputs::normalize(
            &Value::Object(raw),
            &plan["inputs"]["column_mapping"][assay],
        ))?;
        native(inputs::apply_session_value_mappings(
            &mut row,
            &plan["inputs"]["session_key_mapping"],
        ))?;
        require(
            row["assay"] == assay,
            "native session assay conflicts with trial task type",
        )?;
        complete_registered_session_key(&mut row, metadata)?;
        let key = native(inputs::session_key(&row))?;
        if let Some(animal) = trial
            .animal_metadata
            .get("animal")
            .filter(|animal| !animal.is_null())
        {
            let canonical = native(inputs::normalize(&json!({"animal":animal}), &Value::Null))?;
            require(
                row["animal_id"] == canonical["animal_id"],
                "native session context conflicts with trial animal identity",
            )?;
        }
        if let Some(detail) = &nort_detail {
            require(
                row["phase"]
                    == if detail.protocol
                        == bekiper_core::types::ObjectRecognitionProtocol::NortTraining
                    {
                        "training"
                    } else {
                        "test"
                    },
                "native NORT protocol conflicts with canonical phase",
            )?;
        }
        let expected = indexed.get(&key).ok_or_else(|| {
            BekiperError::Data(format!(
                "native trial {} has no matching registered metadata session",
                trial.experiment
            ))
        })?;
        for field in [
            "five_xfad",
            "hcar1",
            "treatment",
            "group",
            "sex",
            "age_treatment_window",
            "excluded",
            "exclusion_reason",
            "pair_id",
            "object_configuration",
        ] {
            if let Some(value) = row.get(field) {
                require(
                    expected.get(field) == Some(value),
                    format!("native trial metadata conflicts with plan: {field}"),
                )?;
            }
        }
        let threshold = if assay == "ymaze" {
            let threshold = row["threshold_pct"]
                .as_f64()
                .or_else(|| {
                    row["proximal_threshold"]
                        .as_f64()
                        .map(|value| value * 100.0)
                })
                .ok_or_else(|| {
                    BekiperError::Data("native Y-maze result needs explicit threshold".into())
                })?;
            if let Some(value) = row
                .get("proximal_threshold")
                .filter(|value| !value.is_null())
            {
                require(
                    value
                        .as_f64()
                        .is_some_and(|value| value * 100.0 == threshold),
                    "native Y-maze threshold_pct conflicts with proximal_threshold",
                )?;
            }
            row["threshold_pct"] = json!(threshold);
            if threshold == 0.0 && is_native_ymaze_comparison(trial) {
                require(
                    seen.insert((
                        key.clone(),
                        Some(0),
                        trial
                            .analysis_role
                            .clone()
                            .unwrap_or_else(|| trial.experiment.clone()),
                    )),
                    "duplicate native comparison session",
                )?;
                let mut comparison = (*expected).clone();
                comparison
                    .as_object_mut()
                    .unwrap()
                    .extend(row.as_object().unwrap().clone());
                comparison["native_result"] = native(serde_json::to_value(trial))?;
                comparison["analysis_role"] = trial
                    .analysis_role
                    .clone()
                    .unwrap_or_else(|| {
                        format!(
                            "ymaze__{}",
                            trial
                                .experiment
                                .rsplit_once("__")
                                .map(|(_, v)| v)
                                .unwrap_or("comparison")
                        )
                    })
                    .into();
                comparison["routing"] = "native_comparison_only".into();
                comparison["measurement_available"] = true.into();
                comparisons.entry(key).or_default().push(comparison);
                continue;
            }
            // Cross-line geometry (owner decision 2026-10-28) makes 0% a principal context.
            let crossline =
                row["scoring_provenance"]["geometry_version"] == "imen_arm_start_crossline_v1";
            require(
                !is_native_ymaze_comparison(trial)
                    && ([25.0, 45.0, 65.0].contains(&threshold) || (crossline && threshold == 0.0)),
                "native principal Y-maze threshold must be 25/45/65; marked comparison must be 0",
            )?;
            Some(threshold as u8)
        } else {
            None
        };
        require(
            seen.insert((key.clone(), threshold, String::new())),
            "duplicate native trial session/threshold",
        )?;
        row["measurement_available"] = true.into();
        row["native_source"] = json!({"experiment":trial.experiment,"task_type":trial.task_type,"analysis_role":trial.analysis_role});
        measurements.entry(key).or_default().push(row);
    }
    let mut rows = Vec::new();
    for meta in metadata {
        let key = native(inputs::session_key(meta))?;
        let mut comparison = comparisons.remove(&key);
        if let Some(values) = measurements.remove(&key) {
            for value in values {
                let mut row = meta.clone();
                row.as_object_mut()
                    .unwrap()
                    .extend(value.as_object().unwrap().clone());
                merge_native_qc(&mut row, meta, &value)?;
                if row["assay"] == "nort" {
                    map_native_nort_roles(&mut row, plan)?;
                }
                if let Some(values) = comparison.take() {
                    row["native_comparisons"] = json!(values);
                }
                rows.push(row);
            }
        } else {
            let mut row = meta.clone();
            if let Some(values) = comparison {
                row["native_comparisons"] = json!(values);
            }
            rows.push(row);
        }
    }
    Ok(rows)
}
