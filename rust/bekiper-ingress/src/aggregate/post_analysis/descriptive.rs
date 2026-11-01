//! Adapt native results without plan-based identity inference or rescoring.
use super::*;

pub(super) fn rows(trials: &[TrialResult]) -> Result<(Vec<Record>, Vec<Record>)> {
    let mut rows = Vec::new();
    let mut omissions = Vec::new();
    let mut seen = BTreeSet::new();
    for trial in trials {
        let assay = match trial.task_type {
            TaskType::OpenField => "oft",
            TaskType::YMaze => "ymaze",
            TaskType::ObjectRecognition => "nort",
            TaskType::Cheeseboard => continue,
        };
        let context = &trial.session_context;
        let metadata = &trial.animal_metadata;
        let retained_metadata = trial
            .detailed
            .get("recorded_trial_metadata")
            .unwrap_or(metadata);
        let text = |value: &Value| match value {
            Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        };
        let animal = text(&context["animal"])
            .or_else(|| text(&retained_metadata["animal"]))
            .or_else(|| text(&metadata["animal"]));
        let Some(animal) = animal.filter(|a| !a.trim().is_empty()) else {
            omissions.push(json!({"analysis_id":trial.experiment,"reason":"recorded_animal_identity_unavailable"}));
            continue;
        };
        for source in [retained_metadata, metadata] {
            if let Some(id) = text(&source["animal"]) {
                require(animal == id, "recorded trial/metadata animal conflict")?;
            }
        }
        let mut row = json!({});
        // Validate the typed object detail before calling the extractor's typed
        // boundary; a malformed native payload must not become a panic.
        let mut labelled_objects = Vec::new();
        if assay == "nort" {
            let detail: bekiper_behaviour::object_recognition::NortResult =
                native(serde_json::from_value(trial.detailed.clone()))?;
            require(
                detail.object_labels.len() == detail.exploration_times.len()
                    && detail.object_labels.len() == detail.bout_counts.len(),
                "NORT object array shape mismatch",
            )?;
            require(
                detail.object_labels.iter().collect::<BTreeSet<_>>().len()
                    == detail.object_labels.len(),
                "duplicate NORT object labels",
            )?;
            labelled_objects.extend(
                detail
                    .object_labels
                    .iter()
                    .zip(&detail.exploration_times)
                    .map(|(label, seconds)| (format!("{label}_exploration_s"), *seconds)),
            );
            row["object_roles"] = json!({"labels":detail.object_labels,"recorded_indices":detail.indices,
                "object_a_placement":context["object_a"],"object_b_placement":context["object_b"]});
        }
        let mut endpoints = Vec::new();
        let extractor = experiment_for_trial(trial);
        let static_columns = extractor.static_columns();
        for (key, value) in extractor.extract_row(trial) {
            let value = cell_value(value)?;
            let identity_or_setting = [
                "animal",
                "cohort",
                "gene",
                "sex",
                "age",
                "timepoint",
                "fps",
                "proximal_threshold",
                "start_mode",
                "zone_max_gap_seconds",
                "pose_gap_fill_seconds",
                "scoring_provenance",
                "experiment",
                "excluded",
                "window_mode",
                "window_start_frame",
                "window_end_frame_exclusive",
                "first_approach_frame",
                "analysis_window_s",
                "object_A_exploration_s",
                "object_B_exploration_s",
                "object_A_bouts",
                "object_B_bouts",
                "object_A_label",
                "object_B_label",
                "object_A_placement",
                "object_B_placement",
                "zone_entry_sequence",
                "raw_arm_entry_sequence",
                "valid_arm_entry_sequence",
                "first_arm_entered",
                "last_arm_entered",
                "arm_preference",
            ]
            .contains(&key.as_str())
                || key.ends_with("_flag");
            if !identity_or_setting
                && (value.is_number() || (value.is_null() && static_columns.contains(&key)))
            {
                endpoints.push(key.clone());
            }
            row[&key] = value;
        }
        for (field, seconds) in labelled_objects {
            require(
                seconds.is_finite() && seconds >= 0.0,
                "invalid native exploration duration",
            )?;
            row[&field] = seconds.into();
            if !endpoints.contains(&field) {
                endpoints.push(field); // coverage: unreachable — the NORT extractor already emits every {label}_exploration_s endpoint
            }
        }
        let recording = trial
            .experiment
            .split_once("__")
            .map_or(trial.experiment.as_str(), |(r, _)| r);
        let variant = trial.experiment.split_once("__").map(|(_, v)| v);
        let timepoint = context["timepoint"]
            .as_str()
            .or_else(|| match context["phase_part"].as_str() {
                Some("pre") => Some("pre"),
                Some("post") => Some("post"),
                Some("1") if context["profile"] == "videomap" => Some("pre"),
                Some("2") if context["profile"] == "videomap" => Some("post"),
                _ => None,
            })
            // A Y-maze recording without a metadata row keeps the
            // `_pre_`/`_post_` timepoint of its name, as the workbook does.
            .or_else(|| {
                (assay == "ymaze")
                    .then(|| super::super::ymaze::ymaze_recording_timepoint(recording))
                    .flatten()
            });
        let phase = if assay == "nort" {
            match trial.detailed["protocol"].as_str() {
                Some("nort_training") => Some("training"),
                Some("nort_novelty") => Some("test"),
                _ => context["phase"]
                    .as_str()
                    .or_else(|| context["stage"].as_str()),
            }
        } else {
            context["phase"].as_str()
        };
        row["animal_id"] = animal.into();
        row["assay"] = assay.into();
        row["session_id"] = context["session_id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(recording)
            .into();
        row["session_id_origin"] = if context["session_id"].is_string() {
            "recorded_context"
        } else {
            "analysis_recording_identity"
        }
        .into();
        row["recording_id"] = recording.into();
        row["analysis_id"] = trial.experiment.clone().into();
        row["analysis_role"] = json!(trial.analysis_role);
        row["variant"] = json!(variant);
        row["timepoint"] = timepoint.unwrap_or("unrecorded").into();
        row["phase"] = phase.unwrap_or("unrecorded").into();
        row["recorded_group"] = text(&context["group"])
            .or_else(|| text(&retained_metadata["group"]))
            .or_else(|| text(&metadata["group"]))
            .or_else(|| text(&metadata["cohort"]))
            .into();
        row["group"] = row["recorded_group"]
            .as_str()
            .unwrap_or("Unrecorded group")
            .into();
        row["excluded"] = (retained_metadata["excluded"] == true
            || metadata["excluded"] == true
            || trial.detailed["excluded"] == true)
            .into();
        row["exclusion_reason"] = retained_metadata
            .get("exclusion_reason")
            .filter(|reason| !reason.is_null())
            .unwrap_or(&metadata["exclusion_reason"])
            .clone();
        let mut collisions = Vec::new();
        for field in ["group", "sex", "age", "genotype", "treatment"] {
            let sources = [
                ("session_context", text(&context[field])),
                ("recorded_trial_metadata", text(&retained_metadata[field])),
                ("aggregate_metadata", text(&metadata[field])),
            ];
            let values = sources
                .iter()
                .filter_map(|(_, value)| value.as_ref())
                .collect::<BTreeSet<_>>();
            if values.len() > 1 {
                collisions.push(json!({"field":field,"sources":sources,"resolution":"recorded_context_then_trial_metadata"}));
                omissions.push(json!({"analysis_id":trial.experiment,"field":field,"reason":"recorded_metadata_collision","retained":true}));
            }
        }
        row["metadata_collisions"] = json!(collisions);
        for field in [
            "scoring_provenance",
            "native_state_distributions",
            "qc_flags",
            "missing_reasons",
        ] {
            row[field] = trial.detailed[field].clone();
        }
        row["threshold_pct"] = row["proximal_threshold"].as_f64().map(|v| v * 100.0).into();
        row["native_session_context"] = context.clone();
        row["native_animal_metadata"] = metadata.clone();
        // Geometry and event observations belong to the individual recording.
        // Policies/implementation identities partition cohorts, while the full
        // evidence remains on each measurement and plotted point.
        let mut lineage = row["scoring_provenance"].clone();
        if let Some(fields) = lineage.as_object_mut() {
            for field in [
                "center",
                "center_polygon",
                "arm_polygons",
                "threshold_radii_squared",
                "arm_frames",
                "crossline_endpoints",
                "analysis_start_frame",
                "analysis_start_timestamp_s",
                "sample_end_timestamp_s",
                "entry_frames",
                "raw_entry_frames",
                "dwell_bouts",
            ] {
                fields.remove(field);
            }
        }
        row["scoring_lineage"] = lineage.clone();
        row["presentation_partition"] = json!({"timepoint":row["timepoint"],"phase":row["phase"],
            "stage":context["stage"],"sequence":context["sequence"],"variant":variant,
            "threshold_pct":row["threshold_pct"],"scoring_lineage":lineage});
        row["presentation_label"] = format!(
            "{assay} · {} · {} · {}",
            row["timepoint"].as_str().unwrap(),
            row["phase"].as_str().unwrap(),
            variant.unwrap_or("recorded")
        )
        .into();
        row["scalar_endpoints"] = json!(endpoints);
        for (field, value) in trial.detailed.as_object().into_iter().flatten() {
            if field != "native_state_distributions" {
                row["recorded_detail"][field] = value.clone();
            }
        }
        for field in ["timepoint", "phase", "recorded_group"] {
            if row[field] == "unrecorded" || row[field].is_null() {
                omissions.push(json!({"analysis_id":trial.experiment,"field":field,"reason":"recorded_metadata_unavailable","retained":true}));
            }
        }
        require(
            seen.insert(trial.experiment.clone()),
            "duplicate descriptive analysis identity",
        )?;
        rows.push(row);
    }
    Ok((rows, omissions))
}

pub(super) fn run(
    config: &IngressConfig,
    run_root: &Path,
    trials: &[TrialResult],
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Record> {
    require(!cancelled(), "descriptive export cancelled")?;
    let mut enriched = trials.to_vec();
    let mut fingerprints = config.post_analysis.frozen_fingerprints.clone();
    if let Some(settings) = &config.aggregate.metadata {
        let base = super::super::run::aggregate_metadata_base_dir(config, run_root);
        let path = base.join(&settings.path);
        let before = native(inputs::fingerprint(&path))?;
        let workbook = super::super::metadata::load_aggregate_metadata(&base, settings)?;
        require(
            before == native(inputs::fingerprint(&path))?,
            "metadata changed during descriptive capture",
        )?; // coverage: unreachable — needs the workbook to change between two reads
        fingerprints.push(before);
        let originals = enriched.clone();
        super::super::metadata::enrich_trials_with_metadata(
            &mut enriched,
            &workbook,
            crate::config::MissingMetadataPolicy::AllowBlanks,
        )?; // coverage: unreachable — AllowBlanks enrichment never fails
        // Preserve both recorded metadata sources, including collisions, rather
        // than overwriting the original evidence at this serialization boundary.
        for (trial, original) in enriched.iter_mut().zip(originals) {
            trial.detailed["recorded_trial_metadata"] = original.animal_metadata;
        }
    }
    let (rows, omissions) = rows(&enriched)?;
    for trial in &enriched {
        let mut fingerprint = bekiper_post_analysis::source::embedded_fingerprint(
            "native_trial.json",
            &native(serde_json::to_string(trial))?,
        );
        fingerprint["role"] = "native_trial_result".into();
        fingerprint["analysis_id"] = trial.experiment.clone().into();
        fingerprints.push(fingerprint);
    }
    let settings = &config.post_analysis;
    let presentation = json!({"analysis_mode":"recorded_descriptive","run_mode":"descriptive",
        "execution":{"ymaze_primary_thresholds":settings.ymaze_primary_thresholds.clone().unwrap_or_default(),"contexts":settings.contexts.clone().unwrap_or_default(),"render_workers":settings.render_workers,
            "primary_workers":settings.primary_workers.min(config.project.runtime.num_threads.unwrap_or(settings.primary_workers))},
        "outputs":{"figures_enabled":settings.figures.enabled,
        "figure_selection":settings.figures.selection,"figure_dpi":settings.figures.dpi,"figure_formats":settings.figures.formats,
         "report":{"enabled":settings.report.enabled,"title":settings.report.title,"compiler":settings.report.compiler}}});
    let run_id = settings.run_id.clone().unwrap_or_else(|| {
        format!(
            "cohort-{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            std::process::id()
        )
    });
    native(
        bekiper_post_analysis::pipeline::publish_descriptive_with_cancel(
            rows,
            omissions,
            fingerprints,
            presentation,
            &post_analysis_output_path(run_root, &run_id)?,
            cancelled,
        ),
    )
}
