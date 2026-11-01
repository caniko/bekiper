/// Build an InspectionManifest from experiment config and analysis results.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_inspection_manifest(
    experiment: &ExperimentConfig,
    labels: &[String],
    summaries: &[HeuristicSummary],
    decompositions: &[HeuristicDecompositionSpec],
    perimeters: &[PerimeterSpec],
    quadrants: &[bekiper_core::inspection::QuadrantSpec],
    total_frames: u64,
    meters_per_pixel: f64,
    scale_derivation: Option<ScaleDerivationInfo>,
    detailed_result: Option<&TaskDetail>,
    evaluated_df: Option<&polars::prelude::DataFrame>,
) -> InspectionManifest {
    let coordinate_columns: HashMap<String, CoordinateColumnSpec> = labels
        .iter()
        .map(|label| {
            (
                label.clone(),
                CoordinateColumnSpec {
                    x: format!("{label}_x"),
                    y: format!("{label}_y"),
                },
            )
        })
        .collect();

    // Build heuristic metadata for the inspection manifest.
    // Prefer the decomposition's declared result column when available so the
    // inspection figures reflect the same semantics as the exported summaries.
    let decomposition_result_columns: HashMap<&str, &str> = decompositions
        .iter()
        .map(|spec| (spec.heuristic_name.as_str(), spec.result_column.as_str()))
        .collect();
    let mut heuristics: Vec<HeuristicMeta> = summaries
        .iter()
        .map(|s| {
            let result_column = decomposition_result_columns
                .get(s.name.as_str())
                .map(|column| (*column).to_string())
                .unwrap_or_else(|| s.name.clone());
            HeuristicMeta {
                name: s.name.clone(),
                result_column,
                true_frames: s.true_frames,
                total_frames: s.total_frames,
                seconds: s.seconds,
                truth_spans: vec![],
            }
        })
        .collect();

    // Per-feature inspection figure declarations — drives the
    // `bekiper inspect plot` overlay loop. Computed from the perimeters,
    // quadrants, heuristics, and coordinate columns we just assembled.
    let scalar_bindings = decompositions
        .iter()
        .flat_map(|s| &s.components)
        .filter_map(|c| Some((c.value_column.as_deref()?, c.value_unit.as_deref()?)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let motion_columns = scalar_bindings.into_iter().collect::<Vec<_>>();
    let mut feature_inspections = crate::feature_inspect::collect_feature_inspections(
        perimeters,
        quadrants,
        &heuristics,
        decompositions,
        &coordinate_columns,
        None,
        scale_derivation.as_ref(),
        &[],
        &motion_columns,
        &[],
    );
    // Exploration computes the same full-trial nose speed for each target.
    // Keep one native trace alongside the skeleton, including disabled-speed
    // policies where this scalar is still useful descriptively.
    let nose_speed_column = summaries
        .iter()
        .filter(|summary| summary.name.starts_with("exploring_"))
        .map(|summary| {
            bekiper_feature::heuristic::diag_column_name(&summary.name, "nose_speed_mps")
        })
        .filter(|column| evaluated_df.is_some_and(|df| df.column(column).is_ok()))
        .min();
    if let Some(value_column) = nose_speed_column {
        feature_inspections.push(bekiper_core::inspection::FeatureInspection::TimeSeries {
            label: "timeseries_nose_speed".into(),
            value_column,
            unit: "m/s".into(),
            description: "Native nose speed; descriptive even when speed filtering is disabled"
                .into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: Some("Nose speed (descriptive)".into()),
            story_key: Some("pose_quality".into()),
        });
    }
    if let Some(skeleton_feature) = skeleton_feature_from_labels(labels) {
        if let bekiper_core::inspection::FeatureInspection::Trajectory {
            coordinate_labels, ..
        } = &skeleton_feature
        {
            let count = total_frames.min(6);
            for sample in 0..count {
                let index = if count > 1 {
                    sample * total_frames.saturating_sub(1) / (count - 1)
                } else {
                    0
                };
                feature_inspections.push(bekiper_core::inspection::FeatureInspection::Trajectory {
                    label: format!("skeleton_geometry_{sample}"),
                    coordinate_labels: coordinate_labels.clone(),
                    description: format!(
                        "Evenly spaced geometry/skeleton check at evaluated row {index}"
                    ),
                    representative_frame_index: Some(index),
                    color: None,
                    linewidth: None,
                    title_override: None,
                    story_key: Some("geometry_examples".into()),
                });
            }
        } // coverage: unreachable — skeleton features are always trajectories
        feature_inspections.insert(0, skeleton_feature);
    }

    let ymaze_overrides = experiment
        .task_params
        .as_ref()
        .and_then(|params| params.ymaze.as_ref());
    feature_inspections.extend(
        crate::feature_inspect::collect_ymaze_arm_section_inspections(
            experiment.task_type,
            perimeters,
            None,
            ymaze_overrides.and_then(|ymaze| ymaze.proximal_threshold),
            detailed_result,
        ),
    );

    // Append task-specific detailed figure declarations (NORT, Y-maze, OFT).
    feature_inspections.extend(crate::feature_inspect::collect_task_detailed_inspections(
        experiment.task_type,
        detailed_result,
        None,
    ));
    feature_inspections.extend(crate::feature_inspect::collect_detection_field_inspections(
        experiment.task_type,
        detailed_result,
        decompositions,
        &experiment
            .task_params
            .as_ref()
            .and_then(|params| params.head_coords.as_ref())
            .map(|coords| bekiper_feature::heuristic::coordinate_label_from_xy(&coords.x))
            .unwrap_or_else(|| "center_ear".into()),
    ));
    let preference = feature_inspections.iter().find_map(|feature| {
        if let bekiper_core::inspection::FeatureInspection::NortDiscrimination { .. } = feature {
            let mut preference = feature.clone();
            if let bekiper_core::inspection::FeatureInspection::NortDiscrimination {
                label,
                story_key,
                description,
                ..
            } = &mut preference
            {
                *label = "preference_summary".into();
                *story_key = Some("preference".into());
                *description =
                    "Measured object exploration shares; native per-channel totals and protocol"
                        .into();
            }
            Some(preference)
        } else {
            None
        }
    });
    feature_inspections.extend(preference);
    if experiment.task_type == TaskType::ObjectRecognition {
        let centroid_x = experiment
            .task_params
            .as_ref()
            .and_then(|params| params.centroid_coords.as_ref())
            .map(|coords| coords.x.as_str())
            .unwrap_or("mouse_center_x");
        let centroid = bekiper_feature::heuristic::coordinate_label_from_xy(centroid_x);
        for feature in &mut feature_inspections {
            if let bekiper_core::inspection::FeatureInspection::CenterPeriphery {
                coordinate_labels,
                ..
            } = feature
            {
                *coordinate_labels = vec![centroid.clone()];
            }
        }
    }

    // Post-process: populate representative_frame_index for every feature
    // that doesn't already have one, using the evaluated DataFrame. This
    // eliminates the Python-side fallback heuristic.
    // Also populate truth_spans on HeuristicMeta so Python can render
    // timelines without re-reading boolean columns from the parquet.
    if let Some(df) = evaluated_df {
        feature_inspections.extend(inspection_states::state_inspections(
            df,
            decompositions,
            perimeters,
            experiment.effective_fps().unwrap_or(0.0),
        ));
        feature_inspections.extend(inspection_states::vector_examples(df, decompositions));
        crate::feature_inspect::populate_representative_frames(
            &mut feature_inspections,
            df,
            &coordinate_columns,
            total_frames,
        );
        crate::feature_inspect::populate_truth_spans(&mut heuristics, df);
        crate::feature_inspect::populate_occupancy_grids(
            &mut feature_inspections,
            df,
            &coordinate_columns,
            quadrants,
        );
    }

    // Post-process: populate story_key for grouping features into
    // multi-panel figures. This moves the grouping logic from Python's
    // _grouping.py to Rust so the Python side is purely rendering.
    crate::feature_inspect::populate_story_keys(&mut feature_inspections, &heuristics, perimeters);

    let mut manifest = InspectionManifest {
        video: VideoMetadata {
            fps: experiment.effective_fps().unwrap_or(0.0),
            total_frames,
            resolution: experiment.video_resolution.unwrap_or((0, 0)),
            meters_per_pixel: MetersPerPixel(meters_per_pixel),
        },
        video_path: experiment.video.clone(),
        session_context: experiment.session_context.clone(),
        frame_image: experiment.frame_image.clone(),
        labels: labels.to_vec(),
        label_colors: labels
            .iter()
            .cloned()
            .chain(
                perimeters
                    .iter()
                    .filter(|p| !["arena", "center", "center_zone"].contains(&p.label().as_str()))
                    .map(|p| p.label().to_string()),
            )
            .map(|label| {
                let color = bekiper_core::inspection::inspection_identity_color(&label).to_string();
                (label, color)
            })
            .collect(),
        coordinate_columns,
        perimeters: perimeters.to_vec(),
        heuristics,
        settings: InspectionSettings {
            invert_y: false,
            minimum_seconds_tolerance: match detailed_result {
                Some(TaskDetail::ObjectRecognition(r)) => r
                    .scoring_provenance
                    .as_ref()
                    .and_then(|p| p["minimum_dwell_seconds"].as_f64())
                    .unwrap_or(
                        bekiper_feature::heuristic::combined::exploration::DEFAULT_MIN_BOUT_SECONDS,
                    ),
                _ => 0.5,
            },
            maximum_seconds_distraction: match detailed_result {
                Some(TaskDetail::ObjectRecognition(r)) => r
                    .scoring_provenance
                    .as_ref()
                    .and_then(|p| p["merge_gap_seconds"].as_f64())
                    .unwrap_or(
                        bekiper_feature::heuristic::combined::exploration::DEFAULT_MAX_GAP_SECONDS,
                    ),
                _ => 1.0 / 3.0,
            },
            meters_per_pixel,
        },
        scale_derivation,
        quadrants: quadrants.to_vec(),
        feature_inspections,
        pixel_perimeters: vec![],
        pixel_quadrants: vec![],
    };

    // Pre-compute pixel-space geometry via the ToVideoPixels trait.
    // This ensures every geometry type drawn on inspection figures goes
    // through the same coordinate conversion — the trait bound prevents
    // new types from silently skipping the conversion.
    manifest.compute_pixel_geometry();

    manifest
}
