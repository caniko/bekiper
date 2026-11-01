pub(crate) fn variant_analysis_role(experiment: &ExperimentConfig) -> Option<String> {
    let (base_name, variant) = experiment.name.rsplit_once("__")?;
    if base_name.is_empty() || variant.is_empty() {
        return None;
    }

    let role_key = experiment
        .detection
        .as_ref()
        .and_then(|detection| detection.recipe)
        .map(|recipe| recipe.as_str().to_string())
        .or_else(|| crate::aggregate::parse_recipe_from_experiment(base_name))
        .or_else(|| {
            crate::analysis_exclusion::aggregate_role_for_experiment(experiment).map(str::to_string)
        })
        .unwrap_or_else(|| experiment.task_type.as_str().to_string());

    Some(format!("{role_key}__{variant}"))
}

#[cfg(feature = "pose-gpu")]
fn validate_ymaze_source_summary(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
    detailed_result: Option<&TaskDetail>,
) -> Result<()> {
    let perimeters = load_perimeters(exp_dir).unwrap_or_default();
    let behaviour_path = exp_dir.behaviour_parquet();
    let df = if behaviour_path.exists() {
        Some(bekiper_pose::io::scan_file(&behaviour_path)?.collect()?)
    } else {
        None
    };
    validate_ymaze_result(experiment, detailed_result, &perimeters, df.as_ref())
}

pub(crate) fn validate_ymaze_result(
    experiment: &ExperimentConfig,
    detailed_result: Option<&TaskDetail>,
    perimeters: &[PerimeterSpec],
    behaviour_df: Option<&DataFrame>,
) -> Result<()> {
    if experiment.task_type != TaskType::YMaze {
        return Ok(());
    }

    let Some(TaskDetail::YMaze(detail)) = detailed_result else {
        return Ok(());
    };

    let zero_arms = detail
        .analysis
        .arm_entries
        .iter()
        .enumerate()
        .filter_map(|(idx, entries)| {
            let duration = detail
                .zone_summaries
                .iter()
                .find(|summary| summary.zone_name == format!("in_arm_{}", ["A", "B", "C"][idx]))
                .map(|summary| summary.duration_seconds)
                .unwrap_or(0.0);
            (*entries == 0 && duration == 0.0).then_some(["A", "B", "C"][idx])
        })
        .collect::<Vec<_>>();

    let meaningful_movement =
        detail.total_distance_meters > 0.05 || detail.total_duration_seconds > 10.0;
    if zero_arms.len() < 2 || !meaningful_movement {
        return Ok(());
    }

    let mut diagnostics = Vec::new();
    diagnostics.push(format!(
        "arm entries A/B/C = {}/{}/{}",
        detail.analysis.arm_entries[0],
        detail.analysis.arm_entries[1],
        detail.analysis.arm_entries[2]
    ));
    diagnostics.push(format!(
        "arm durations A/B/C = {:.3}/{:.3}/{:.3}s",
        ymaze_zone_duration(detail, "in_arm_A"),
        ymaze_zone_duration(detail, "in_arm_B"),
        ymaze_zone_duration(detail, "in_arm_C")
    ));

    if let Some(df) = behaviour_df {
        diagnostics.push(format!(
            "boolean true frames: in_arm_A={}, in_arm_B={}, in_arm_C={}, in_center={}",
            bool_true_count(df, "in_arm_A").unwrap_or(0),
            bool_true_count(df, "in_arm_B").unwrap_or(0),
            bool_true_count(df, "in_arm_C").unwrap_or(0),
            bool_true_count(df, "in_center").unwrap_or(0)
        ));
        if let Some(bbox) = coordinate_bbox(df, "center_ear_x", "center_ear_y") {
            diagnostics.push(format!(
                "center_ear bbox x={:.4}..{:.4}, y={:.4}..{:.4}",
                bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y
            ));
        }
        if let Some(bbox) = coordinate_bbox(df, "base_tail_x", "base_tail_y") {
            diagnostics.push(format!(
                "base_tail bbox x={:.4}..{:.4}, y={:.4}..{:.4}",
                bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y
            ));
        }
    }
    diagnostics.extend(ymaze_perimeter_diagnostics(perimeters));

    Err(BekiperError::Config(format!(
        "{}: invalid Y-maze source summary; meaningful movement is present but at least two arms have zero entries/duration (zero arms: {}). This usually means DLC pose coordinates and radial-maze annotations are not in the same coordinate frame. Fix upstream DLC pose data or Y-maze manual annotation reference mapping, then regenerate behaviour and summary artifacts. {}",
        experiment.name,
        zero_arms.join(", "),
        diagnostics.join("; ")
    )))
}

fn ymaze_zone_duration(detail: &bekiper_behaviour::radial_arm::YMazeResult, zone: &str) -> f64 {
    detail
        .zone_summaries
        .iter()
        .find(|summary| summary.zone_name == zone)
        .map(|summary| summary.duration_seconds)
        .unwrap_or(0.0)
}

fn bool_true_count(df: &DataFrame, column: &str) -> Option<u32> {
    df.column(column).ok()?.bool().ok()?.sum()
}

fn ymaze_perimeter_diagnostics(perimeters: &[PerimeterSpec]) -> Vec<String> {
    perimeters
        .iter()
        .find_map(|perimeter| {
            if let PerimeterSpec::RadialMaze {
                center_vertices,
                arms,
                ..
            } = perimeter
            {
                let mut diagnostics = Vec::new();
                diagnostics.push(format!(
                    "center bbox {}",
                    vertex_bbox_string(center_vertices)
                ));
                for (idx, arm) in arms.iter().enumerate() {
                    diagnostics.push(format!(
                        "arm_{} bbox {}",
                        ["A", "B", "C"].get(idx).copied().unwrap_or("?"),
                        vertex_bbox_string(arm)
                    ));
                }
                Some(diagnostics)
            } else {
                None
            }
        })
        .unwrap_or_default()
}
