fn skeleton_feature_from_labels(
    labels: &[String],
) -> Option<bekiper_core::inspection::FeatureInspection> {
    let has = |label: &str| labels.iter().any(|candidate| candidate == label);
    let coordinate_labels = if has("nose")
        && has("left_ear")
        && has("right_ear")
        && has("center_ear")
        && has("base_tail")
    {
        vec!["nose", "left_ear", "center_ear", "right_ear", "base_tail"]
    } else if has("nose") && has("left_ear") && has("right_ear") && has("center_ear") && has("tail")
    {
        vec!["nose", "left_ear", "center_ear", "right_ear", "tail"]
    } else if has("nose")
        && has("left_ear")
        && has("right_ear")
        && has("center_ear")
        && has("tail_base")
    {
        vec!["nose", "left_ear", "center_ear", "right_ear", "tail_base"]
    } else {
        return None;
    };

    Some(bekiper_core::inspection::FeatureInspection::Trajectory {
        label: "skeleton_pose".into(),
        coordinate_labels: coordinate_labels
            .into_iter()
            .map(ToString::to_string)
            .collect(),
        description: "Zoomed skeleton pose at representative frame".into(),
        representative_frame_index: None,
        color: Some("#00d1b2".into()),
        linewidth: Some(2.0),
        title_override: Some("Skeleton pose zoom".into()),
        story_key: Some("pose_quality".into()),
    })
}

pub(crate) fn retain_tracking_stage(
    df: &mut DataFrame,
    source: &DataFrame,
    labels: &[String],
    stage: &str,
    (factor_x, factor_y): (f64, f64),
    manifest: &mut InspectionManifest,
) -> Result<()> {
    use polars::prelude::*;
    let aligned = match (df.column("frame"), source.column("frame")) {
        (Ok(_), Ok(_)) => {
            let keys = |table: &DataFrame| -> Result<Vec<u64>> {
                let column = table.column("frame")?;
                if !column.dtype().is_integer() {
                    return Err(BekiperError::Data(
                        "tracking frame keys must be integers".into(),
                    ));
                }
                let column = column.cast(&DataType::UInt64)?;
                let keys = column
                    .u64()?
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        BekiperError::Data(
                            "tracking frame keys must be nonnegative and non-null".into(),
                        )
                    })?;
                if keys.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(BekiperError::Data(
                        "tracking frame keys must be unique and increasing".into(),
                    ));
                }
                Ok(keys)
            };
            let source_keys = keys(source)?;
            let indices = keys(df)?
                .into_iter()
                .map(|key| {
                    source_keys
                        .binary_search(&key)
                        .map(|index| index as IdxSize)
                        .map_err(|_| {
                            BekiperError::Data(format!(
                                "tracking {stage} is missing source frame {key}"
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            source.take(&IdxCa::from_vec("".into(), indices))?
        }
        (Err(_), Err(_)) if df.height() == source.height() => source.clone(),
        _ => {
            return Err(BekiperError::Data(format!(
                "tracking {stage} requires source frame keys to align cropped or keyed tables"
            )));
        }
    };
    // Raw pose has no joined clock in ordinary runs. Compare clocks only when
    // both stages retain them, after projecting onto the evaluated source keys.
    for key in ["frame_index", "timestamp", "elapsed", "elapsed_s"] {
        if let (Ok(left), Ok(right)) = (df.column(key), aligned.column(key))
            && !left.equals_missing(right)
        {
            return Err(BekiperError::Data(format!(
                "tracking {stage} {key} does not align with evaluated data"
            )));
        }
    }
    for label in labels {
        let (x, y) = (format!("{label}_x"), format!("{label}_y"));
        if aligned.column(&x).is_err() || aligned.column(&y).is_err() {
            continue;
        }
        let binding = format!("__bk_stage__{stage}__{label}");
        let (new_x, new_y) = (format!("{binding}_x"), format!("{binding}_y"));
        let projected = aligned
            .clone()
            .lazy()
            .select([
                (col(&x).cast(DataType::Float64) * lit(factor_x)).alias(&new_x),
                (col(&y).cast(DataType::Float64) * lit(factor_y)).alias(&new_y),
            ])
            .collect()?;
        for column in projected.columns() {
            df.with_column(column.clone())?;
        }
        manifest
            .coordinate_columns
            .insert(binding, CoordinateColumnSpec { x: new_x, y: new_y });
    }
    Ok(())
}

pub(crate) fn emit_tracking_inspections(labels: &[String], manifest: &mut InspectionManifest) {
    for label in labels {
        for stage in ["raw_pose", "augmented", "evaluated"] {
            let binding = if stage == "evaluated" {
                label.clone()
            } else {
                format!("__bk_stage__{stage}__{label}")
            };
            if !manifest.coordinate_columns.contains_key(&binding) {
                continue;
            }
            manifest.feature_inspections.push(
                bekiper_core::inspection::FeatureInspection::Trajectory {
                    label: format!("tracking_{stage}_{label}"),
                    coordinate_labels: vec![binding],
                    description: format!(
                        "{stage} tracking: {label}; source-row alignment; metric coordinates"
                    ),
                    representative_frame_index: Some(
                        manifest.video.total_frames.saturating_sub(1) / 2,
                    ),
                    color: Some("#24577a".into()),
                    linewidth: None,
                    title_override: Some(format!("{label} — {stage}")),
                    story_key: Some(format!("tracking_{label}")),
                },
            );
        }
        for axis in ["x", "y"] {
            for stage in ["raw_pose", "augmented", "evaluated"] {
                let binding = if stage == "evaluated" {
                    label.clone()
                } else {
                    format!("__bk_stage__{stage}__{label}")
                };
                let Some(columns) = manifest.coordinate_columns.get(&binding) else {
                    continue;
                };
                let value_column = {
                    let c = columns;
                    if axis == "x" {
                        c.x.clone()
                    } else {
                        c.y.clone()
                    }
                };
                manifest.feature_inspections.push(
                    bekiper_core::inspection::FeatureInspection::TimeSeries {
                        label: format!("tracking_trace_{stage}_{label}_{axis}"),
                        value_column,
                        unit: "m".into(),
                        description: format!("{label} {axis}(t) — {stage}; aligned source rows"),
                        representative_frame_index: Some(
                            manifest.video.total_frames.saturating_sub(1) / 2,
                        ),
                        color: Some("#24577a".into()),
                        linewidth: None,
                        title_override: Some(format!("{label} {axis}(t) — {stage}")),
                        story_key: Some(format!("tracking_timeseries_{label}")),
                    },
                );
            }
        }
    }
}
