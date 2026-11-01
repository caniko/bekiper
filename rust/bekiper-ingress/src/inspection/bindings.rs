/// Validate explicit declarations before persistence, including nested native
/// ray/scalar/state bindings. Renderers may not silently reconstruct them.
pub fn validate_bindings(
    manifest: &bekiper_core::inspection::InspectionManifest,
    df: &polars::prelude::DataFrame,
) -> Result<()> {
    if df.height() == 0 {
        return Err(data("inspection requires evaluated rows"));
    }
    let mut features = std::collections::BTreeSet::new();
    for feature in &manifest.feature_inspections {
        if !features.insert(feature.label()) {
            return Err(data(format!(
                "duplicate inspection feature {}",
                feature.label()
            )));
        }
        if feature
            .representative_frame_index()
            .is_some_and(|row| row >= df.height() as u64)
        {
            return Err(data(format!(
                "inspection representative row outside evaluated table: {}",
                feature.label()
            )));
        }
    }
    for heuristic in &manifest.heuristics {
        let column = df.column(&heuristic.result_column)?;
        column.bool()?;
    }
    for (label, coordinate) in &manifest.coordinate_columns {
        for column in [&coordinate.x, &coordinate.y] {
            if df.column(column).is_err() {
                return Err(data(format!(
                    "inspection coordinate {label} references missing column {column}"
                )));
            }
        }
    }
    fn walk(
        value: &Value,
        manifest: &bekiper_core::inspection::InspectionManifest,
        df: &polars::prelude::DataFrame,
    ) -> Result<()> {
        match value {
            Value::Array(values) => {
                for v in values {
                    walk(v, manifest, df)?;
                }
            }
            Value::Object(fields) => {
                for (key, value) in fields {
                    let bindings: Vec<_> = if key.ends_with("_column") {
                        value.as_str().into_iter().collect()
                    } else if key.ends_with("_columns") {
                        value
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(Value::as_str)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    for column in bindings {
                        if df.column(column).is_err() {
                            return Err(data(format!(
                                "inspection {key} references missing column {column}"
                            )));
                        }
                    }
                    let labels: Vec<_> = if key == "coordinate_labels" {
                        value
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(Value::as_str)
                            .collect()
                    } else if [
                        "origin_label",
                        "ray_origin_label",
                        "from_coordinate_label",
                        "to_coordinate_label",
                    ]
                    .contains(&key.as_str())
                    {
                        value.as_str().into_iter().collect()
                    } else {
                        Vec::new()
                    };
                    for label in labels {
                        if !manifest.coordinate_columns.contains_key(label) {
                            return Err(data(format!(
                                "inspection references missing coordinate {label}"
                            )));
                        }
                    }
                    if key.ends_with("perimeter_label")
                        && let Some(label) = value.as_str()
                    {
                        if let Some(geometry) =
                            fields.get("evaluated_perimeter").filter(|p| !p.is_null())
                        {
                            if geometry["label"] != label {
                                return Err(data(
                                    "evaluated perimeter label does not match binding",
                                ));
                            }
                        } else if !manifest.perimeters.iter().any(|p| p.label() == label) {
                            return Err(data(format!(
                                "inspection references missing perimeter {label}"
                            )));
                        }
                    }
                    if key == "quadrant_label"
                        && let Some(label) = value.as_str()
                        && !manifest.quadrants.iter().any(|q| q.label == label)
                    {
                        return Err(data(format!(
                            "inspection references missing quadrant {label}"
                        )));
                    }
                    walk(value, manifest, df)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    walk(
        &serde_json::to_value(&manifest.feature_inspections).map_err(data)?,
        manifest,
        df,
    )
}
