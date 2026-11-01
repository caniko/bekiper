pub(crate) fn validate_capabilities(
    manifest: &bekiper_core::inspection::InspectionManifest,
    decompositions: &[bekiper_core::inspection::HeuristicDecompositionSpec],
    detailed: Option<&bekiper_behaviour::TaskDetail>,
) -> Result<()> {
    use bekiper_core::inspection::FeatureInspection;
    let expected = |label: &str| -> Result<()> {
        if manifest
            .feature_inspections
            .iter()
            .any(|f| f.label() == label)
        {
            Ok(())
        } else {
            Err(data(format!(
                "inspection missing expected capability {label}"
            )))
        }
    };
    for spec in decompositions {
        if !spec.components.is_empty() {
            expected(&format!("heuristic_decomposition_{}", spec.heuristic_name))?;
        }
        for component in &spec.components {
            if component.ray_origin_label.is_some()
                && [
                    &component.ray_direction_x_column,
                    &component.ray_direction_y_column,
                    &component.ray_target_x_column,
                    &component.ray_target_y_column,
                ]
                .iter()
                .any(|c| c.is_none())
            {
                return Err(data(format!(
                    "incomplete evaluated ray binding in {}",
                    spec.heuristic_name
                )));
            }
        }
    }
    for heuristic in &manifest.heuristics {
        let value = serde_json::to_string(&manifest.feature_inspections).map_err(data)?;
        if !value.contains(&format!("\"{}\"", heuristic.result_column)) {
            return Err(data(format!(
                "inspection missing presentation for {}",
                heuristic.name
            )));
        }
    }
    match detailed {
        Some(bekiper_behaviour::TaskDetail::ObjectRecognition(detail))
            if !detail.object_labels.is_empty() =>
        {
            expected("preference_summary")?;
            if decompositions
                .iter()
                .any(|d| d.heuristic_name.starts_with("observation_"))
            {
                expected("session_composition_observation")?;
            }
        }
        Some(bekiper_behaviour::TaskDetail::OpenField(_)) => {
            if !manifest
                .feature_inspections
                .iter()
                .any(|f| matches!(f, FeatureInspection::CenterPeriphery { .. }))
            {
                return Err(data("inspection missing center/periphery capability"));
            }
        }
        Some(bekiper_behaviour::TaskDetail::YMaze(_))
            if !manifest
                .feature_inspections
                .iter()
                .any(|f| matches!(f, FeatureInspection::YMazeArmSections { .. })) =>
        {
            return Err(data("inspection missing Y-maze section capability"));
        }
        _ => {}
    }
    Ok(())
}
