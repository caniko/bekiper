pub fn post_analysis_output_path(run_root: &Path, run_id: &str) -> Result<PathBuf> {
    require(
        crate::config::post_analysis::valid_run_id(run_id),
        "unsafe post-analysis run_id",
    )?;
    Ok(run_root.join("data/_post_analysis").join(run_id))
}

fn apply_project_outputs(
    plan: &mut Record,
    settings: &crate::config::PostAnalysisConfig,
) -> Result<()> {
    if !plan["outputs"].is_object() {
        plan["outputs"] = json!({});
    }
    let mut requests = native(inputs::requested_output_ids(plan, &[]))?;
    requests.retain(|name| name != "figures" && name != "report" && name != "report_pdf");
    if settings.figures.enabled
        && settings
            .figures
            .selection
            .as_ref()
            .is_none_or(|names| !names.is_empty())
    {
        requests.push("figures".into());
    }
    if settings.report.enabled {
        requests.push("report".into());
    }
    // Metrics are the minimal auditable output when visual/report selections
    // suppress every previously requested export.
    if requests.is_empty() {
        requests.push("metrics".into());
    }
    plan["outputs"]["requested_outputs"] = json!(requests);
    plan["outputs"]["figures_enabled"] = settings.figures.enabled.into();
    plan["outputs"]["figure_selection"] = json!(settings.figures.selection);
    plan["outputs"]["figure_formats"] = json!(settings.figures.formats);
    plan["outputs"]["figure_dpi"] = settings.figures.dpi.into();
    plan["outputs"]["report"] = json!({"enabled":settings.report.enabled,"title":settings.report.title,"compiler":settings.report.compiler});
    Ok(())
}

fn cell_value(value: CellValue) -> Result<Record> {
    Ok(match value {
        CellValue::Null => Value::Null,
        CellValue::Bool(value) => value.into(),
        CellValue::Int(value) => value.into(),
        CellValue::Float(value) => {
            require(value.is_finite(), "nonfinite native measurement")?;
            json!(value)
        }
        CellValue::FloatList(value) => {
            require(
                value.iter().all(|value| value.is_finite()),
                "nonfinite native measurement list",
            )?;
            json!(value)
        }
        CellValue::String(value) => value.into(),
    })
}
