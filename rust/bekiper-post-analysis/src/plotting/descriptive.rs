//! Presentation of recorded native endpoints. No registered decisions, scoring,
//! pairing or inference are resolved along this path.
use super::*;

pub fn build_descriptive_bundle(rows: &[Record], config: &Record) -> Result<Record> {
    // Preserve exact per-recording evidence once. Endpoint/style projections
    // carry the complete analysis identity as a reference, rather than cloning
    // native dwell bouts and geometry into every view and eligibility record.
    let mut provenance = serde_json::Map::new();
    let mut projected = Vec::with_capacity(rows.len());
    for row in rows {
        let identity = string(row, "analysis_id")?;
        ensure!(
            provenance
                .insert(identity.into(), row["scoring_provenance"].clone())
                .is_none(),
            "duplicate descriptive analysis identity"
        );
        let mut view = row.clone();
        view.as_object_mut()
            .context("descriptive row must be an object")?
            .remove("scoring_provenance");
        view["scoring_provenance_ref"] = identity.into();
        projected.push(view);
    }
    let rows = projected.as_slice();
    let formats = strings_or_default(
        &config["outputs"]["figure_formats"],
        &["png"],
        "figure_formats",
    )?;
    ensure!(
        !formats.is_empty()
            && formats
                .iter()
                .all(|f| ["png", "pdf", "svg"].contains(&f.as_str())),
        "unsupported descriptive figure format"
    );
    let dpi = config["outputs"]["figure_dpi"].as_u64().unwrap_or(150);
    let policy = PlotPolicy {
        verification: None,
        uncertainty: None,
        primary_threshold: None,
        population: "recorded_descriptive",
        formats,
        dpi: u32::try_from(dpi)?,
    };
    ensure!(policy.dpi > 0, "positive figure DPI required");
    let selection = config["outputs"]["figure_selection"].as_array();
    let enabled = config["outputs"]["figures_enabled"] != false;
    let selected = |name: &str| enabled && selection.is_none_or(|s| s.iter().any(|n| n == name));
    if let Some(selection) = selection {
        for name in selection {
            ensure!(
                STATIC_FIGURES.iter().any(|base| name == *base
                    || name.as_str() == Some(format!("{base}_distributions").as_str()))
                    || name == "diagnostic_nort_states",
                "unknown descriptive figure selection: {name}"
            );
        }
    }
    let mut figures = Vec::new();
    let mut omitted = Vec::new();
    let mut eligibility = Vec::new();
    if let Some(selection) = selection {
        for name in selection {
            ensure!(
                name.as_str()
                    .is_some_and(|name| name.starts_with("figure_"))
                    || name == "diagnostic_nort_states",
                "selected figure {name} requires a registered plan"
            );
        }
    }
    for assay in crate::contracts::assays_in_report_order().iter().copied() {
        let mut partitions: BTreeMap<String, Vec<Record>> = BTreeMap::new();
        for row in rows.iter().filter(|r| r["assay"] == assay) {
            partitions
                .entry(row["presentation_partition"].to_string())
                .or_default()
                .push(row.clone());
        }
        let base = format!("figure_{assay}");
        let distributions_name = format!("{base}_distributions");
        for (partition_index, (partition, members)) in partitions.iter().enumerate() {
            let endpoints = members
                .iter()
                .flat_map(|r| r["scalar_endpoints"].as_array().into_iter().flatten())
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>();
            let sources = members.iter().collect::<Vec<_>>();
            for (endpoint_index, endpoint) in endpoints.into_iter().enumerate() {
                let mut panel = categorical_panel(
                    &sources,
                    endpoint,
                    &format!(
                        "{} · {}",
                        members[0]["presentation_label"].as_str().unwrap_or(assay),
                        endpoint_label(endpoint)
                    ),
                    None,
                    &policy,
                )?;
                panel["session_partition"] = members[0]["presentation_partition"].clone();
                panel["partition_key"] = partition.clone().into();
                distributions::attach(&mut panel)?;
                eligibility.extend(
                    panel["eligibility_ledger"]
                        .as_array()
                        .context("descriptive ledger")?
                        .iter()
                        .cloned(),
                );
                for (family, styles) in [
                    (&base, &["points"][..]),
                    (&distributions_name, &["violin", "boxplot", "histogram"][..]),
                ] {
                    if !selected(family) {
                        continue;
                    }
                    let panels = styles
                        .iter()
                        .map(|style| {
                            let mut copy = panel.clone();
                            if *style != "points" {
                                copy["style"] = (*style).into();
                            }
                            copy
                        })
                        .collect();
                    let mut page = figure(
                        &format!("{family}_partition_{partition_index}_endpoint_{endpoint_index}"),
                        panels,
                        &policy,
                    );
                    page["parent_figure"] = family.clone().into();
                    page["summary_label"] = "Recorded descriptive measurements; sessions, variants and scoring policies partitioned. Individual observations and descriptive means/distributions; no inferential tests or uncertainty interval selected. Repeated recordings remain individual observations; n_animals counts distinct animals. KDE tails are smoothing, not observed measurements.".into();
                    figures.push(page);
                }
            }
        }
        for name in [&base, &distributions_name] {
            if !figures.iter().any(|f| f["parent_figure"] == *name) {
                omitted.push(json!({"name":name,"status":if selected(name) {"unavailable"} else {"not_selected"},"reason":if !selected(name) {"figure_selection"} else if partitions.is_empty() {"no_recorded_measurements"} else {"no_recorded_scalar_endpoints"}}));
            }
        }
    }
    if selected("diagnostic_nort_states") {
        let mut partitions: BTreeMap<String, Vec<Record>> = BTreeMap::new();
        let mut state_eligibility = BTreeSet::new();
        for row in rows.iter().filter(|r| r["assay"] == "nort") {
            partitions
                .entry(row["presentation_partition"].to_string())
                .or_default()
                .push(row.clone());
        }
        for (partition_index, members) in partitions.values().enumerate() {
            for (index, panels) in states::panels(members, &policy)?.chunks(6).enumerate() {
                for panel in panels {
                    // Violin, box and histogram artists share one measurement.
                    for entry in panel["eligibility_ledger"]
                        .as_array()
                        .context("state ledger")?
                    {
                        if state_eligibility.insert(entry.to_string()) {
                            eligibility.push(entry.clone());
                        }
                    }
                }
                let mut page = figure(
                    &format!("diagnostic_nort_states_partition_{partition_index}_page_{index}"),
                    panels.to_vec(),
                    &policy,
                );
                page["parent_figure"] = "diagnostic_nort_states".into();
                page["summary_label"] = format!("{} · Recorded native whole-session fractions; Missing, Unassigned and overlap retained; no rescoring or inferential uncertainty.", members[0]["presentation_label"].as_str().unwrap_or("NORT")).into();
                figures.push(page);
            }
        }
    }
    if !figures
        .iter()
        .any(|f| f["parent_figure"] == "diagnostic_nort_states")
    {
        omitted.push(json!({"name":"diagnostic_nort_states","status":if selected("diagnostic_nort_states") {"unavailable"} else {"not_selected"},
            "reason":if selected("diagnostic_nort_states") {"native_state_not_recorded"} else {"figure_selection"}}));
    }
    for name in ["supplement_ymaze", "supplement_nort", "qc_ymaze", "qc_nort"] {
        omitted
            .push(json!({"name":name,"status":"not_selected","reason":"requires_registered_plan"}));
    }
    let mut plotting_data = Vec::new();
    let mut summaries = Vec::new();
    for f in &mut figures {
        let name = f["name"].clone();
        for (i, p) in f["panels"]
            .as_array_mut()
            .context("descriptive panels")?
            .iter_mut()
            .enumerate()
        {
            p["panel_id"] = format!("panel_{i}").into();
            for (field, destination) in [
                ("points", &mut plotting_data),
                ("summaries", &mut summaries),
            ] {
                for value in p[field].as_array().context("descriptive panel values")? {
                    let mut value = value.clone();
                    value["figure"] = name.clone();
                    value["panel"] = p["endpoint"].clone();
                    value["panel_id"] = p["panel_id"].clone();
                    value["label"] = p["label"].clone();
                    destination.push(value);
                }
            }
        }
    }
    Ok(
        json!({"schema_version":PLOTTING_SCHEMA_VERSION,"analysis_mode":"recorded_descriptive",
        "scoring_provenance_by_analysis":provenance,
        "figures":figures,"plotting_data":plotting_data,"summaries":summaries,"unrendered_figures":omitted,
        "eligibility_ledger":eligibility,
        "population_policy":policy.population,"summary_uncertainty":null}),
    )
}
