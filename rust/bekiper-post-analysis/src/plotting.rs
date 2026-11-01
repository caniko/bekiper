//! Rust-owned plotting populations and statistical quantities. Python consumes
//! this versioned bundle as drawing instructions, not as an analysis dataset.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::contracts::{
    KEY, PLOTTING_SCHEMA_VERSION, Record, nort_primary_fields, oft_primary_fields,
    ymaze_score_fields,
};
use crate::inputs::{SubgroupPlan, choice, validated_subgroup_plan};
use crate::statistics::pairing;
mod descriptive;
mod distributions;
mod states;
pub use descriptive::build_descriptive_bundle;

const STATIC_FIGURES: [&str; 7] = [
    "figure_oft",
    "figure_ymaze",
    "figure_nort",
    "supplement_ymaze",
    "supplement_nort",
    "qc_ymaze",
    "qc_nort",
];

fn known_figures(subgroup: Option<&SubgroupPlan<'_>>) -> BTreeSet<String> {
    let mut known = STATIC_FIGURES
        .iter()
        .map(|name| (*name).into())
        .collect::<BTreeSet<String>>();
    known.extend(
        STATIC_FIGURES
            .iter()
            .filter(|name| !name.starts_with("qc_"))
            .map(|name| format!("{name}_distributions")),
    );
    known.insert("diagnostic_nort_states".into());
    if let Some(plan) = subgroup {
        for (index, stratum) in plan.strata.iter().enumerate() {
            for assay in &stratum.assays {
                known.insert(subgroup_name(assay, stratum.field, index));
            }
        }
    }
    known
}

/// Disabled output retains the registered figure inventory without resolving
/// plotting populations or summary policies that are unnecessary for this run.
pub fn disabled_bundle(config: &Record) -> Result<Record> {
    let reason = if config["outputs"]["figures_enabled"] == false {
        "figures_disabled"
    } else if config["outputs"]["figure_selection"]
        .as_array()
        .is_some_and(Vec::is_empty)
    {
        "empty_figure_selection"
    } else {
        "figures_not_requested"
    };
    let subgroup = validated_subgroup_plan(config)?;
    let omitted = known_figures(subgroup.as_ref())
        .into_iter()
        .map(|name| json!({"name":name,"status":"disabled","reason":reason}))
        .collect::<Vec<_>>();
    Ok(
        json!({"schema_version":PLOTTING_SCHEMA_VERSION,"figures":[],"plotting_data":[],"summaries":[],"unrendered_figures":omitted}),
    )
}

struct PlotPolicy<'a> {
    verification: Option<&'a str>,
    uncertainty: Option<&'a str>,
    primary_threshold: Option<u64>,
    population: &'a str,
    formats: Vec<String>,
    dpi: u32,
}

/// All eligibility, pairing, animal counts, mean/SEM/CI quantities, threshold
/// selection and subgroup membership are fixed before crossing into Python.
pub fn build_bundle(rows: &[Record], config: &Record) -> Result<Record> {
    let verification = choice(config, "inclusion.upstream_verification_requirement", "D12")
        .map(|value| {
            value
                .as_str()
                .context("verification requirement must be a string")
        })
        .transpose()?;
    ensure!(
        matches!(
            verification,
            None | Some("allow_unverified" | "require_verified")
        ),
        "Unknown upstream verification requirement"
    );
    let uncertainty = choice(config, "statistics.summary_uncertainty", "D09")
        .map(|value| {
            value
                .as_str()
                .context("summary uncertainty must be a string")
        })
        .transpose()?;
    ensure!(
        matches!(uncertainty, None | Some("SEM" | "95% CI")),
        "Summary uncertainty must be registered SEM or 95% CI"
    );
    let primary_threshold = choice(
        config,
        "assays.ymaze.primary_threshold_arm_length_pct",
        "D03",
    )
    .map(|value| {
        value
            .as_f64()
            .filter(|value| {
                crate::inputs::allowed_thresholds(config)
                    .iter()
                    .any(|t| *t as f64 == *value)
            })
            .map(|value| value as u64)
            .context("primary Y-maze threshold must be 25, 45 or 65 (cross-line also 0)")
    })
    .transpose()?;
    let mut formats = strings_or_default(
        &config["outputs"]["figure_formats"],
        &["svg"],
        "figure_formats",
    )?;
    ensure!(
        !formats.is_empty()
            && formats
                .iter()
                .all(|format| matches!(format.as_str(), "svg" | "pdf" | "png")),
        "Unsupported/empty figure formats"
    );
    unique(&formats, "figure formats")?;
    let dpi = if config["outputs"]["figure_dpi"].is_null() {
        150
    } else {
        config["outputs"]["figure_dpi"]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0)
            .context("figure_dpi must be a positive u32")?
    };
    if config["outputs"]["report"]["enabled"] == true
        && !formats.iter().any(|format| format == "pdf")
    {
        formats.push("pdf".into());
    }
    let policy = PlotPolicy {
        verification,
        uncertainty,
        primary_threshold,
        population: verification.unwrap_or("descriptive_metadata_nonexcluded"),
        formats,
        dpi,
    };
    let thresholds = thresholds(config)?;
    let subgroup = validated_subgroup_plan(config)?;
    let known = known_figures(subgroup.as_ref());
    let selection = if config["outputs"]["figure_selection"].is_null() {
        None
    } else {
        Some(strings_or_default(
            &config["outputs"]["figure_selection"],
            &[],
            "figure_selection",
        )?)
    };
    if let Some(selected) = &selection {
        unique(selected, "figure selection")?;
        ensure!(
            selected.iter().all(|name| known.contains(name)),
            "Unknown figure_selection identifier"
        );
    }
    let requested = |name: &str, applicable: bool| {
        selection.as_ref().map_or(applicable, |selection| {
            selection.iter().any(|chosen| chosen == name)
        })
    };
    let selected = |name: &str, applicable: bool| {
        requested(name, applicable) || requested(&format!("{name}_distributions"), false)
    };
    let mut figures = Vec::new();
    let mut unrendered = Vec::new();
    if requested(
        "diagnostic_nort_states",
        rows.iter()
            .any(|r| r.get("native_state_distributions").is_some()),
    ) {
        let panels = states::panels(rows, &policy)?;
        if !panels.is_empty() {
            for (index, chunk) in panels.chunks(6).enumerate() {
                let name = if index == 0 {
                    "diagnostic_nort_states".into()
                } else {
                    format!("diagnostic_nort_states_page_{index}")
                };
                let mut page = figure(&name, chunk.to_vec(), &policy);
                if index > 0 {
                    page["parent_figure"] = "diagnostic_nort_states".into();
                }
                page["summary_label"] = "Descriptive whole-session state fractions from native evaluated masks; missing, unassigned and overlapping states retained. Each point is an individual animal in the recorded pre/post and training/test partition. These are not conditional object-preference scores. KDE tails are descriptive smoothing, not observations beyond the measured range.".into();
                figures.push(page);
            }
        }
    }
    type AssayPanels = fn(&[&Record], &PlotPolicy<'_>) -> Result<Vec<Record>>;
    for (assay, build) in [
        ("oft", oft_panels as AssayPanels),
        ("ymaze", ymaze_panels as AssayPanels),
        ("nort", nort_panels as AssayPanels),
    ] {
        let source: Vec<_> = rows.iter().filter(|row| row["assay"] == assay).collect();
        let main_name = format!("figure_{assay}");
        if selected(&main_name, !source.is_empty()) {
            let panels = build(&source, &policy)?;
            figures.push(figure(&main_name, panels, &policy));
        }
        let supplement_name = format!("supplement_{assay}");
        if assay == "ymaze"
            && selected(
                &supplement_name,
                !source.is_empty() && !thresholds.is_empty(),
            )
        {
            if thresholds.is_empty() {
                unrendered.push(json!({"name":supplement_name,"status":"disabled","reason":"empty_sensitivity_selection"}));
            } else {
                let mut panels = Vec::new();
                for threshold in &thresholds {
                    let members: Vec<_> = source
                        .iter()
                        .copied()
                        .filter(|row| {
                            row["timepoint"] == "post"
                                && row["threshold_pct"].as_f64() == Some(*threshold as f64)
                        })
                        .collect();
                    panels.push(categorical_panel(
                        &members,
                        "spontaneous_alternation_pct",
                        &format!("Sensitivity SA — {threshold}% imen radial vertex span"),
                        None,
                        &policy,
                    )?);
                    panels.push(categorical_panel(
                        &members,
                        "spontaneous_alternation_pct_standard",
                        &format!("Standard SA — {threshold}% (secondary)"),
                        None,
                        &policy,
                    )?);
                }
                figures.push(figure(&supplement_name, panels, &policy));
            }
        }
        if assay == "nort" && selected(&supplement_name, !source.is_empty()) {
            let training = phase_post(&source, "training");
            let test = phase_post(&source, "test");
            let mut panels = vec![
                categorical_panel(&training, "object_bias", "Object bias", Some(0.0), &policy)?,
                categorical_panel(
                    &training,
                    "side_bias",
                    "Source algebraic side_bias (not spatial)",
                    Some(0.0),
                    &policy,
                )?,
            ];
            for endpoint in [
                "valid_bouts",
                "removed_duration_bouts",
                "removed_speed_bouts",
                "removed_angle_bouts",
            ] {
                panels.push(categorical_panel(
                    &test,
                    endpoint,
                    &endpoint.replace('_', " "),
                    None,
                    &policy,
                )?);
            }
            figures.push(figure(&supplement_name, panels, &policy));
        }
    }
    if let Some(plan) = &subgroup {
        for (index, stratum) in plan.strata.iter().enumerate() {
            let field = stratum.field;
            for assay in &stratum.assays {
                let assay = assay.as_str();
                let name = subgroup_name(assay, field, index);
                if !selected(&name, true) {
                    continue;
                }
                let mut panels = Vec::new();
                for level in &stratum.levels {
                    let members: Vec<_> = rows
                        .iter()
                        .filter(|row| {
                            primary_post_member(row, assay, &policy) && row[field] == *level
                        })
                        .collect();
                    let endpoints: &[&str] = match assay {
                        "oft" => oft_primary_fields(),
                        "ymaze" => &ymaze_score_fields()[..1],
                        "nort" => nort_primary_fields(),
                        _ => bail!("Unknown subgroup assay"), // coverage: unreachable — validated_subgroup_plan admits only oft/ymaze/nort
                    };
                    for endpoint in endpoints {
                        let label = format!("{endpoint} — {field}={level}");
                        let mut panel = categorical_panel(
                            &members,
                            endpoint,
                            &label,
                            (*endpoint == "DI").then_some(0.0),
                            &policy,
                        )?;
                        panel["stratum"] = json!({"field":field,"level":level});
                        if assay == "ymaze" && policy.primary_threshold.is_none() {
                            panel["reason"] = json!("blocked:D03:primary_threshold");
                        }
                        panels.push(panel);
                    }
                }
                figures.push(figure(&name, panels, &policy));
            }
        }
    }
    for (assay, x, ys) in [
        (
            "ymaze",
            "spontaneous_alternation_pct",
            &[
                "valid_total_arm_entries",
                "total_distance_m",
                "mean_speed_m_per_s",
                "neutral_triads_pct",
            ][..],
        ),
        ("nort", "total_test_exploration", &["DI"][..]),
    ] {
        let name = format!("qc_{assay}");
        if !selected(&name, rows.iter().any(|row| row["assay"] == assay)) {
            continue;
        }
        let members: Vec<_> = rows
            .iter()
            .filter(|row| primary_post_member(row, assay, &policy))
            .collect();
        let panels = ys
            .iter()
            .map(|y| scatter_panel(&members, x, y, &policy))
            .collect::<Result<Vec<_>>>()?;
        figures.push(figure(&name, panels, &policy));
    }
    let mut distribution_figures = Vec::new();
    for base in &figures {
        let base_name = base["name"].as_str().context("figure name")?;
        let name = format!("{base_name}_distributions");
        if !known.contains(&name) || !requested(&name, true) {
            continue;
        }
        let mut panels = Vec::new();
        let session_panels;
        let sources = if base_name == "figure_nort" {
            session_panels = nort_distribution_panels(rows, &policy)?;
            &session_panels
        } else {
            base["panels"].as_array().context("figure panels")?
        };
        for (source_index, source) in sources.iter().enumerate() {
            if source["kind"] != "categorical" {
                continue; // coverage: unreachable — distribution families derive only from categorical panels
            }
            let mut panel = source.clone();
            distributions::attach(&mut panel)?;
            for style in ["violin", "boxplot", "histogram"] {
                let mut copy = panel.clone();
                copy["style"] = style.into();
                copy["native_source_panel_index"] = source_index.into();
                copy["label"] =
                    format!("{} — {style}", panel["label"].as_str().unwrap_or("")).into();
                panels.push(copy);
            }
        }
        if !panels.is_empty() {
            let mut f = figure(&name, panels, &policy);
            f["summary_label"] = "Native Gaussian KDE / linear-interpolation quartiles / shared 12-bin histograms. Individual animals retained; repeated sessions and phases remain separate. Constant/singleton populations have no density curve. KDE tails are descriptive smoothing, not observations beyond the measured range.".into();
            // Endpoint pages are alternate layouts of these exact native panels.
            // Keep the historical overview and export readable three-panel pages.
            let mut pages: BTreeMap<String, Vec<Record>> = BTreeMap::new();
            for panel in f["panels"].as_array().context("distribution panels")? {
                let partition = &panel["session_partition"];
                let mut key = [
                    partition["timepoint"].as_str(),
                    partition["phase"].as_str(),
                    panel["endpoint"].as_str(),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("_");
                if partition.is_null() {
                    key.push_str(&format!(
                        "_panel_{}",
                        panel["native_source_panel_index"]
                            .as_u64()
                            .context("native source panel index")?
                    ));
                }
                let key = key.to_ascii_lowercase();
                pages.entry(key).or_default().push(panel.clone());
            }
            for (key, panels) in pages {
                let mut page = figure(&format!("{name}_{key}"), panels, &policy);
                page["parent_figure"] = name.clone().into();
                page["summary_label"] = f["summary_label"].clone();
                distribution_figures.push(page);
            }
            distribution_figures.push(f);
        }
    }
    figures.retain(|f| {
        requested(
            f["parent_figure"]
                .as_str()
                .or(f["name"].as_str())
                .unwrap_or(""),
            true,
        )
    });
    figures.extend(distribution_figures);
    for name in known {
        if figures.iter().any(|figure| figure["name"] == name)
            || unrendered.iter().any(|figure| figure["name"] == name)
        {
            continue;
        }
        let (status, reason) = if selection
            .as_ref()
            .is_some_and(|names| !names.contains(&name))
        {
            ("not_selected", "figure_selection")
        } else if name == "supplement_ymaze" && thresholds.is_empty() {
            ("disabled", "empty_sensitivity_selection")
        } else {
            ("unavailable", "no_assay_measurements")
        };
        unrendered.push(json!({"name":name,"status":status,"reason":reason}));
    }
    let mut plotting_data = Vec::new();
    let mut summaries = Vec::new();
    for figure in &mut figures {
        let name = figure["name"].clone();
        for (index, panel) in figure["panels"]
            .as_array_mut()
            .context("figure panels")?
            .iter_mut()
            .enumerate()
        {
            let panel_id = format!("panel_{index}");
            panel["panel_id"] = panel_id.clone().into();
            for (field, destination, extra_fields) in [
                (
                    "points",
                    &mut plotting_data,
                    &["label", "reference_line", "x_endpoint"][..],
                ),
                ("summaries", &mut summaries, &["label"][..]),
            ] {
                for record in panel[field]
                    .as_array()
                    .with_context(|| format!("panel {field}"))?
                {
                    let mut record = record.clone();
                    record["figure"] = name.clone();
                    record["panel"] = panel["endpoint"].clone();
                    record["panel_id"] = panel_id.clone().into();
                    for &field in extra_fields {
                        record[field] = panel[field].clone();
                    }
                    destination.push(record);
                }
            }
        }
    }
    Ok(
        json!({"schema_version":PLOTTING_SCHEMA_VERSION,"figures":figures,"plotting_data":plotting_data,"summaries":summaries,
        "population_policy":policy.population,"summary_uncertainty":policy.uncertainty,"unrendered_figures":unrendered}),
    )
}

// Shared only by the primary post-treatment QC and subgroup populations.
// Training, paired pre/post and sensitivity panels have distinct membership.
fn primary_post_member(row: &Record, assay: &str, policy: &PlotPolicy<'_>) -> bool {
    row["assay"] == assay
        && row["timepoint"] == "post"
        && (assay != "nort" || row["phase"] == "test")
        && (assay != "ymaze" || is_primary(row, policy))
}

mod assays;
use assays::{nort_distribution_panels, nort_panels, oft_panels, ymaze_panels};

mod categorical;
use categorical::{categorical_panel, finish_categorical, object_panel};

mod summary;
use summary::summary;

mod scatter;
use scatter::scatter_panel;

mod points;
use points::point_for;

mod figure;
use figure::{endpoint_label, figure};

mod values;
use values::{
    compact_pairing_row, is_primary, missing_reason, number, phase_post, string,
    strings_or_default, subgroup_name, thresholds, unique,
};
