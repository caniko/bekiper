//! Synthetic native distribution payloads for visual inspection; no study data.
//! Usage: cargo run -p bekiper-post-analysis --example distribution_smoke -- <fresh-directory> [native-trial.json]
use std::{error::Error, fs, path::PathBuf};

use bekiper_post_analysis::plotting::build_bundle;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(args.next().ok_or("fresh-directory required")?);
    let native_trial = args.next().map(PathBuf::from);
    if args.next().is_some() || root.exists() {
        return Err("expected one fresh output directory".into());
    }
    let config = json!({
        "run_mode":"synthetic", "synthetic_fixture":true,
        "statistics":{"summary_uncertainty":"SEM"},
        "outputs":{"figure_selection":["figure_nort_distributions"],"figure_formats":["png","svg"],"figure_dpi":90},
        "inclusion":{"population_policy":"all_nonexcluded","verification_policy":"allow_unverified"},
        "decisions":{"D09":{"status":"resolved"},"D12":{"status":"resolved"}}
    });
    let mut rows = Vec::new();
    for timepoint in ["pre", "post"] {
        for phase in ["training", "test"] {
            for animal in 0..12 {
                let ta = 2.0 + animal as f64 * 0.3;
                let tb = 3.5 + (animal % 4) as f64 * 0.7;
                let mut row = json!({
                    "animal_id":format!("synthetic-{animal:02}"),"assay":"nort",
                    "session_id":format!("{timepoint}-{phase}"),"timepoint":timepoint,"phase":phase,
                    "group":if animal < 6 {"Synthetic A"} else {"Synthetic B"},
                    "excluded":false,"qc_flags":[],"input_verification":{"status":"unresolved"},
                });
                let scores = bekiper_post_analysis::metrics::nort_scores(
                    Some(ta),
                    Some(tb),
                    Some(tb),
                    Some(ta),
                )?;
                row.as_object_mut()
                    .ok_or("row object required")?
                    .extend(scores.as_object().ok_or("scores object required")?.clone());
                rows.push(row);
            }
        }
    }
    let mut bundle = build_bundle(&rows, &config)?;
    for figure in bundle["figures"].as_array_mut().ok_or("figures required")? {
        let original = figure["summary_label"].as_str().unwrap_or("");
        figure["summary_label"] =
            format!("Synthetic drawing fixture; no experimental measurements. {original}").into();
    }
    fs::create_dir_all(&root)?;
    fs::write(root.join("rows.json"), serde_json::to_vec_pretty(&rows)?)?;
    fs::write(
        root.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    fs::write(
        root.join("plotting_bundle.json"),
        serde_json::to_vec_pretty(&bundle)?,
    )?;
    if let Some(path) = native_trial {
        let trial: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        let payload = trial["detailed"]["native_state_distributions"]
            .as_array()
            .ok_or("native state payload required")?;
        let mut state_rows = rows.clone();
        for row in &mut state_rows {
            row["native_state_distributions"] = json!(payload);
        }
        let mut state_config = config.clone();
        state_config["outputs"]["figure_selection"] = json!(["diagnostic_nort_states"]);
        let mut state_bundle = build_bundle(&state_rows, &state_config)?;
        for figure in state_bundle["figures"]
            .as_array_mut()
            .ok_or("state figures")?
        {
            let original = figure["summary_label"].as_str().unwrap_or("");
            figure["summary_label"] = format!("Synthetic presentation fixture: replicated native trial masks, not independent study observations. {original}").into();
        }
        let directory = root.join("native_states");
        fs::create_dir(&directory)?;
        fs::write(
            directory.join("rows.json"),
            serde_json::to_vec_pretty(&state_rows)?,
        )?;
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec_pretty(&state_config)?,
        )?;
        fs::write(
            directory.join("plotting_bundle.json"),
            serde_json::to_vec_pretty(&state_bundle)?,
        )?;
    }
    // Exercise the native-to-Python boundary with tied, constant, singleton
    // and outlier populations, not just the smooth NORT drawing fixture.
    let mut edge_config = config.clone();
    edge_config["outputs"]["figure_selection"] = json!(["figure_oft_distributions"]);
    let mut edge_rows = Vec::new();
    for (group, values) in [
        ("Zero-heavy", vec![0.0, 0.0, 0.0, 1.0]),
        ("Upper ties", vec![0.0, 1.0, 1.0, 1.0]),
        ("Singleton", vec![2.0]),
        ("Constant", vec![0.0; 4]),
        ("Outlier", vec![0.0, 0.0, 0.0, 0.0, 20.0]),
    ] {
        for (index, value) in values.into_iter().enumerate() {
            edge_rows.push(json!({
                "animal_id":format!("{group}-{index}"), "assay":"oft",
                "session_id":"post-session", "timepoint":"post", "phase":"session",
                "group":group, "excluded":false, "qc_flags":[],
                "input_verification":{"status":"unresolved"},
                "total_distance_m":value, "mean_speed_m_per_s":value / 300.0,
                "center_time_pct":value, "center_entries":value,
                "latency_to_first_center_entry_s":value
            }));
        }
    }
    let mut edge_bundle = build_bundle(&edge_rows, &edge_config)?;
    for figure in edge_bundle["figures"]
        .as_array_mut()
        .ok_or("figures required")?
    {
        let original = figure["summary_label"].as_str().unwrap_or("");
        figure["summary_label"] =
            format!("Synthetic regression populations; no experimental measurements. {original}")
                .into();
    }
    let edge_root = root.join("edge_cases");
    fs::create_dir(&edge_root)?;
    fs::write(
        edge_root.join("rows.json"),
        serde_json::to_vec_pretty(&edge_rows)?,
    )?;
    fs::write(
        edge_root.join("config.json"),
        serde_json::to_vec_pretty(&edge_config)?,
    )?;
    fs::write(
        edge_root.join("plotting_bundle.json"),
        serde_json::to_vec_pretty(&edge_bundle)?,
    )?;
    println!("{}", root.display());
    Ok(())
}
