//! Deterministic synthetic inspection inputs through the direct-source engine.
//! Usage: cargo run -p bekiper-ingress --example inspection_diagnostics_smoke -- <fresh-directory>
use std::{error::Error, fs, path::PathBuf};

use bekiper_core::{inspection::PerimeterSpec, scale::ScaleConfig};
use bekiper_ingress::{
    analysis::analyze_augmented_pose, config::ExperimentConfig, export::export_inspection,
};
use image::{Rgb, RgbImage};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(args.next().ok_or("fresh-directory argument required")?);
    if args.next().is_some() || root.exists() {
        return Err("expected one fresh output directory".into());
    }
    fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    let mut experiment: ExperimentConfig = toml::from_str(
        r#"
        name = "synthetic_diagnostics"
        task_type = "object_recognition"
        fps = 10.0
        meters_per_pixel = 0.001
        [task_params.exploration]
        window_mode = "configured_full_trial"
        speed_filter_mode = "disabled"
        min_bout_seconds = 0.5
        max_gap_seconds = 1.0
        [task_params.exploration.merge_rules.observation]
        any_of = ["olfaction", "whisker_interaction"]
        exclude = ["body_proximity"]
    "#,
    )?;
    experiment.video_resolution = Some((400, 240));
    experiment.frame_image = Some(root.join("frame.png"));
    experiment.video = Some(root.join("video.mp4"));
    let geometry = vec![
        PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 200.0,
            center_y: 120.0,
            width: 380.0,
            height: 220.0,
        },
        PerimeterSpec::Circle {
            label: "object_1".into(),
            center_x: 110.0,
            center_y: 120.0,
            radius: 18.0,
        },
        PerimeterSpec::Circle {
            label: "object_2".into(),
            center_x: 290.0,
            center_y: 120.0,
            radius: 18.0,
        },
        PerimeterSpec::Polygon {
            label: "center_zone".into(),
            vertices: vec![(170.0, 85.0), (230.0, 85.0), (225.0, 155.0), (170.0, 150.0)],
        },
    ];
    let (_, scale) = ScaleConfig::Manual(0.001).resolve_with_info(&geometry)?;
    let n = 240;
    let mut columns = vec![
        Column::new("frame".into(), (0..n as u64).collect::<Vec<_>>()),
        Column::new(
            "elapsed_s".into(),
            (0..n).map(|i| i as f64 / 10.0).collect::<Vec<_>>(),
        ),
    ];
    let mut positions = Vec::new();
    for label in [
        "nose",
        "center_ear",
        "left_ear",
        "right_ear",
        "tail_base",
        "mouse_center",
    ] {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..n {
            let phase = i as f64 * 0.09;
            let cx = if i < n / 2 { 0.11 } else { 0.29 };
            let nose = (cx + 0.043 * phase.cos(), 0.12 + 0.043 * phase.sin());
            let heading = phase
                + if i % 60 < 40 {
                    std::f64::consts::PI
                } else {
                    0.0
                };
            let (dx, dy) = (heading.cos(), heading.sin());
            let (back, side) = match label {
                "nose" => (0.0, 0.0),
                "center_ear" => (0.018, 0.0),
                "left_ear" => (0.018, 0.009),
                "right_ear" => (0.018, -0.009),
                "mouse_center" => (0.048, 0.0),
                _ => (0.085, 0.0),
            };
            xs.push(nose.0 - back * dx - side * dy);
            ys.push(nose.1 - back * dy + side * dx);
            if label == "nose" {
                positions.push(nose);
            }
        }
        columns.push(Column::new(format!("{label}_x").into(), xs));
        columns.push(Column::new(format!("{label}_y").into(), ys));
    }
    let mut augmented = DataFrame::new(n, columns)?;
    for (prefix, end, start) in [
        ("snout", "nose", "center_ear"),
        ("left_ear", "left_ear", "center_ear"),
        ("right_ear", "right_ear", "center_ear"),
    ] {
        augmented = augmented
            .lazy()
            .with_columns([
                (col(format!("{end}_x")) - col(format!("{start}_x"))).alias(format!("{prefix}_dx")),
                (col(format!("{end}_y")) - col(format!("{start}_y"))).alias(format!("{prefix}_dy")),
            ])
            .collect()?;
    }
    let raw = augmented
        .clone()
        .lazy()
        .with_columns(
            [
                "nose",
                "center_ear",
                "left_ear",
                "right_ear",
                "tail_base",
                "mouse_center",
            ]
            .iter()
            .flat_map(|label| {
                ["x", "y"].map(|axis| {
                    let name = format!("{label}_{axis}");
                    (col(&name) / lit(0.001)).alias(name)
                })
            })
            .collect::<Vec<_>>(),
        )
        .collect()?;
    let analysis = analyze_augmented_pose(&experiment, None, &geometry, &scale, raw, augmented)?;
    // Round-trip exact ray/buffer geometry before Python sees this fixture.
    let text = toml::to_string_pretty(&analysis.manifest)?;
    let decoded: bekiper_core::inspection::InspectionManifest = toml::from_str(&text)?;
    assert_eq!(
        analysis.manifest.feature_inspections.len(),
        decoded.feature_inspections.len()
    );
    for (original, roundtrip) in analysis
        .manifest
        .feature_inspections
        .iter()
        .zip(&decoded.feature_inspections)
    {
        assert_eq!(
            serde_json::to_value(original)?,
            serde_json::to_value(roundtrip)?,
            "metadata changed in {}",
            original.label()
        );
    }
    for expected in [
        "recipe_olfaction_object_1",
        "recipe_body_proximity_object_1",
        "multiobject_olfaction",
        "cumulative_observation_object_1",
        "heuristic_decomposition_fusion_comparison_observation_object_1",
        "tracking_raw_pose_nose",
        "timeseries_nose_speed",
        "sensory_disagreement_olfaction_object_1_nose",
        "attention_state_distribution_olfaction_object_1_nose",
        "session_composition_observation",
        "tracking_trace_raw_pose_nose_x",
        "geometry_vectors_olfaction_object_1_ray_hits_object_0",
    ] {
        assert!(
            decoded
                .feature_inspections
                .iter()
                .any(|f| f.label() == expected),
            "missing {expected}"
        );
    }
    // Exercise production bindings, rather than checking declarations alone.
    let mut bound_rays = 0;
    let mut bound_buffers = 0;
    let evaluated = analysis
        .result
        .evaluated_df
        .as_ref()
        .ok_or("evaluated frame table required")?;
    for feature in &decoded.feature_inspections {
        if let bekiper_core::inspection::FeatureInspection::HeuristicDecomposition {
            components,
            result_column,
            ..
        } = feature
        {
            assert!(
                evaluated.column(result_column).is_ok(),
                "missing final binding {result_column}"
            );
            for component in components {
                assert!(evaluated.column(&component.result_column).is_ok());
                for column in [
                    &component.ray_direction_x_column,
                    &component.ray_direction_y_column,
                    &component.ray_target_x_column,
                    &component.ray_target_y_column,
                    &component.value_column,
                ]
                .into_iter()
                .flatten()
                {
                    assert!(
                        evaluated.column(column).is_ok(),
                        "missing diagnostic {column}"
                    );
                }
                if let Some(origin) = &component.ray_origin_label {
                    assert!(decoded.coordinate_columns.contains_key(origin));
                    assert!(component.evaluated_perimeter.is_some());
                    bound_rays += 1;
                }
                if component.buffer_meters.is_some() {
                    assert!(component.buffer_perimeter.is_some());
                    bound_buffers += 1;
                }
            }
        }
    }
    assert!(bound_rays >= 6 && bound_buffers >= 6);
    for feature in &decoded.feature_inspections {
        if let bekiper_core::inspection::FeatureInspection::StateDistribution {
            source_columns,
            seconds,
            percentages,
            frame_counts,
            denominator_seconds,
            denominator_frames,
            category_labels,
            ..
        } = feature
        {
            assert!(source_columns.iter().all(|c| evaluated.column(c).is_ok()));
            assert_eq!(frame_counts.iter().sum::<u64>(), *denominator_frames);
            assert!((seconds.iter().sum::<f64>() - denominator_seconds).abs() < 1e-8);
            assert!((percentages.iter().sum::<f64>() - 100.0).abs() < 1e-8);
            assert!(!category_labels.iter().any(|c| c.contains("expanded")));
        }
    }
    fs::write(
        root.join("trial.json"),
        serde_json::to_vec_pretty(&analysis.trial)?,
    )?;
    export_inspection(
        &root,
        &experiment.name,
        &analysis.result,
        &analysis.manifest,
    )?;
    fs::create_dir(root.join("frames"))?;
    for (i, (x, y)) in positions.iter().enumerate() {
        let mut frame = RgbImage::from_pixel(400, 240, Rgb([210, 210, 210]));
        for ox in [110_i32, 290] {
            for px in ox - 18..=ox + 18 {
                for py in 102..=138 {
                    if (px - ox).pow(2) + (py - 120_i32).pow(2) <= 18 * 18 {
                        frame.put_pixel(px as u32, py as u32, Rgb([75, 75, 75]));
                    }
                }
            }
        }
        for px in (*x / 0.001) as i32 - 3..=(*x / 0.001) as i32 + 3 {
            for py in (*y / 0.001) as i32 - 3..=(*y / 0.001) as i32 + 3 {
                frame.put_pixel(px as u32, py as u32, Rgb([35, 35, 35]));
            }
        }
        frame.save(root.join("frames").join(format!("frame_{i:06}.png")))?;
        if i == 0 {
            frame.save(root.join("frame.png"))?;
        }
    }
    fs::write(
        root.join("README.txt"),
        "Synthetic drawing fixture: 240 frames, 10 fps, 400×240 pixels, 0.001 m/pixel. No experimental measurements. Native masks, exact eroded/buffer geometry and four fusion/tolerance states; raw pixel and augmented/evaluated metric stages. Encode frames as video.mp4 before video inspection.\n",
    )?;
    println!("{}", root.display());
    Ok(())
}
