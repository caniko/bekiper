use std::path::{Path, PathBuf};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{PerimeterSpec, PerimetersFile};
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_pose::detect as pose_detect;
use bekiper_pose::io::{InputFormat, scan_file};

use crate::config::ExperimentConfig;
use crate::stage::ExperimentOutputDir;

/// Resolve the Python interpreter.
///
/// Uses `python3` from PATH — in the nix devShell this is the nix-provided
/// Python with `bekiper-cli` and `bekiper-inspect` available as nix packages.
pub(crate) fn resolve_python() -> PathBuf {
    PathBuf::from("python3")
}

fn load_perimeters_file(path: &Path, path_label: &str) -> Result<Vec<PerimeterSpec>> {
    let content = std::fs::read_to_string(path)?;
    let pf: PerimetersFile = toml::from_str(&content)
        .map_err(|e| BekiperError::Config(format!("invalid {path_label}: {e}")))?;
    Ok(pf.perimeters)
}

pub(crate) fn load_perimeters(exp_dir: &ExperimentOutputDir) -> Result<Vec<PerimeterSpec>> {
    let path = exp_dir.perimeters_toml();
    if !path.exists() {
        return Ok(Vec::new());
    }
    load_perimeters_file(&path, "perimeters.toml")
}

pub(crate) fn load_pixel_perimeters(exp_dir: &ExperimentOutputDir) -> Result<Vec<PerimeterSpec>> {
    let path = exp_dir.pixel_perimeters_toml();
    if !path.exists() {
        return Err(BekiperError::Config(format!(
            "pixel_perimeters.toml not found at {} — rerun from detect stage",
            path.display()
        )));
    }
    load_perimeters_file(&path, "pixel_perimeters.toml")
}

/// Load the full scale derivation info from scale.toml (written by augment stage).
pub(crate) fn load_scale_info(exp_dir: &ExperimentOutputDir) -> Result<ScaleDerivationInfo> {
    let path = exp_dir.scale_toml();
    if !path.exists() {
        return Err(BekiperError::Config(
            "scale.toml not found — run augment stage first".into(),
        ));
    }
    let content = std::fs::read_to_string(&path)?;
    toml::from_str(&content).map_err(|e| BekiperError::Config(format!("invalid scale.toml: {e}")))
}

/// Discover body-part labels from the original data file or from parquet column names.
pub(crate) fn discover_labels(
    experiment: &ExperimentConfig,
    poses_path: &Path,
) -> Result<Vec<String>> {
    if let Some(data_file) = experiment.data_files.first() {
        return discover_labels_from_source(data_file);
    }

    // Fallback: infer labels from parquet column names (strip _x suffix)
    let mut lf = scan_file(poses_path)?;
    let schema = lf.collect_schema()?;
    let mut labels: Vec<String> = Vec::new();
    for name in schema.iter_names() {
        if let Some(label) = name.strip_suffix("_x")
            && label != "frame"
        {
            labels.push(label.to_string());
        }
    }
    if labels.is_empty() {
        return Err(BekiperError::Data(
            "could not discover body-part labels from poses.parquet".into(),
        ));
    }
    Ok(labels)
}

pub(crate) fn discover_labels_from_source(path: &Path) -> Result<Vec<String>> {
    if matches!(InputFormat::from_path(path)?, InputFormat::Parquet) {
        return discover_labels_from_parquet(path);
    }

    let reader = pose_detect::open(path)?;
    Ok(reader.labels().to_vec())
}

/// Discover body-part labels from parquet column names (strip _x suffix).
pub(crate) fn discover_labels_from_parquet(path: &Path) -> Result<Vec<String>> {
    let mut lf = scan_file(path)?;
    let schema = lf.collect_schema()?;
    let mut labels: Vec<String> = Vec::new();
    for name in schema.iter_names() {
        if let Some(label) = name.strip_suffix("_x")
            && label != "frame"
        {
            labels.push(label.to_string());
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use crate::stage::ExperimentOutputDir;

    use super::*;

    #[test]
    fn discover_labels_from_source_supports_parquet() {
        let dir = std::env::temp_dir().join("bekiper_discover_labels_from_parquet");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("poses.parquet");

        let mut df = polars::df! {
            "frame" => &[0i64, 1],
            "nose_x" => &[1.0f64, 2.0],
            "nose_y" => &[3.0f64, 4.0],
            "nose_likelihood" => &[0.9f64, 0.95],
            "tail_base_x" => &[5.0f64, 6.0],
            "tail_base_y" => &[7.0f64, 8.0],
            "tail_base_likelihood" => &[0.8f64, 0.85],
        }
        .unwrap();
        bekiper_pose::io::write_parquet(&mut df, &path).unwrap();

        let labels = discover_labels_from_source(&path).unwrap();
        assert_eq!(labels, vec!["nose".to_string(), "tail_base".to_string()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn perimeter_loaders_use_split_checkpoint_paths() {
        let dir = std::env::temp_dir().join("bekiper_split_perimeter_loaders");
        let _ = std::fs::remove_dir_all(&dir);
        let exp_dir = ExperimentOutputDir::new(&dir, "exp");
        exp_dir.ensure_checkpoints_dir().unwrap();

        let pixel = PerimetersFile {
            perimeters: vec![PerimeterSpec::Rectangle {
                label: "arena".into(),
                center_x: 500.0,
                center_y: 400.0,
                width: 800.0,
                height: 600.0,
            }],
        };
        std::fs::write(
            exp_dir.pixel_perimeters_toml(),
            toml::to_string_pretty(&pixel).unwrap(),
        )
        .unwrap();

        let meter = PerimetersFile {
            perimeters: vec![PerimeterSpec::Rectangle {
                label: "arena".into(),
                center_x: 0.5,
                center_y: 0.4,
                width: 0.8,
                height: 0.6,
            }],
        };
        std::fs::write(
            exp_dir.perimeters_toml(),
            toml::to_string_pretty(&meter).unwrap(),
        )
        .unwrap();

        let pixel_loaded = load_pixel_perimeters(&exp_dir).unwrap();
        let meter_loaded = load_perimeters(&exp_dir).unwrap();

        assert_eq!(pixel_loaded, pixel.perimeters);
        assert_eq!(meter_loaded, meter.perimeters);

        std::fs::remove_dir_all(&dir).ok();
    }
}
