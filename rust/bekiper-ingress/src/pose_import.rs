use std::path::{Path, PathBuf};

use polars::prelude::{LazyFrame, col, lit};
use serde::Deserialize;

use bekiper_core::error::{BekiperError, Result};
use bekiper_pose::PoseBackend;
use bekiper_pose::detect as pose_detect;
use bekiper_pose::io::{InputFormat, scan_file, write_parquet};

/// Options for importing a pose file into the normalized parquet layout.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoseImportOptions {
    pub invert_y: bool,
    pub frame_height_px: Option<u32>,
}

/// Metadata describing one normalized pose import.
#[derive(Debug, Clone)]
pub struct PoseImportReport {
    pub format: InputFormat,
    pub backend: Option<PoseBackend>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct DlcCropSidecar {
    #[serde(default)]
    dlc_crop: Option<DlcCropOffset>,
    #[serde(default)]
    x_offset: Option<f64>,
    #[serde(default)]
    y_offset: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct DlcCropOffset {
    x_offset: f64,
    y_offset: f64,
}

/// Collect `.csv`, `.h5`, and `.hdf5` inputs from a file or directory.
pub fn collect_convert_inputs(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        if is_convert_input(path) {
            return Ok(vec![path.to_path_buf()]);
        }
        return Err(BekiperError::Data(format!(
            "unsupported convert input '{}'; expected .csv, .h5, or .hdf5",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(BekiperError::Data(format!(
            "path does not exist or is not readable: {}",
            path.display()
        )));
    }

    let mut files = Vec::new();
    collect_convert_inputs_recursive(path, &mut files)?;
    files.sort();
    Ok(files)
}

/// Convert one supported pose file into normalized parquet.
pub fn import_pose_file_to_parquet(
    input_path: &Path,
    output_path: &Path,
    options: PoseImportOptions,
) -> Result<PoseImportReport> {
    let (lf, report) = load_normalized_pose(input_path, options)?;
    let mut df = lf.collect()?;
    write_parquet(&mut df, output_path)?;
    Ok(report)
}

/// Read a normalized pose plan directly from its source, without creating a
/// checkpoint. HDF5 decoding allocates column buffers; subsequent in-memory
/// plans share those buffers rather than serializing them between stages.
pub fn load_normalized_pose(
    input_path: &Path,
    options: PoseImportOptions,
) -> Result<(LazyFrame, PoseImportReport)> {
    let (lf, report) = load_pose_lazyframe(input_path)?;
    let lf = apply_optional_dlc_crop_offset(lf, input_path, &report.labels)?;
    let lf = apply_optional_invert_y(lf, &report.labels, options)?;
    Ok((lf, report))
}

/// Bounded HDF5 decoding with a cancellation checkpoint between hyperslabs.
/// Retained trial columns share batches; no full-table compound copy is built.
#[cfg(feature = "pose-hdf5")]
pub fn load_normalized_pose_batched(
    input: &Path,
    batch_rows: usize,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<(polars::prelude::DataFrame, Vec<String>)> {
    use polars::prelude::IntoLazy;
    checkpoint()?;
    if matches!(InputFormat::from_path(input)?, InputFormat::Hdf5) {
        let reader = bekiper_pose::deeplabcut::DeepLabCutReader::from_file(input)?;
        use bekiper_pose::PoseReader;
        let labels = reader.labels().to_vec();
        let mut frame: Option<polars::prelude::DataFrame> = None;
        reader.read_h5_batches(input, batch_rows, |batch| {
            checkpoint()?;
            if let Some(df) = &mut frame {
                df.vstack_mut(&batch)?;
            } else {
                frame = Some(batch);
            }
            Ok(())
        })?;
        let frame = frame.ok_or_else(|| BekiperError::Data("pose input is empty".into()))?;
        let frame = apply_optional_dlc_crop_offset(frame.lazy(), input, &labels)?.collect()?;
        Ok((frame, labels))
    } else {
        let (lf, report) = load_normalized_pose(input, PoseImportOptions::default())?;
        let df = lf.collect()?;
        checkpoint()?;
        Ok((df, report.labels))
    }
}

/// Decoded dimensions used for admission, without materializing pose payloads.
#[cfg(feature = "pose-hdf5")]
pub fn pose_source_shape(path: &Path) -> Result<(usize, usize)> {
    use polars::prelude::*;
    match InputFormat::from_path(path)? {
        InputFormat::Hdf5 => {
            bekiper_pose::deeplabcut::DeepLabCutReader::from_file(path)?.h5_shape(path)
        }
        InputFormat::Parquet => {
            let mut reader = ParquetReader::new(std::fs::File::open(path)?);
            Ok((reader.num_rows()?, reader.schema()?.len()))
        }
        InputFormat::Csv => {
            use std::io::BufRead;
            let reader = pose_detect::open(path)?;
            let rows = std::io::BufReader::new(std::fs::File::open(path)?)
                .lines()
                .try_fold(0usize, |n, line| line.map(|_| n.saturating_add(1)))?
                .saturating_sub(3);
            Ok((rows, 1 + 3 * reader.labels().len()))
        }
    }
}

fn collect_convert_inputs_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_convert_inputs_recursive(&path, out)?;
        } else if is_convert_input(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_convert_input(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            ext.eq_ignore_ascii_case("csv")
                || ext.eq_ignore_ascii_case("h5")
                || ext.eq_ignore_ascii_case("hdf5")
        })
}

fn load_pose_lazyframe(path: &Path) -> Result<(LazyFrame, PoseImportReport)> {
    let format = InputFormat::from_path(path)?;
    match format {
        InputFormat::Parquet => {
            let labels = discover_labels_from_parquet(path)?;
            Ok((
                scan_file(path)?,
                PoseImportReport {
                    format,
                    backend: None,
                    labels,
                },
            ))
        }
        InputFormat::Csv | InputFormat::Hdf5 => {
            let reader = pose_detect::open(path)?;
            let backend = reader.backend();
            let labels = reader.labels().to_vec();
            Ok((
                reader.read(path)?,
                PoseImportReport {
                    format,
                    backend: Some(backend),
                    labels,
                },
            ))
        }
    }
}

fn apply_optional_dlc_crop_offset(
    lf: LazyFrame,
    input_path: &Path,
    labels: &[String],
) -> Result<LazyFrame> {
    let Some(offset) = load_dlc_crop_offset(input_path)? else {
        return Ok(lf);
    };
    if offset.x_offset == 0.0 && offset.y_offset == 0.0 {
        return Ok(lf);
    }

    let mut exprs = Vec::with_capacity(labels.len() * 2);
    for label in labels {
        let x_col = format!("{label}_x");
        let y_col = format!("{label}_y");
        exprs.push((col(x_col.as_str()) + lit(offset.x_offset)).alias(x_col));
        exprs.push((col(y_col.as_str()) + lit(offset.y_offset)).alias(y_col));
    }
    Ok(lf.with_columns(exprs))
}

fn load_dlc_crop_offset(input_path: &Path) -> Result<Option<DlcCropOffset>> {
    let sidecar = input_path.with_extension("dlc_crop.toml");
    if !sidecar.exists() {
        return Ok(None);
    }

    let contents = std::fs::read_to_string(&sidecar).map_err(|e| {
        BekiperError::Data(format!(
            "failed to read DLC crop sidecar {}: {e}",
            sidecar.display()
        ))
    })?;
    let parsed: DlcCropSidecar = toml::from_str(&contents).map_err(|e| {
        BekiperError::Config(format!(
            "invalid DLC crop sidecar {}: {e}",
            sidecar.display()
        ))
    })?;
    if let Some(offset) = parsed.dlc_crop {
        return Ok(Some(offset));
    }
    match (parsed.x_offset, parsed.y_offset) {
        (Some(x_offset), Some(y_offset)) => Ok(Some(DlcCropOffset { x_offset, y_offset })),
        (None, None) => Ok(None),
        _ => Err(BekiperError::Config(format!(
            "invalid DLC crop sidecar {}: x_offset and y_offset must both be present",
            sidecar.display()
        ))),
    }
}

fn apply_optional_invert_y(
    lf: LazyFrame,
    labels: &[String],
    options: PoseImportOptions,
) -> Result<LazyFrame> {
    if !options.invert_y {
        return Ok(lf);
    }

    let Some(frame_height_px) = options.frame_height_px else {
        return Err(BekiperError::Config(
            "invert_y requires a source frame height in pixels".into(),
        ));
    };

    let height = frame_height_px as f64;
    let exprs = labels
        .iter()
        .map(|label| {
            let y_col = format!("{label}_y");
            (lit(height) - col(y_col.as_str())).alias(y_col)
        })
        .collect::<Vec<_>>();
    Ok(lf.with_columns(exprs))
}

fn discover_labels_from_parquet(path: &Path) -> Result<Vec<String>> {
    let mut lf = scan_file(path)?;
    let schema = lf.collect_schema()?;
    let mut labels = Vec::new();
    for name in schema.iter_names() {
        if let Some(label) = name.strip_suffix("_x")
            && label != "frame"
        {
            labels.push(label.to_string());
        }
    }
    if labels.is_empty() {
        return Err(BekiperError::Data(format!(
            "could not discover body-part labels from {}",
            path.display()
        )));
    }
    Ok(labels)
}

#[cfg(test)]
#[path = "pose_import/tests.rs"]
mod tests;
