use std::io::BufRead;
use std::path::Path;

use polars::prelude::*;

use bekiper_core::error::{BekiperError, Result};

use crate::{PoseBackend, PoseReader};

/// Reader for DeepLabCut output files (`.csv` and pandas/pytables `.h5`).
///
/// **CSV format**: multi-level header
///   Row 0: scorer name
///   Row 1: body part labels
///   Row 2: coordinate type (x, y, likelihood)
///
/// **HDF5 format**: pandas DataFrame stored via pytables in the
/// `df_with_missing` group, with a compound table dataset
/// `{i64 index, [f64; 3*K] values_block_0}` where K is the number of
/// body parts. Labels live in the pickled `non_index_axes` attribute.
///
/// Both formats produce the same flattened `{label}_x`, `{label}_y`,
/// `{label}_likelihood` columns plus a `frame` column.
pub struct DeepLabCutReader {
    labels: Vec<String>,
}

/// Returns true for `.h5` / `.hdf5` paths, case-insensitive.
fn is_hdf5_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("h5") || e.eq_ignore_ascii_case("hdf5"))
        .unwrap_or(false)
}

impl DeepLabCutReader {
    /// Create a reader with known labels (skips header parsing).
    pub fn with_labels(labels: Vec<String>) -> Self {
        Self { labels }
    }

    /// Create a reader that auto-detects labels from a DLC CSV or HDF5 file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let labels = if is_hdf5_path(path) {
            parse_dlc_h5_header(path)?
        } else {
            parse_dlc_header(path)?
        };
        Ok(Self { labels })
    }

    /// Read a DLC CSV, returning a LazyFrame with flat `{label}_{coord}` columns
    /// and a `frame` column.
    fn read_csv(&self, path: &Path) -> Result<LazyFrame> {
        let df = CsvReadOptions::default()
            .with_skip_rows(2)
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?;

        let mut renamed = df;
        let col_names: Vec<String> = renamed
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // First column is the frame index
        renamed.rename(&col_names[0], "frame".into())?;

        // Remaining columns are grouped by 3: x, y, likelihood per label
        let data_cols = &col_names[1..];
        for (i, label) in self.labels.iter().enumerate() {
            let base = i * 3;
            if base + 2 < data_cols.len() {
                renamed.rename(&data_cols[base], format!("{label}_x").into())?;
                renamed.rename(&data_cols[base + 1], format!("{label}_y").into())?;
                renamed.rename(&data_cols[base + 2], format!("{label}_likelihood").into())?;
            }
        }

        Ok(renamed.lazy())
    }

    /// Read a DLC pandas/pytables HDF5 file, returning a LazyFrame with the
    /// same flat `{label}_{coord}` columns as the CSV path.
    ///
    /// The dataset `/df_with_missing/table` is a compound type
    /// `{i64 index, [f64; 3*K] values_block_0}` where K is the body-part
    /// count. Because hdf5-metno's `H5Type` derive needs the compound size at
    /// compile time, we monomorphise a small set of common widths and
    /// dispatch at runtime.
    #[cfg(feature = "hdf5")]
    fn read_h5(&self, path: &Path) -> Result<LazyFrame> {
        let mut output: Option<DataFrame> = None;
        self.read_h5_batches(path, 8192, |batch| {
            if let Some(df) = &mut output {
                df.vstack_mut(&batch)?;
            } else {
                output = Some(batch);
            }
            Ok(())
        })?;
        Ok(output
            .ok_or_else(|| BekiperError::Data("DLC HDF5 table is empty".into()))?
            .lazy())
    }

    /// Inspect decoded dimensions without reading payload rows.
    #[cfg(feature = "hdf5")]
    pub fn h5_shape(&self, path: &Path) -> Result<(usize, usize)> {
        let file = hdf5_metno::File::open(path).map_err(|e| BekiperError::Data(e.to_string()))?;
        let dataset = file
            .dataset("df_with_missing/table")
            .map_err(|e| BekiperError::Data(e.to_string()))?;
        if dataset.ndim() != 1 {
            return Err(BekiperError::Data(
                "DLC HDF5 table must have one dimension".into(),
            ));
        }
        Ok((dataset.size(), 1 + 3 * self.labels.len()))
    }

    /// Read ordered hyperslabs, converting only the current slab to Arrow
    /// columns. Returning an error from `emit` stops before reading another slab.
    #[cfg(feature = "hdf5")]
    pub fn read_h5_batches(
        &self,
        path: &Path,
        batch_rows: usize,
        mut emit: impl FnMut(DataFrame) -> Result<()>,
    ) -> Result<()> {
        if batch_rows == 0 {
            return Err(BekiperError::Config(
                "HDF5 batch_rows must be positive".into(),
            ));
        }
        let file = hdf5_metno::File::open(path).map_err(|e| BekiperError::Data(e.to_string()))?;
        let dataset = file
            .dataset("df_with_missing/table")
            .map_err(|e| BekiperError::Data(e.to_string()))?;
        if dataset.ndim() != 1 {
            return Err(BekiperError::Data(
                "DLC HDF5 table must have one dimension".into(),
            ));
        }
        macro_rules! dispatch {
            ($($n:literal),*) => { match 3 * self.labels.len() {
                $($n => read_h5_table_batches::<$n>(&dataset, &self.labels, batch_rows, &mut emit),)*
                width => Err(BekiperError::Data(format!("unsupported DLC compound width {width}"))),
            } };
        }
        dispatch!(
            3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
            69, 72, 75, 78, 81, 84, 87, 90, 93, 96
        )
    }

    #[cfg(not(feature = "hdf5"))]
    fn read_h5(&self, path: &Path) -> Result<LazyFrame> {
        Err(BekiperError::Data(format!(
            "DLC h5 {} requires the bekiper-pose hdf5 feature",
            path.display()
        )))
    }
}

impl PoseReader for DeepLabCutReader {
    fn backend(&self) -> PoseBackend {
        PoseBackend::DeepLabCut
    }

    fn read(&self, path: &Path) -> Result<LazyFrame> {
        if is_hdf5_path(path) {
            self.read_h5(path)
        } else {
            self.read_csv(path)
        }
    }

    fn labels(&self) -> &[String] {
        &self.labels
    }
}

// ---------------------------------------------------------------------------
// HDF5 reading helpers
// ---------------------------------------------------------------------------

/// One row of `/df_with_missing/table` for a given compile-time float count.
///
/// `H5Type` derive supports const generics; we monomorphise this struct over
/// a fixed set of common K (body-part counts) below.
#[cfg(feature = "hdf5")]
#[derive(hdf5_metno::H5Type, Clone)]
#[repr(C)]
struct DlcRow<const N: usize> {
    index: i64,
    values_block_0: [f64; N],
}

/// Temporary compound storage is bounded by one hyperslab. There is no
/// full-table row-major staging buffer alongside the retained column buffers.
#[cfg(feature = "hdf5")]
fn read_h5_table_batches<const N: usize>(
    dataset: &hdf5_metno::Dataset,
    labels: &[String],
    batch_rows: usize,
    emit: &mut dyn FnMut(DataFrame) -> Result<()>,
) -> Result<()> {
    for start in (0..dataset.size()).step_by(batch_rows) {
        let end = start.saturating_add(batch_rows).min(dataset.size());
        let slab = dataset
            .read_slice_1d::<DlcRow<N>, _>(start..end)
            .map_err(|e| BekiperError::Data(format!("DLC HDF5 slab: {e}")))?;
        let mut columns = Vec::with_capacity(1 + N);
        columns.push(Column::new(
            "frame".into(),
            slab.iter().map(|r| r.index).collect::<Vec<_>>(),
        ));
        for (i, label) in labels.iter().enumerate() {
            for (j, suffix) in ["x", "y", "likelihood"].iter().enumerate() {
                columns.push(Column::new(
                    format!("{label}_{suffix}").into(),
                    slab.iter()
                        .map(|r| r.values_block_0[3 * i + j])
                        .collect::<Vec<_>>(),
                ));
            }
        }
        drop(slab);
        emit(DataFrame::new(end - start, columns)?)?;
    }
    Ok(())
}

/// Extract body-part labels from a DLC pandas/pytables HDF5 file.
///
/// The label list lives in the pickled `/df_with_missing` `non_index_axes`
/// attribute. Pytables writes pickle protocol 0 (text), one opcode per line.
/// V-prefixed lines are Unicode string opcodes, and after the scorer (which
/// is on a `(V…` line because it's the first element of the inner tuple),
/// the body parts and coord names appear on plain `V…` lines.
///
/// We collect those, drop the standard coord triple `x`/`y`/`likelihood`,
/// and what remains is the body parts in encounter order. The result is
/// validated against the table's compound width — if they disagree, we
/// fail loudly rather than guess.
#[cfg(feature = "hdf5")]
fn parse_dlc_h5_header(path: &Path) -> Result<Vec<String>> {
    let f = hdf5_metno::File::open(path)
        .map_err(|e| BekiperError::Data(format!("DLC h5 {}: open: {e}", path.display())))?;
    let g = f.group("df_with_missing").map_err(|e| {
        BekiperError::Data(format!(
            "DLC h5 {}: group df_with_missing: {e}",
            path.display()
        ))
    })?;
    let attr = g.attr("non_index_axes").map_err(|e| {
        BekiperError::Data(format!(
            "DLC h5 {}: attr non_index_axes: {e}",
            path.display()
        ))
    })?;
    // Pytables stores this as fixed-length ASCII. STRSIZE varies with the
    // model name length but stays well under 8 KiB for any realistic skeleton.
    let pickle = attr
        .read_scalar::<hdf5_metno::types::FixedAscii<8192>>()
        .map_err(|e| {
            BekiperError::Data(format!(
                "DLC h5 {}: read non_index_axes: {e}",
                path.display()
            ))
        })?;

    let labels = extract_dlc_labels_from_pickle(pickle.as_str());

    // Cross-check against the actual compound width on disk.
    let ds = g.dataset("table").map_err(|e| {
        BekiperError::Data(format!("DLC h5 {}: dataset table: {e}", path.display()))
    })?;
    let dtype_size = ds
        .dtype()
        .map_err(|e| BekiperError::Data(format!("DLC h5 {}: dtype: {e}", path.display())))?
        .size();
    // Compound layout: i64 index (8 bytes) + N * f64
    if dtype_size < 8 || (dtype_size - 8) % 8 != 0 {
        return Err(BekiperError::Data(format!(
            "DLC h5 {}: unexpected compound size {dtype_size}",
            path.display()
        )));
    }
    let n_floats = (dtype_size - 8) / 8;
    if n_floats % 3 != 0 {
        return Err(BekiperError::Data(format!(
            "DLC h5 {}: compound width {n_floats} is not a multiple of 3 (x/y/likelihood)",
            path.display()
        )));
    }
    let expected = n_floats / 3;
    if labels.len() != expected {
        return Err(BekiperError::Data(format!(
            "DLC h5 {}: extracted {} body-part labels but table has room for {expected} \
             (compound width = {n_floats}); pickled non_index_axes parsing may be wrong",
            path.display(),
            labels.len()
        )));
    }
    if labels.is_empty() {
        return Err(BekiperError::Data(format!(
            "DLC h5 {}: no body part labels found in non_index_axes",
            path.display()
        )));
    }
    Ok(labels)
}

#[cfg(not(feature = "hdf5"))]
fn parse_dlc_h5_header(path: &Path) -> Result<Vec<String>> {
    Err(BekiperError::Data(format!(
        "DLC h5 {} requires the bekiper-pose hdf5 feature",
        path.display()
    )))
}

/// Pure helper for testing: parses a pytables pickle-protocol-0 string and
/// returns DLC body-part labels (in encounter order, with `x`/`y`/`likelihood`
/// removed). Caller is responsible for validating the count against the
/// dataset's compound width.
#[cfg(any(feature = "hdf5", test))]
fn extract_dlc_labels_from_pickle(pickle: &str) -> Vec<String> {
    pickle
        .lines()
        .map(|l| l.trim_start())
        .filter_map(|l| l.strip_prefix('V'))
        .filter(|s| !matches!(*s, "x" | "y" | "likelihood"))
        .map(|s| s.to_string())
        .collect()
}

/// Parse the DLC multi-level header to extract unique body part labels in order.
fn parse_dlc_header(path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();

    // Skip row 0 (scorer)
    lines.next();

    // Row 1: bodyparts
    let bodyparts_line = lines
        .next()
        .ok_or_else(|| BekiperError::Data("DLC CSV missing bodyparts header row".into()))??;

    let parts: Vec<&str> = bodyparts_line.split(',').collect();
    // Skip first cell ("bodyparts"), then deduplicate preserving order
    let mut labels = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for part in &parts[1..] {
        let part = part.trim();
        if !part.is_empty() && seen.insert(part.to_string()) {
            labels.push(part.to_string());
        }
    }

    if labels.is_empty() {
        return Err(BekiperError::Data(
            "No body part labels found in DLC header".into(),
        ));
    }

    Ok(labels)
}

#[cfg(test)]
#[path = "deeplabcut/tests.rs"]
mod tests;
