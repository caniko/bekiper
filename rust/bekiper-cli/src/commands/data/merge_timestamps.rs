use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::Args;
use polars::prelude::Column;

use bekiper_pose::detect as pose_detect;
use bekiper_pose::io::{InputFormat, scan_file};
use bekiper_pose::steps::timestamp::load_elapsed_timestamps;

/// Merge per-frame timestamp CSVs into authoritative `*-timestamped.parquet` pose files.
///
/// Scans one dataset directory containing per-animal or per-session subdirectories.
/// Within each subdirectory, timestamp CSVs and coordinate files are matched by the
/// filename prefix before `delimiter`, mirroring the legacy ingress helper.
#[derive(Args)]
pub struct MergeTimestampsArgs {
    /// Dataset directory containing per-unit subdirectories.
    pub dataset_dir: PathBuf,

    /// Overwrite existing `*-timestamped.parquet` outputs.
    #[arg(long, default_value_t = false)]
    pub force: bool,

    /// Delete the matched timestamp CSV and source coordinate file after a successful merge.
    #[arg(long, default_value_t = false)]
    pub delete_source: bool,

    /// Delimiter used to extract the shared label from timestamp/coordinate stems.
    #[arg(long, default_value = ".")]
    pub delimiter: String,
}

#[derive(Default)]
struct MergeStats {
    merged: usize,
    skipped_existing: usize,
    skipped_missing_timestamp: usize,
    skipped_mismatched_lengths: usize,
}

pub fn run(args: MergeTimestampsArgs) -> anyhow::Result<()> {
    let unit_dirs = dataset_unit_dirs(&args.dataset_dir)?;
    if unit_dirs.is_empty() {
        tracing::info!(dir = %args.dataset_dir.display(), "no dataset-unit directories found");
        return Ok(());
    }

    let mut totals = MergeStats::default();

    for unit_dir in &unit_dirs {
        let unit_stats =
            merge_dataset_unit(unit_dir, &args.delimiter, args.force, args.delete_source)?;
        totals.merged += unit_stats.merged;
        totals.skipped_existing += unit_stats.skipped_existing;
        totals.skipped_missing_timestamp += unit_stats.skipped_missing_timestamp;
        totals.skipped_mismatched_lengths += unit_stats.skipped_mismatched_lengths;
    }

    tracing::info!(
        dataset_dir = %args.dataset_dir.display(),
        unit_dirs = unit_dirs.len(),
        merged = totals.merged,
        skipped_existing = totals.skipped_existing,
        skipped_missing_timestamp = totals.skipped_missing_timestamp,
        skipped_mismatched_lengths = totals.skipped_mismatched_lengths,
        "timestamp merge complete"
    );

    Ok(())
}

fn merge_dataset_unit(
    unit_dir: &Path,
    delimiter: &str,
    force: bool,
    delete_source: bool,
) -> anyhow::Result<MergeStats> {
    let timestamp_files = timestamp_files_by_label(unit_dir, delimiter)?;
    let coordinate_files = coordinate_files_by_label(unit_dir, delimiter)?;
    let mut stats = MergeStats::default();

    for (label, coord_file) in coordinate_files {
        let output_path = timestamped_output_path(&coord_file)?;
        if output_path.exists() && !force {
            stats.skipped_existing += 1;
            continue;
        }

        let Some(timestamp_file) = timestamp_files.get(&label) else {
            stats.skipped_missing_timestamp += 1;
            continue;
        };

        let elapsed = load_elapsed_timestamps(timestamp_file)?;
        let mut df = load_pose_dataframe(&coord_file)?;
        if elapsed.len() != df.height() {
            tracing::warn!(
                coord = %coord_file.display(),
                timestamp = %timestamp_file.display(),
                frame_count = df.height(),
                timestamp_count = elapsed.len(),
                "timestamp merge skipped due to frame-count mismatch"
            );
            stats.skipped_mismatched_lengths += 1;
            continue;
        }

        let _ = df.with_column(Column::new("elapsed_s".into(), elapsed))?;
        bekiper_pose::io::write_parquet(&mut df, &output_path)?;

        if delete_source {
            std::fs::remove_file(timestamp_file)?;
            std::fs::remove_file(&coord_file)?;
        }

        tracing::info!(
            coord = %coord_file.display(),
            timestamp = %timestamp_file.display(),
            output = %output_path.display(),
            delete_source,
            "merged timestamps into pose parquet"
        );
        stats.merged += 1;
    }

    Ok(stats)
}

fn dataset_unit_dirs(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn timestamp_files_by_label(
    unit_dir: &Path,
    delimiter: &str,
) -> anyhow::Result<HashMap<String, PathBuf>> {
    let mut by_label: HashMap<String, PathBuf> = HashMap::new();
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(unit_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if ext.eq_ignore_ascii_case("csv") && stem.contains("timestamps-") {
            candidates.push(path);
        }
    }
    candidates.sort();

    for path in candidates {
        let label = leading_label(&path, delimiter)?;
        if let Some(existing) = by_label.insert(label.clone(), path.clone()) {
            tracing::warn!(
                label,
                kept = %existing.display(),
                ignored = %path.display(),
                "multiple timestamp files found for label, using the last sorted path"
            );
        }
    }
    Ok(by_label)
}

fn coordinate_files_by_label(
    unit_dir: &Path,
    delimiter: &str,
) -> anyhow::Result<HashMap<String, PathBuf>> {
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(unit_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if is_coordinate_source(&path) {
            candidates.push(path);
        }
    }
    candidates.sort();

    let mut by_label: HashMap<String, PathBuf> = HashMap::new();

    for path in candidates {
        let label = leading_label(&path, delimiter)?;
        match by_label.get(&label) {
            Some(existing) if coordinate_priority(existing) <= coordinate_priority(&path) => {}
            Some(existing) => {
                tracing::info!(
                    label,
                    old = %existing.display(),
                    new = %path.display(),
                    "preferring higher-priority coordinate source"
                );
                by_label.insert(label, path);
            }
            None => {
                by_label.insert(label, path);
            }
        }
    }

    Ok(by_label)
}

fn is_coordinate_source(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };

    let supported = ext.eq_ignore_ascii_case("parquet")
        || ext.eq_ignore_ascii_case("csv")
        || ext.eq_ignore_ascii_case("h5")
        || ext.eq_ignore_ascii_case("hdf5");
    supported && stem.contains("coordinates-") && !stem.ends_with("-timestamped")
}

fn coordinate_priority(path: &Path) -> u8 {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("parquet") => 0,
        Some(ext) if ext.eq_ignore_ascii_case("csv") => 1,
        Some(ext) if ext.eq_ignore_ascii_case("h5") || ext.eq_ignore_ascii_case("hdf5") => 2,
        _ => 255,
    }
}

fn leading_label(path: &Path, delimiter: &str) -> anyhow::Result<String> {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid UTF-8 filename: {}", path.display()))?;
    Ok(stem.split(delimiter).next().unwrap_or(stem).to_string())
}

fn timestamped_output_path(coord_file: &Path) -> anyhow::Result<PathBuf> {
    let stem = coord_file
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid UTF-8 filename: {}", coord_file.display()))?;
    Ok(coord_file.with_file_name(format!("{stem}-timestamped.parquet")))
}

fn load_pose_dataframe(path: &Path) -> anyhow::Result<polars::prelude::DataFrame> {
    match InputFormat::from_path(path)? {
        InputFormat::Parquet => Ok(scan_file(path)?.collect()?),
        InputFormat::Csv | InputFormat::Hdf5 => {
            let reader = pose_detect::open(path)?;
            Ok(reader.read(path)?.collect()?)
        }
    }
}

#[cfg(test)]
#[path = "merge_timestamps/tests.rs"]
mod tests;
