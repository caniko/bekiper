use std::path::{Path, PathBuf};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::PerimeterSpec;
use bekiper_core::types::{DetectionRecipe, MetersPerPixel};

use super::scan::ScanConfig;

/// Recursively collect files matching the given extensions.
pub(super) fn collect_videos(
    dir: &Path,
    extensions: &[String],
    out: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_videos(&path, extensions, out)?;
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
            && extensions.iter().any(|e| e.eq_ignore_ascii_case(ext))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// Derive a snake_case experiment name from a relative video path.
///
/// `NORT videos/NORT (post)/T1/Config A/post.T1.configA.video-10.mp4`
/// -> `nort_videos_nort_post_t1_config_a_post_t1_configa_video_10`
pub(super) fn derive_experiment_name(rel_path: &Path) -> String {
    let s = rel_path.with_extension("").to_string_lossy().to_string();
    let mut name = String::with_capacity(s.len());
    let mut prev_was_sep = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if prev_was_sep && !name.is_empty() {
                name.push('_');
            }
            name.push(c.to_ascii_lowercase());
            prev_was_sep = false;
        } else {
            prev_was_sep = true;
        }
    }
    name
}

/// Resolve a precomputed pose file mirrored from a discovered video path.
///
/// If `pose_data_dir = "deeplabcut/poses"` and the video is
/// `NORT videos/.../video-10.mp4`, the required pose file is
/// `deeplabcut/poses/NORT videos/.../video-10.h5`.
pub(super) fn resolve_mirrored_pose_data_file(
    base_dir: &Path,
    pose_data_dir: &Path,
    rel_video_path: &Path,
) -> Result<PathBuf> {
    let mut rel_pose_path = rel_video_path.to_path_buf();
    rel_pose_path.set_extension("h5");
    let absolute = base_dir.join(pose_data_dir).join(&rel_pose_path);
    if !absolute.exists() {
        return Err(BekiperError::Data(format!(
            "missing mirrored pose file for video {}: expected {}",
            rel_video_path.display(),
            absolute.display()
        )));
    }
    Ok(pose_data_dir.join(rel_pose_path))
}

/// Pick the recipe for a video: check `recipe_rules` glob patterns first,
/// then fall back to the scan's default `recipe`.
pub(super) fn resolve_recipe(scan: &ScanConfig, rel_path: &Path) -> Option<DetectionRecipe> {
    let path_str = rel_path.to_string_lossy();

    for (pattern, recipe) in &scan.recipe_rules {
        if glob_matches(pattern, &path_str) {
            return Some(*recipe);
        }
    }

    scan.recipe
}

/// Simple glob matching supporting `*` (single segment) and `**` (any depth).
pub(super) fn glob_matches(pattern: &str, path: &str) -> bool {
    let pattern = pattern.replace('\\', "/");
    let path = path.replace('\\', "/");

    let pat_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    match_segments(&pat_parts, &path_parts)
}

fn match_segments(pat: &[&str], path: &[&str]) -> bool {
    if pat.is_empty() {
        return path.is_empty();
    }
    if pat[0] == "**" {
        for i in 0..=path.len() {
            if match_segments(&pat[1..], &path[i..]) {
                return true;
            }
        }
        return false;
    }
    if path.is_empty() {
        return false;
    }
    if segment_matches(pat[0], path[0]) {
        match_segments(&pat[1..], &path[1..])
    } else {
        false
    }
}

pub(super) fn segment_matches(pattern: &str, segment: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern.eq_ignore_ascii_case(segment);
    }
    let mut remaining = segment;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        let part_lower = part.to_ascii_lowercase();
        let remaining_lower = remaining.to_ascii_lowercase();
        if i == 0 {
            if !remaining_lower.starts_with(&part_lower) {
                return false;
            }
            remaining = &remaining[part.len()..];
        } else if let Some(pos) = remaining_lower.find(&part_lower) {
            remaining = &remaining[pos + part.len()..];
        } else {
            return false;
        }
    }
    true
}

/// Find a DLC data file for a given animal and sequence.
///
/// Looks for a normalized `{seq}.coordinates-{animal}_*.parquet` first,
/// then legacy `.csv`, then raw `.h5`.
pub(super) fn find_dlc_data_file(animal_dir: &Path, animal: u32, sequence: u32) -> Result<PathBuf> {
    let prefix = format!("{sequence}.coordinates-{animal}_");

    // Prefer normalized parquet produced by the Rust convert path or the
    // legacy timestamp merge utility.
    let parquet_files = find_files_with_prefix_and_ext(animal_dir, &prefix, "parquet")?;
    if let Some(timestamped) = parquet_files.iter().find(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| stem.ends_with("-timestamped"))
    }) {
        return Ok(timestamped.clone());
    }
    if let Some(parquet) = parquet_files.into_iter().next() {
        return Ok(parquet);
    }

    // Then prefer CSV over raw HDF5.
    if let Some(csv) = find_file_with_prefix_and_ext(animal_dir, &prefix, "csv")? {
        return Ok(csv);
    }

    // Fall back to H5
    if let Some(h5) = find_file_with_prefix_and_ext(animal_dir, &prefix, "h5")? {
        return Ok(h5);
    }
    if let Some(hdf5) = find_file_with_prefix_and_ext(animal_dir, &prefix, "hdf5")? {
        return Ok(hdf5);
    }

    Err(BekiperError::Data(format!(
        "no DLC data file found for animal {} sequence {} in {}",
        animal,
        sequence,
        animal_dir.display()
    )))
}

fn find_file_with_prefix_and_ext(dir: &Path, prefix: &str, ext: &str) -> Result<Option<PathBuf>> {
    Ok(find_files_with_prefix_and_ext(dir, prefix, ext)?
        .into_iter()
        .next())
}

fn find_files_with_prefix_and_ext(dir: &Path, prefix: &str, ext: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(prefix) && name.ends_with(&format!(".{ext}")) {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
}

/// Find the per-frame timestamps CSV for a given animal and sequence.
///
/// Looks for `{seq}.timestamps-{animal}_*.csv`.  Each row carries the
/// wall-clock time the frame was captured (ISO 8601 in the last column).
/// Returns `None` when no file exists (non-fatal — the pipeline falls
/// back to constant-FPS timing).
pub(super) fn find_timestamps_file(
    animal_dir: &Path,
    animal: u32,
    sequence: u32,
) -> Option<PathBuf> {
    let prefix = format!("{sequence}.timestamps-{animal}_");
    find_file_with_prefix_and_ext(animal_dir, &prefix, "csv")
        .ok()
        .flatten()
}

/// Load a single scale calibration from a `meters_per_pixel-*-{id}.csv` file.
///
/// Filenames follow the pattern `meters_per_pixel-{method}-{value}-{mpp_id}.csv`.
/// The physical meters value is between the method and the ID suffix.
/// We parse the CSV directly (same format: `label,x1,y1,x2,y2,image,w,h`).
pub(super) fn load_single_scale_calibration(
    animal_dir: &Path,
    mpp_id: &str,
) -> Result<MetersPerPixel> {
    // The mpp_id may reference a different animal's dir
    // (e.g. 782 uses 781's calibration files).
    let target_animal = mpp_id.split('_').next().unwrap_or(mpp_id);
    let search_dir = animal_dir
        .parent()
        .unwrap_or(animal_dir)
        .join(target_animal);
    let search_dir = if search_dir.exists() {
        search_dir
    } else {
        animal_dir.to_path_buf()
    };

    let target_suffix = format!("-{mpp_id}.csv");
    for entry in std::fs::read_dir(&search_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("meters_per_pixel-") || !name_str.ends_with(&target_suffix) {
            continue;
        }

        // Extract physical meters from filename:
        // "meters_per_pixel-diagonal-0.707-1035_0.csv"
        //  strip prefix -> "diagonal-0.707-1035_0.csv"
        //  strip suffix -> "diagonal-0.707"
        //  split on '-' -> ["diagonal", "0.707"]
        let stem = name_str
            .strip_prefix("meters_per_pixel-")
            .unwrap()
            .strip_suffix(&target_suffix)
            .unwrap_or(&name_str);
        // stem = "diagonal-0.707" or similar
        let physical_meters: f64 = stem
            .rsplit('-')
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| {
                BekiperError::Data(format!(
                    "cannot parse physical meters from filename: {}",
                    name_str
                ))
            })?;

        // Parse CSV to get pixel endpoints
        let path = entry.path();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(&path)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;

        if let Some(record) = reader.records().next() {
            let record =
                record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
            if record.len() >= 5 {
                let x1: f64 = record[1]
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: x1: {e}", path.display())))?;
                let y1: f64 = record[2]
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: y1: {e}", path.display())))?;
                let x2: f64 = record[3]
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: x2: {e}", path.display())))?;
                let y2: f64 = record[4]
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: y2: {e}", path.display())))?;

                let pixel_distance = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
                if pixel_distance > 0.0 {
                    return Ok(MetersPerPixel(physical_meters / pixel_distance));
                }
            }
        }
    }

    Err(BekiperError::Data(format!(
        "no scale calibration found for mpp_id '{}' in {}",
        mpp_id,
        search_dir.display()
    )))
}

/// Load perimeters for a given sequence from circle CSV files.
///
/// Looks for `{seq}.perimeter-circle-{animal}.csv` (or the `periemter` typo variant).
pub(super) fn load_sequence_perimeters(
    animal_dir: &Path,
    animal: u32,
    sequence: u32,
    _recipe: DetectionRecipe,
) -> Result<Vec<PerimeterSpec>> {
    // Search for perimeter files matching this sequence.
    // Filenames vary: `{seq}.perimeter-circle-{animal}.csv`,
    // `{seq}.perimeter-circle-{animal}_{sub}.csv`, `{seq}.perimeter-{animal}.csv`, etc.
    let prefix = format!("{sequence}.");
    for entry in std::fs::read_dir(animal_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(&prefix)
            && (name.contains("perimeter") || name.contains("periemter"))
            && name.ends_with(".csv")
        {
            return bekiper_annotation::parse_circle_csv(&entry.path());
        }
    }

    tracing::warn!(
        animal,
        sequence,
        "no perimeter file found, skipping perimeters"
    );
    Ok(Vec::new())
}

/// Load the arena center point and construct a center zone rectangle.
///
/// Creates a perimeter labeled `"center_zone"` (not `"arena"`) in **pixel
/// space** so that the augment stage's `perimeter.scale(mpp)` correctly
/// converts it to meters.
pub(super) fn load_center_zone(
    animal_dir: &Path,
    center_id: &str,
    center_zone_meters: (f64, f64),
    mpp: MetersPerPixel,
) -> Result<Option<PerimeterSpec>> {
    let center = load_center_point(animal_dir, center_id)?;
    let Some(center) = center else {
        return Ok(None);
    };

    // Convert meter dimensions to pixel space so that the augment stage's
    // `perimeter.scale(mpp)` correctly converts back to meters.
    let width_px = center_zone_meters.0 / mpp.0;
    let height_px = center_zone_meters.1 / mpp.0;

    Ok(Some(PerimeterSpec::Rectangle {
        label: "center_zone".into(),
        center_x: center.x,
        center_y: center.y,
        width: width_px,
        height: height_px,
    }))
}

/// Construct the full arena boundary rectangle from `arena_meters`.
///
/// Uses the same center point as the center zone. The resulting perimeter
/// is labeled `"arena"` and stored in pixel space for the augment stage.
pub(super) fn load_arena_boundary(
    animal_dir: &Path,
    center_id: &str,
    arena_meters: (f64, f64),
    mpp: MetersPerPixel,
) -> Result<Option<PerimeterSpec>> {
    let center = load_center_point(animal_dir, center_id)?;
    let Some(center) = center else {
        return Ok(None);
    };

    let width_px = arena_meters.0 / mpp.0;
    let height_px = arena_meters.1 / mpp.0;

    Ok(Some(PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: center.x,
        center_y: center.y,
        width: width_px,
        height: height_px,
    }))
}

/// Resolve the center-point CSV for an animal, returning the parsed point.
fn load_center_point(
    animal_dir: &Path,
    center_id: &str,
) -> Result<Option<bekiper_annotation::CenterPoint>> {
    // The center_id may reference a different animal's dir
    let target_animal = center_id.split('_').next().unwrap_or(center_id);
    let search_dir = animal_dir
        .parent()
        .unwrap_or(animal_dir)
        .join(target_animal);
    let search_dir = if search_dir.exists() {
        search_dir
    } else {
        animal_dir.to_path_buf()
    };

    let center_path = search_dir.join(format!("center-{center_id}.csv"));
    if !center_path.exists() {
        tracing::debug!(center_id, "center file not found, skipping");
        return Ok(None);
    }

    Ok(Some(bekiper_annotation::parse_center_csv(&center_path)?))
}

/// Metadata extracted from a video metadata YAML file.
pub(super) struct VideoMetaFromYaml {
    pub fps: f64,
    pub resolution: Option<(u32, u32)>,
    /// The base name (e.g. `0.video-1179_0`) — the video file shares this stem.
    pub stem: String,
}
