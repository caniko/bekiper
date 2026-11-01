/// Load video metadata from a `{seq}.video_metadata-{animal}.yaml` file.
///
/// Parses `average_fps`, `resolution`, and `name` (stem) from the YAML.
/// Also accepts the legacy `{seq}.video-{animal}_*-metadata.yaml` pattern.
pub(super) fn load_video_metadata(
    animal_dir: &Path,
    animal: u32,
    sequence: u32,
) -> Result<VideoMetaFromYaml> {
    // Try canonical name first, then scan for legacy pattern.
    let canonical = animal_dir.join(format!("{sequence}.video_metadata-{animal}.yaml"));
    let meta_path = if canonical.is_file() {
        canonical
    } else {
        let legacy_prefix = format!("{sequence}.video-{animal}_");
        let legacy_suffix = "-metadata.yaml";
        let mut found = None;
        for entry in std::fs::read_dir(animal_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&legacy_prefix) && name.ends_with(legacy_suffix) {
                found = Some(entry.path());
                break;
            }
        }
        match found {
            Some(p) => p,
            None => {
                tracing::warn!(
                    animal,
                    sequence,
                    "no video metadata found, defaulting to 30.0 fps"
                );
                return Ok(VideoMetaFromYaml {
                    fps: 30.0,
                    resolution: None,
                    stem: format!("{sequence}.video-{animal}"),
                });
            }
        }
    };

    let content = std::fs::read_to_string(&meta_path)?;
    let mut fps: Option<f64> = None;
    let mut stem: Option<String> = None;
    let mut res_lines: Vec<u32> = Vec::new();
    let mut in_resolution = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("average_fps:") {
            fps = Some(rest.trim().parse().map_err(|e| {
                BekiperError::Data(format!("{}: average_fps: {e}", meta_path.display()))
            })?);
            in_resolution = false;
        } else if let Some(rest) = trimmed.strip_prefix("name:") {
            stem = Some(rest.trim().to_string());
            in_resolution = false;
        } else if trimmed.starts_with("resolution:") {
            in_resolution = true;
        } else if in_resolution {
            if let Some(rest) = trimmed.strip_prefix("- ") {
                if let Ok(v) = rest.trim().parse::<u32>() {
                    res_lines.push(v);
                }
            } else {
                in_resolution = false;
            }
        }
    }

    let resolution = if res_lines.len() >= 2 {
        Some((res_lines[0], res_lines[1]))
    } else {
        None
    };

    let fname = meta_path.file_name().unwrap_or_default().to_string_lossy();
    let stem = stem.unwrap_or_else(|| fname.strip_suffix(".yaml").unwrap_or(&fname).to_string());

    Ok(VideoMetaFromYaml {
        fps: fps.unwrap_or(30.0),
        resolution,
        stem,
    })
}

/// Search for a video file matching `{stem}.{ext}` in *animal_dir*.
///
/// Tries common video extensions in order: `.mp4`, `.avi`, `.mkv`, `.mov`.
pub(super) fn find_experiment_video(animal_dir: &Path, stem: &str) -> Option<PathBuf> {
    const EXTENSIONS: &[&str] = &["mp4", "avi", "mkv", "mov"];
    for ext in EXTENSIONS {
        let candidate = animal_dir.join(format!("{stem}.{ext}"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Search for a frame screenshot image: `{seq}.frame-{animal}.jpeg`.
///
/// These are arena screenshots extracted from the source video during
/// data preparation.  The Python inspector loads them directly as the
/// background frame for inspection figures.
pub(super) fn find_frame_image(animal_dir: &Path, animal: u32, sequence: u32) -> Option<PathBuf> {
    let candidate = animal_dir.join(format!("{sequence}.frame-{animal}.jpeg"));
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bekiper_io_helpers_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn find_dlc_data_file_prefers_parquet() {
        let dir = test_dir("prefers_parquet");
        let prefix = "1.coordinates-7_test";
        let csv = dir.join(format!("{prefix}.csv"));
        let h5 = dir.join(format!("{prefix}.h5"));
        let parquet = dir.join(format!("{prefix}.parquet"));

        std::fs::write(&csv, b"csv").unwrap();
        std::fs::write(&h5, b"h5").unwrap();
        std::fs::write(&parquet, b"parquet").unwrap();

        let selected = find_dlc_data_file(&dir, 7, 1).unwrap();
        assert_eq!(selected, parquet);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_dlc_data_file_prefers_timestamped_parquet_over_raw_parquet() {
        let dir = test_dir("prefers_timestamped_parquet");
        let raw = dir.join("4.coordinates-11_test.parquet");
        let timestamped = dir.join("4.coordinates-11_test-timestamped.parquet");
        std::fs::write(&raw, b"parquet").unwrap();
        std::fs::write(&timestamped, b"parquet").unwrap();

        let selected = find_dlc_data_file(&dir, 11, 4).unwrap();
        assert_eq!(selected, timestamped);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_dlc_data_file_falls_back_to_h5() {
        let dir = test_dir("falls_back_to_h5");
        let h5 = dir.join("2.coordinates-8_test.h5");
        std::fs::write(&h5, b"h5").unwrap();

        let selected = find_dlc_data_file(&dir, 8, 2).unwrap();
        assert_eq!(selected, h5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_dlc_data_file_accepts_hdf5_extension() {
        let dir = test_dir("falls_back_to_hdf5");
        let hdf5 = dir.join("5.coordinates-12_test.hdf5");
        std::fs::write(&hdf5, b"hdf5").unwrap();

        let selected = find_dlc_data_file(&dir, 12, 5).unwrap();
        assert_eq!(selected, hdf5);

        std::fs::remove_dir_all(&dir).ok();
    }
}
