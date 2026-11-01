/// Extract full video context from a video file path.
///
/// Parses filenames like `post.T1.configA.video-10.mp4` or paths containing
/// `(pre)/T2/Config B/video-7.mp4`.
///
/// Stage defaults to 0 when no T1/T2 is found (e.g. Y-maze: `pre.video-18.mp4`).
pub fn extract_video_context(video_path: &Path) -> Option<VideoContext> {
    let animal = extract_video_number(video_path)?;

    // Collect all path components + filename stem for stage/phase_part detection
    let path_str = video_path.to_str().unwrap_or("");
    let path_lower = path_str.to_lowercase();

    // Extract stage from T1/T2 in path or filename (0 if absent)
    let stage = if path_lower.contains("t1") || path_lower.contains("/t1/") {
        1
    } else if path_lower.contains("t2") || path_lower.contains("/t2/") {
        2
    } else {
        0
    };

    // Extract phase_part from pre/post in path or filename
    let phase_part = if path_lower.contains("pre") {
        1
    } else if path_lower.contains("post") {
        2
    } else {
        return None;
    };

    Some(VideoContext {
        animal,
        stage,
        phase_part,
    })
}

// ── Metadata CSV reader ─────────────────────────────────────────────────────

/// How a perimeter is sourced for a given experiment.
#[derive(Debug, Clone, PartialEq)]
pub enum PerimeterSource {
    /// Use the perimeter coordinates directly (same apparatus/setup as annotation).
    Direct { perimeter_id: String },
    /// Re-reference: translate coordinates from the annotation image to the target image.
    ReReferenced {
        perimeter_id: String,
        /// Image name stem from ChangeReferenceImageName (e.g. "A1_training_1").
        target_image: String,
    },
}

/// A row from the NORT-style metadata CSV.
#[derive(Debug)]
pub struct MetadataRow {
    pub phase: String,
    pub phase_part: u32,
    pub animal: u32,
    pub stage: u32,
    pub source: Option<PerimeterSource>,
}

/// Read the metadata CSV and extract (Animal, Stage, PhasePart) → perimeter ID mappings.
///
/// Supports two column layouts:
///
/// - **NORT-style**: `Phase`, `PhasePart`, `Animal`, `Stage`, `Perimeter`, `ChangeReference`
/// - **Y-maze-style**: `Part`, `Number`, `ChangeReference`, `ChangeReferenceImageName`
///
/// Phase/PhasePart/Part use forward-fill (only set on the first row of each group).
/// Uses `Perimeter` if non-empty, otherwise falls back to `ChangeReference`.
/// Stage defaults to 0 when the column is absent (e.g. Y-maze).
pub fn read_metadata_csv(path: &Path) -> Result<Vec<MetadataRow>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let headers = reader
        .headers()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?
        .clone();

    // Flexible column detection: NORT columns or Y-maze fallbacks
    let phase_idx = find_column_opt(&headers, "Phase");
    let phase_part_idx =
        find_column_opt(&headers, "PhasePart").or_else(|| find_column_opt(&headers, "Part"));
    let animal_idx =
        find_column_opt(&headers, "Animal").or_else(|| find_column_opt(&headers, "Number"));
    let stage_idx = find_column_opt(&headers, "Stage");
    let perimeter_idx = find_column_opt(&headers, "Perimeter");
    let change_ref_idx = find_column_opt(&headers, "ChangeReference");
    let change_ref_img_idx = find_column_opt(&headers, "ChangeReferenceImageName");

    let animal_idx = animal_idx.ok_or_else(|| {
        BekiperError::Data(format!(
            "{}: metadata CSV must have an 'Animal' or 'Number' column",
            path.display()
        ))
    })?;

    if perimeter_idx.is_none() && change_ref_idx.is_none() {
        return Err(BekiperError::Data(format!(
            "{}: metadata CSV must have a 'Perimeter' or 'ChangeReference' column",
            path.display()
        )));
    }

    let mut rows = Vec::new();
    // Forward-fill values for Phase and PhasePart
    let mut current_phase = String::new();
    let mut current_phase_part: u32 = 0;

    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;

        // Forward-fill Phase (optional)
        if let Some(idx) = phase_idx {
            let phase_str = record.get(idx).unwrap_or("").trim();
            if !phase_str.is_empty() {
                current_phase = phase_str.to_string();
            }
        }

        // Forward-fill PhasePart / Part (optional)
        if let Some(idx) = phase_part_idx {
            let pp_str = record.get(idx).unwrap_or("").trim();
            if let Ok(pp) = pp_str.parse::<u32>() {
                current_phase_part = pp;
            }
        }

        let animal_str = record.get(animal_idx).unwrap_or("").trim();
        let animal: u32 = match animal_str.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Stage is optional (default 0 for Y-maze)
        let stage: u32 = stage_idx
            .and_then(|idx| record.get(idx)?.trim().parse().ok())
            .unwrap_or(0);

        // Resolve perimeter ID: prefer Perimeter column, fall back to ChangeReference.
        // If ChangeReferenceImageName is present, the perimeter needs re-referencing
        // regardless of which column holds the ID (metadata sometimes puts the ID
        // in the Perimeter column even when re-referencing is intended).
        let perimeter_id = perimeter_idx.and_then(|i| {
            let v = record.get(i).unwrap_or("").trim();
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        });
        let change_ref_id = change_ref_idx.and_then(|i| {
            let v = record.get(i).unwrap_or("").trim();
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        });
        let target_image = change_ref_img_idx.and_then(|i| {
            let v = record.get(i).unwrap_or("").trim();
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        });

        // Pick the perimeter ID from whichever column has it
        let id = perimeter_id.or(change_ref_id);

        let source = id.map(|id| {
            if let Some(target) = target_image {
                // ChangeReferenceImageName present → needs re-referencing
                PerimeterSource::ReReferenced {
                    perimeter_id: id,
                    target_image: target,
                }
            } else {
                // No change-reference image → use perimeter directly
                PerimeterSource::Direct { perimeter_id: id }
            }
        });

        rows.push(MetadataRow {
            phase: current_phase.clone(),
            phase_part: current_phase_part,
            animal,
            stage,
            source,
        });
    }

    Ok(rows)
}

/// Build a lookup: (animal, stage, phase_part) → PerimeterSource from metadata rows.
pub fn build_metadata_lookup(rows: &[MetadataRow]) -> HashMap<(u32, u32, u32), PerimeterSource> {
    rows.iter()
        .filter_map(|r| {
            r.source
                .as_ref()
                .map(|src| ((r.animal, r.stage, r.phase_part), src.clone()))
        })
        .collect()
}

// ── Video-map CSV ───────────────────────────────────────────────────────────

/// Read an explicit video-to-perimeter mapping CSV.
///
/// Format (with header): `video_number,perimeter_id`
pub fn read_video_map(path: &Path) -> Result<HashMap<u32, String>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let mut map = HashMap::new();

    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        if record.len() < 2 {
            continue;
        }
        let video_number: u32 = record[0]
            .trim()
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: video_number: {e}", path.display())))?;
        let perimeter_id = record[1].trim().to_string();
        if !perimeter_id.is_empty() {
            map.insert(video_number, perimeter_id);
        }
    }

    Ok(map)
}

// ── High-level resolver ─────────────────────────────────────────────────────

/// Annotation directory resolver.
///
/// Loads metadata once, then resolves perimeters for individual experiments.
/// Supports both direct perimeters and re-referenced perimeters (translated
/// using reference point offsets).
pub struct AnnotationResolver {
    canonical: Option<canonical::CanonicalAnnotationStore>,
    plugin_dir: PathBuf,
    /// (animal, stage, phase_part) → PerimeterSource  (from metadata.csv)
    metadata_lookup: Option<HashMap<(u32, u32, u32), PerimeterSource>>,
    /// video_number → perimeter_id  (from video_map.csv, simple fallback)
    simple_lookup: Option<HashMap<u32, String>>,
}
