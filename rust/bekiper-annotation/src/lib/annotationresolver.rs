impl AnnotationResolver {
    /// Create a resolver from an annotation directory.
    ///
    /// Looks for `video_map.csv` first (simple mapping), then falls back to
    /// `metadata.csv` (compound key: animal + stage + phase_part).
    pub fn from_dir(annotation_dir: &Path) -> Result<Self> {
        let canonical = canonical::CanonicalAnnotationStore::open(annotation_dir)?;
        let plugin_dir = annotation_dir.join("plugin_files");
        if !plugin_dir.is_dir() && canonical.is_none() {
            return Err(BekiperError::Data(format!(
                "annotation plugin_files directory not found: {}",
                plugin_dir.display()
            )));
        }

        // Prefer explicit video_map.csv
        let video_map_path = annotation_dir.join("video_map.csv");
        if video_map_path.exists() {
            let lookup = read_video_map(&video_map_path)?;
            tracing::info!(
                count = lookup.len(),
                "loaded video_map.csv annotation mapping"
            );
            return Ok(Self {
                canonical,
                plugin_dir,
                metadata_lookup: None,
                simple_lookup: Some(lookup),
            });
        }

        // Fall back to metadata.csv
        let metadata_path = annotation_dir.join("metadata.csv");
        if metadata_path.exists() {
            let rows = read_metadata_csv(&metadata_path)?;
            let lookup = build_metadata_lookup(&rows);
            tracing::info!(
                count = lookup.len(),
                total_rows = rows.len(),
                "loaded metadata.csv annotation mapping"
            );
            return Ok(Self {
                canonical,
                plugin_dir,
                metadata_lookup: Some(lookup),
                simple_lookup: None,
            });
        }

        if canonical.is_some() {
            return Ok(Self {
                canonical,
                plugin_dir,
                metadata_lookup: None,
                simple_lookup: None,
            });
        }

        Err(BekiperError::Data(format!(
            "no video_map.csv or metadata.csv found in {}",
            annotation_dir.display()
        )))
    }

    /// Resolve perimeters for a video using its full context (animal, stage, phase_part).
    ///
    /// For direct perimeters, loads coordinates as-is.
    /// For re-referenced perimeters, loads the base coordinates and translates them
    /// using the offset between the source and target reference points.
    ///
    /// Falls back to simple video_number lookup if using video_map.csv.
    /// Returns `None` if no mapping exists.
    pub fn resolve(&self, ctx: &VideoContext) -> Result<Option<Vec<PerimeterSpec>>> {
        if let Some(canonical) = &self.canonical
            && let Some(specs) = canonical.resolve(ctx)?
        {
            return Ok(Some(specs));
        }

        // Simple lookup path (video_map.csv)
        if let Some(ref lookup) = self.simple_lookup {
            return match lookup.get(&ctx.animal) {
                Some(id) => {
                    let specs = load_perimeter_by_id(&self.plugin_dir, id)?;
                    Ok(Some(specs))
                }
                None => Ok(None),
            };
        }

        // Metadata lookup path
        let source = match self
            .metadata_lookup
            .as_ref()
            .and_then(|l| l.get(&(ctx.animal, ctx.stage, ctx.phase_part)))
        {
            Some(s) => s,
            None => return Ok(None),
        };

        match source {
            PerimeterSource::Direct { perimeter_id } => {
                let specs = load_perimeter_by_id(&self.plugin_dir, perimeter_id)?;
                Ok(Some(specs))
            }
            PerimeterSource::ReReferenced {
                perimeter_id,
                target_image,
            } => {
                let mut specs = load_perimeter_by_id(&self.plugin_dir, perimeter_id)?;

                // Load source reference point (from annotation image)
                let source_ref = load_source_reference(&self.plugin_dir, perimeter_id)?;

                // Check if target matches the source image (no translation needed)
                let source_stem = source_ref
                    .image_name
                    .strip_suffix(".png")
                    .unwrap_or(&source_ref.image_name);
                if target_image == source_stem {
                    tracing::debug!(
                        perimeter_id,
                        target_image,
                        "target matches source image, no re-referencing needed"
                    );
                } else {
                    // Load target reference points and translate
                    let change_refs = load_change_references(&self.plugin_dir, perimeter_id)?;
                    if let Some(target_ref) = change_refs.get(target_image) {
                        let dx = target_ref.x - source_ref.x;
                        let dy = target_ref.y - source_ref.y;
                        translate_perimeters(&mut specs, dx, dy);
                        tracing::debug!(
                            perimeter_id,
                            target_image,
                            dx,
                            dy,
                            "re-referenced perimeters"
                        );
                    } else {
                        tracing::warn!(
                            perimeter_id,
                            target_image,
                            "no change_reference entry found for target image, using untranslated coordinates"
                        );
                    }
                }

                Ok(Some(specs))
            }
        }
    }

    /// Like [`AnnotationResolver::resolve`], but falls back to the source (untranslated) perimeter
    /// when the metadata has no entry for this video AND the annotation dir
    /// contains exactly one shared perimeter plugin (e.g. a single Y-maze).
    ///
    /// Returns `(specs, fallback)` where `fallback` is true when the source
    /// perimeter was used instead of a metadata-resolved one.
    pub fn resolve_with_fallback(
        &self,
        ctx: &VideoContext,
    ) -> Result<Option<(Vec<PerimeterSpec>, bool)>> {
        // Try exact lookup first
        if let Some(specs) = self.resolve(ctx)? {
            return Ok(Some((specs, false)));
        }

        // Fallback: if exactly one perimeter plugin exists, use it untranslated
        if let Some(specs) = self.sole_perimeter()? {
            tracing::info!(
                animal = ctx.animal,
                stage = ctx.stage,
                phase_part = ctx.phase_part,
                "no metadata entry — using shared source perimeter as fallback"
            );
            return Ok(Some((specs, true)));
        }

        Ok(None)
    }

    /// If the plugin dir contains exactly one perimeter (one radial-maze dir,
    /// or one rectangle CSV, etc.), load and return it.  Returns `None` when
    /// there are zero or more than one.
    fn sole_perimeter(&self) -> Result<Option<Vec<PerimeterSpec>>> {
        let mut found: Vec<PathBuf> = Vec::new();

        for entry in std::fs::read_dir(&self.plugin_dir)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", self.plugin_dir.display())))?
        {
            let entry = entry
                .map_err(|e| BekiperError::Data(format!("{}: {e}", self.plugin_dir.display())))?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if (name_str.starts_with("radial-maze_") && entry.path().is_dir())
                || (name_str.starts_with("perimeter-rectangle-")
                    && (name_str.ends_with(".csv") || name_str.ends_with(".json")))
            {
                found.push(entry.path());
            }
        }

        if found.len() != 1 {
            return Ok(None);
        }

        let path = &found[0];
        if path.is_dir() {
            Ok(Some(parse_radial_maze(path)?))
        } else if path.extension().is_some_and(|e| e == "json") {
            Ok(Some(parse_coco_json(path)?))
        } else {
            Ok(Some(parse_rectangle_csv(path)?))
        }
    }

    /// Load scale calibrations from `meters_per_pixel-*.csv` files in the plugin dir.
    pub fn scale_calibrations(&self) -> Result<Vec<ScaleCalibration>> {
        load_scale_calibrations(&self.plugin_dir)
    }
}

/// Extract the video number from a video filename.
///
/// Looks for patterns like `video-10.mp4`, `video_10.mp4`, `Test 10.szv`.
/// Returns `None` if no number can be extracted.
pub fn extract_video_number(video_path: &Path) -> Option<u32> {
    let stem = video_path.file_stem()?.to_str()?;

    // Try "video-N" or "video_N" pattern
    if let Some(rest) = stem
        .strip_prefix("video-")
        .or_else(|| stem.strip_prefix("video_"))
    {
        return rest.parse().ok();
    }

    // Try suffix pattern: "something.video-N" or "something.video_N"
    for part in stem.rsplit('.') {
        if let Some(rest) = part
            .strip_prefix("video-")
            .or_else(|| part.strip_prefix("video_"))
        {
            return rest.parse().ok();
        }
    }

    // Try "prefix.T1.configA.video-N" pattern (split by '.')
    for segment in stem.split('.') {
        if let Some(rest) = segment
            .strip_prefix("video-")
            .or_else(|| segment.strip_prefix("video_"))
        {
            return rest.parse().ok();
        }
    }

    // Try "Test N" pattern
    if let Some(rest) = stem.strip_prefix("Test ") {
        return rest.trim().parse().ok();
    }

    // Last resort: trailing number
    let trailing: String = stem
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if !trailing.is_empty() {
        let num_str: String = trailing.chars().rev().collect();
        return num_str.parse().ok();
    }

    None
}

// ── COCO JSON types ─────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct CocoAnnotation {
    #[serde(default)]
    annotations: Vec<CocoAnn>,
    #[serde(default)]
    categories: Vec<CocoCategory>,
}

#[derive(Debug, serde::Deserialize)]
struct CocoAnn {
    category_id: u64,
    #[serde(default)]
    segmentation: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    bbox: Vec<f64>,
}

#[derive(Debug, serde::Deserialize)]
struct CocoCategory {
    id: u64,
    name: String,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn find_column_opt(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    headers.iter().position(|h| h.trim() == name)
}
