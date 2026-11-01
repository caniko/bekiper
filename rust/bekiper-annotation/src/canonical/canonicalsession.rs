impl CanonicalSession {
    pub fn matches_context(&self, ctx: &VideoContext) -> bool {
        if let (Some(animal), Some(stage), Some(phase_part)) =
            (self.animal, self.stage, self.phase_part)
        {
            return animal == ctx.animal && stage == ctx.stage && phase_part == ctx.phase_part;
        }
        self.video_number == Some(ctx.animal)
    }

    pub fn group_key(&self) -> Option<AnnotationGroupKey> {
        let target_image = self.target_image.clone()?;
        Some(AnnotationGroupKey {
            source_perimeter_id: self
                .source_perimeter_id
                .clone()
                .unwrap_or_else(|| self.perimeter_id.clone()),
            target_image,
        })
    }

    pub fn correction_kind(&self) -> CanonicalCorrectionKind {
        if self.coordinate_space == CoordinateSpace::Source {
            return CanonicalCorrectionKind::SourceShared;
        }
        let Some(key) = self.group_key() else {
            return CanonicalCorrectionKind::LegacyTarget;
        };
        if self.perimeter_id
            == group_target_perimeter_id(&key.source_perimeter_id, &key.target_image)
        {
            CanonicalCorrectionKind::GroupTarget
        } else if self.perimeter_id
            == individual_target_perimeter_id(
                &key.source_perimeter_id,
                &key.target_image,
                &self.key,
            )
        {
            CanonicalCorrectionKind::IndividualTarget
        } else {
            CanonicalCorrectionKind::LegacyTarget
        }
    }
}

impl CanonicalAnnotationStore {
    fn export_session_mappings(&self) -> Result<()> {
        let sessions = self.load_sessions()?;
        if sessions.is_empty() {
            return Ok(());
        }

        let metadata_path = self.annotation_dir.join("metadata.csv");
        if metadata_path.exists() {
            export_metadata_mappings(&metadata_path, &sessions)?;
        }

        let video_map_path = self.annotation_dir.join("video_map.csv");
        if video_map_path.exists() {
            export_video_map_mappings(&video_map_path, &sessions)?;
        }

        Ok(())
    }
}

fn change_state(annotation_dir: &Path) -> Result<(bool, bool)> {
    let root = CanonicalAnnotationStore::path_for(annotation_dir);
    if !root.join("sync_state.toml").exists() {
        return Ok((true, root.exists()));
    }
    let content = fs::read_to_string(root.join("sync_state.toml"))?;
    let state: SyncState = toml::from_str(&content)
        .map_err(|e| BekiperError::Config(format!("invalid sync_state.toml: {e}")))?;
    Ok((
        state.legacy_fingerprints != legacy_fingerprints(annotation_dir)?,
        state.canonical_fingerprints != canonical_fingerprints(&root)?,
    ))
}

fn export_video_map_mappings(path: &Path, sessions: &[CanonicalSession]) -> Result<()> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;
    let headers = reader
        .headers()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?
        .clone();
    let number_index = headers
        .iter()
        .position(|header| header.trim() == "video_number")
        .ok_or_else(|| BekiperError::Data(format!("{}: missing video_number", path.display())))?;
    let id_index = headers
        .iter()
        .position(|header| header.trim() == "perimeter_id")
        .ok_or_else(|| BekiperError::Data(format!("{}: missing perimeter_id", path.display())))?;
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        let mut row = record.iter().map(ToOwned::to_owned).collect::<Vec<_>>();
        let video_number = row
            .get(number_index)
            .map(String::as_str)
            .map(str::trim)
            .and_then(|value| value.parse::<u32>().ok());
        if let Some(session) = sessions
            .iter()
            .filter(|session| session.coordinate_space == CoordinateSpace::Target)
            .find(|session| session.video_number == video_number)
        {
            row[id_index] = session.perimeter_id.clone();
        }
        rows.push(row);
    }
    write_csv_with_headers(path, headers.iter(), rows)
}

fn export_metadata_mappings(path: &Path, sessions: &[CanonicalSession]) -> Result<()> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;
    let headers = reader
        .headers()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?
        .clone();

    let phase_part_index = headers
        .iter()
        .position(|header| matches!(header.trim(), "PhasePart" | "Part"));
    let animal_index = headers
        .iter()
        .position(|header| matches!(header.trim(), "Animal" | "Number"));
    let stage_index = headers.iter().position(|header| header.trim() == "Stage");
    let perimeter_index = headers
        .iter()
        .position(|header| header.trim() == "Perimeter");
    let change_reference_index = headers
        .iter()
        .position(|header| header.trim() == "ChangeReference");
    let target_image_index = headers
        .iter()
        .position(|header| header.trim() == "ChangeReferenceImageName");

    let mut rows = Vec::new();
    let mut current_phase_part = 0u32;
    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        let mut row = record.iter().map(ToOwned::to_owned).collect::<Vec<_>>();
        if let Some(index) = phase_part_index
            && let Some(value) = row.get(index).map(String::as_str).map(str::trim)
            && !value.is_empty()
            && let Ok(parsed) = value.parse::<u32>()
        {
            current_phase_part = parsed;
        }
        let animal = animal_index
            .and_then(|index| row.get(index))
            .map(String::as_str)
            .map(str::trim)
            .and_then(|value| value.parse::<u32>().ok());
        let stage = stage_index
            .and_then(|index| row.get(index))
            .map(String::as_str)
            .map(str::trim)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if let Some(session) = sessions
            .iter()
            .filter(|session| session.coordinate_space == CoordinateSpace::Target)
            .find(|session| {
                session.animal == animal
                    && session.stage == Some(stage)
                    && session.phase_part == Some(current_phase_part)
            })
        {
            if let Some(index) = perimeter_index {
                row[index] = session.perimeter_id.clone();
                if let Some(index) = change_reference_index {
                    row[index].clear();
                }
            } else if let Some(index) = change_reference_index {
                row[index] = session.perimeter_id.clone();
            }
            if let Some(index) = target_image_index {
                row[index].clear();
            }
        }
        rows.push(row);
    }
    write_csv_with_headers(path, headers.iter(), rows)
}

fn write_csv_with_headers<'a>(
    path: &Path,
    headers: impl Iterator<Item = &'a str>,
    rows: Vec<Vec<String>>,
) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to write {}: {e}", path.display())))?;
    writer
        .write_record(headers)
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
    }
    writer
        .flush()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))
}

fn import_reference_document(plugin_dir: &Path, perimeter_id: &str) -> Result<ReferenceDocument> {
    let source = match load_source_reference(plugin_dir, perimeter_id) {
        Ok(point) => Some(point.into()),
        Err(_) => None,
    };
    let targets = load_change_references(plugin_dir, perimeter_id)?
        .into_iter()
        .map(|(key, point)| (key, point.into()))
        .collect();
    Ok(ReferenceDocument { source, targets })
}

fn legacy_fingerprints(annotation_dir: &Path) -> Result<Vec<FileFingerprint>> {
    let mut roots = Vec::new();
    for name in ["metadata.csv", "video_map.csv"] {
        let path = annotation_dir.join(name);
        if path.exists() {
            roots.push(path);
        }
    }
    let plugin_dir = annotation_dir.join("plugin_files");
    if plugin_dir.exists() {
        roots.push(plugin_dir);
    }
    fingerprints(annotation_dir, &roots)
}

fn canonical_fingerprints(root: &Path) -> Result<Vec<FileFingerprint>> {
    let mut roots = Vec::new();
    for name in ["manifest.toml", "sessions", "perimeters", "references"] {
        let path = root.join(name);
        if path.exists() {
            roots.push(path);
        }
    }
    fingerprints(root, &roots)
}

fn fingerprints(base: &Path, roots: &[PathBuf]) -> Result<Vec<FileFingerprint>> {
    let mut files = Vec::new();
    for root in roots {
        collect_files(root, &mut files)?;
    }
    files.sort();
    files
        .into_iter()
        .map(|path| fingerprint_file(base, &path))
        .collect()
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            collect_files(&entry?.path(), files)?;
        }
    }
    Ok(())
}

fn fingerprint_file(base: &Path, path: &Path) -> Result<FileFingerprint> {
    let meta = fs::metadata(path)?;
    let modified_unix_s = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    Ok(FileFingerprint {
        path: path
            .strip_prefix(base)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/"),
        len: meta.len(),
        modified_unix_s,
    })
}

fn sole_legacy_perimeter_id(plugin_dir: &Path) -> Result<Option<String>> {
    if !plugin_dir.exists() {
        return Ok(None);
    }
    let mut ids = Vec::new();
    for entry in fs::read_dir(plugin_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(id) = name.strip_prefix("perimeter-rectangle-").and_then(|value| {
            value
                .strip_suffix(".csv")
                .or_else(|| value.strip_suffix(".json"))
        }) {
            ids.push(id.to_string());
        } else if let Some(id) = name.strip_prefix("radial-maze_")
            && entry.path().is_dir()
        {
            ids.push(id.to_string());
        }
    }
    ids.sort();
    ids.dedup();
    Ok((ids.len() == 1).then(|| ids.remove(0)))
}

fn write_rectangle_csv(path: &Path, perimeters: &[PerimeterSpec]) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to write {}: {e}", path.display())))?;
    for spec in perimeters {
        let (label, x, y, width, height) = bounding_rect_from_spec(spec);
        writer
            .write_record([
                label,
                format_number(x),
                format_number(y),
                format_number(width),
                format_number(height),
            ])
            .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
    }
    writer
        .flush()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))
}

fn write_circle_csv(path: &Path, perimeters: &[PerimeterSpec]) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to write {}: {e}", path.display())))?;
    for (index, spec) in perimeters.iter().enumerate() {
        let PerimeterSpec::Circle {
            center_x,
            center_y,
            radius,
            ..
        } = spec
        else {
            continue;
        };
        writer
            .write_record([
                (index + 1).to_string(),
                format_number(*center_x),
                format_number(*center_y),
                format_number(center_x + radius),
                format_number(*center_y),
            ])
            .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
    }
    writer
        .flush()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))
}

fn write_coco_json(path: &Path, perimeters: &[PerimeterSpec]) -> Result<()> {
    let mut categories = Vec::new();
    let mut annotations = Vec::new();
    for (index, spec) in perimeters.iter().enumerate() {
        let (label, x, y, width, height) = bounding_rect_from_spec(spec);
        let category_id = (index + 1) as u64;
        categories.push(json!({ "id": category_id, "name": label }));
        let vertices = segmentation_vertices_from_spec(spec, x, y, width, height);
        let flat = vertices
            .iter()
            .flat_map(|(vx, vy)| [*vx, *vy])
            .collect::<Vec<_>>();
        annotations.push(json!({
            "id": index as u64,
            "iscrowd": 0,
            "image_id": 1,
            "category_id": category_id,
            "segmentation": [flat],
            "bbox": [x, y, width, height],
            "area": polygon_area(&vertices).abs().max(width * height),
        }));
    }
    let content = serde_json::to_string_pretty(&json!({
        "images": [{ "id": 1 }],
        "annotations": annotations,
        "categories": categories,
    }))
    .map_err(|e| BekiperError::Data(format!("failed to serialize {}: {e}", path.display())))?;
    fs::write(path, content)?;
    Ok(())
}
