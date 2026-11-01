impl RadialMazeAsset {
    pub fn load(dir: &Path) -> Result<Self> {
        let arm_lines_path = dir.join("arm-lines.csv");
        let center_path = dir.join("center-triangle.json");

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(&arm_lines_path)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;

        let mut arm_lines = Vec::new();
        for record in reader.records() {
            let record = record
                .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
            if record.len() < 5 {
                continue;
            }
            arm_lines.push(ArmLineRow {
                label: record[0].to_string(),
                x1: record[1].trim().parse().map_err(|e| {
                    BekiperError::Data(format!("{}: x1: {e}", arm_lines_path.display()))
                })?,
                y1: record[2].trim().parse().map_err(|e| {
                    BekiperError::Data(format!("{}: y1: {e}", arm_lines_path.display()))
                })?,
                x2: record[3].trim().parse().map_err(|e| {
                    BekiperError::Data(format!("{}: x2: {e}", arm_lines_path.display()))
                })?,
                y2: record[4].trim().parse().map_err(|e| {
                    BekiperError::Data(format!("{}: y2: {e}", arm_lines_path.display()))
                })?,
                extras: record.iter().skip(5).map(ToOwned::to_owned).collect(),
            });
        }

        let center_triangle = if center_path.exists() {
            let content = std::fs::read_to_string(&center_path)?;
            serde_json::from_str(&content)
                .map_err(|e| BekiperError::Data(format!("{}: {e}", center_path.display())))?
        } else {
            json!({
                "annotations": [],
                "categories": [{ "id": 1, "name": "center" }],
            })
        };

        Ok(Self {
            dir: dir.to_path_buf(),
            arm_lines,
            center_triangle,
        })
    }

    pub fn to_specs(&self) -> Result<Vec<PerimeterSpec>> {
        parse_radial_maze(&self.dir)
    }

    pub fn source_image_name(&self) -> Option<&str> {
        self.arm_lines
            .first()
            .and_then(|row| row.extras.first())
            .map(String::as_str)
    }

    pub fn replace_from_spec(&mut self, spec: &PerimeterSpec) -> Result<()> {
        let PerimeterSpec::RadialMaze {
            center_vertices,
            arms,
            ..
        } = spec
        else {
            return Err(BekiperError::Data(
                "radial maze asset can only be replaced from a radial_maze perimeter".into(),
            ));
        };

        let arm_template = self.arm_lines.first().cloned();
        self.arm_lines = arms
            .iter()
            .enumerate()
            .map(|(index, arm)| {
                let template = self.arm_lines.get(index).or(arm_template.as_ref());
                let label = template
                    .map(|row| row.label.clone())
                    .unwrap_or_else(|| ((b'a' + index as u8) as char).to_string());
                let extras = template.map(|row| row.extras.clone()).unwrap_or_default();
                let ((x1, y1), (x2, y2)) =
                    radial_arm_distal_edge(center_vertices, arm).unwrap_or_default();
                ArmLineRow {
                    label,
                    x1,
                    y1,
                    x2,
                    y2,
                    extras,
                }
            })
            .collect();

        let image_id = self
            .center_triangle
            .get("images")
            .and_then(Value::as_array)
            .and_then(|images| images.first())
            .and_then(|image| image.get("id"))
            .and_then(Value::as_u64)
            .unwrap_or(1);

        let flat: Vec<f64> = canonicalize_vertices(center_vertices.clone())
            .iter()
            .flat_map(|(x, y)| [*x, *y])
            .collect();
        let (x, y, width, height) = vertices_bounding_box(center_vertices);
        self.center_triangle["annotations"] = Value::Array(vec![json!({
            "id": 0,
            "iscrowd": 0,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [flat],
            "bbox": [x, y, width, height],
            "area": polygon_area(center_vertices).abs(),
        })]);
        self.center_triangle["categories"] = Value::Array(vec![json!({
            "id": 1,
            "name": "center",
        })]);

        Ok(())
    }

    pub fn write(&self) -> Result<()> {
        let arm_lines_path = self.dir.join("arm-lines.csv");
        let center_path = self.dir.join("center-triangle.json");

        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_path(&arm_lines_path)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
        for row in &self.arm_lines {
            let mut record = vec![
                row.label.clone(),
                format_number(row.x1),
                format_number(row.y1),
                format_number(row.x2),
                format_number(row.y2),
            ];
            record.extend(row.extras.clone());
            writer
                .write_record(&record)
                .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
        }
        writer
            .flush()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;

        let content = serde_json::to_string_pretty(&self.center_triangle)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", center_path.display())))?;
        std::fs::write(center_path, content)?;
        Ok(())
    }
}

impl EditablePerimeterAsset {
    pub fn load(plugin_dir: &Path, perimeter_id: &str) -> Result<Self> {
        let csv_path = plugin_dir.join(format!("perimeter-rectangle-{perimeter_id}.csv"));
        if csv_path.exists() {
            return Ok(Self::RectangleCsv {
                perimeter_id: perimeter_id.to_string(),
                asset: RectangleCsvAsset::load(&csv_path)?,
            });
        }

        let json_path = plugin_dir.join(format!("perimeter-rectangle-{perimeter_id}.json"));
        if json_path.exists() {
            return Ok(Self::CocoJson {
                perimeter_id: perimeter_id.to_string(),
                asset: CocoJsonAsset::load(&json_path)?,
            });
        }

        for name in [
            format!("perimeter-circle_point-{perimeter_id}.csv"),
            format!("perimeter-circle-point-{perimeter_id}.csv"),
        ] {
            let path = plugin_dir.join(name);
            if path.exists() {
                return Ok(Self::CircleCsv {
                    perimeter_id: perimeter_id.to_string(),
                    asset: CircleCsvAsset::load(&path, CircleCsvKind::CenterPoint)?,
                });
            }
        }

        let circle_path = plugin_dir.join(format!("perimeter-circle-{perimeter_id}.csv"));
        if circle_path.exists() {
            return Ok(Self::CircleCsv {
                perimeter_id: perimeter_id.to_string(),
                asset: CircleCsvAsset::load(&circle_path, CircleCsvKind::EdgePoint)?,
            });
        }

        let radial_maze_dir = plugin_dir.join(format!("radial-maze_{perimeter_id}"));
        if radial_maze_dir.is_dir() {
            return Ok(Self::RadialMaze {
                perimeter_id: perimeter_id.to_string(),
                asset: RadialMazeAsset::load(&radial_maze_dir)?,
            });
        }

        Err(BekiperError::Data(format!(
            "unsupported or missing editable perimeter asset '{perimeter_id}' in {}",
            plugin_dir.display()
        )))
    }

    pub fn perimeter_id(&self) -> &str {
        match self {
            Self::RectangleCsv { perimeter_id, .. }
            | Self::CircleCsv { perimeter_id, .. }
            | Self::CocoJson { perimeter_id, .. }
            | Self::RadialMaze { perimeter_id, .. }
            | Self::Canonical { perimeter_id, .. } => perimeter_id,
        }
    }

    pub fn family(&self) -> PerimeterAssetFamily {
        match self {
            Self::RectangleCsv { .. } => PerimeterAssetFamily::RectangleCsv,
            Self::CircleCsv { .. } => PerimeterAssetFamily::CircleCsv,
            Self::CocoJson { .. } => PerimeterAssetFamily::CocoJson,
            Self::RadialMaze { .. } => PerimeterAssetFamily::RadialMaze,
            Self::Canonical { .. } => PerimeterAssetFamily::Canonical,
        }
    }

    pub fn source_image_name(&self) -> Option<&str> {
        match self {
            Self::RectangleCsv { asset, .. } => asset.source_image_name(),
            Self::CircleCsv { asset, .. } => asset.source_image_name(),
            Self::CocoJson { asset, .. } => asset.source_image_name(),
            Self::RadialMaze { asset, .. } => asset.source_image_name(),
            Self::Canonical { .. } => None,
        }
    }

    pub fn to_specs(&self) -> Result<Vec<PerimeterSpec>> {
        match self {
            Self::RectangleCsv { asset, .. } => Ok(asset.to_specs()),
            Self::CircleCsv { asset, .. } => Ok(asset.to_specs()),
            Self::CocoJson { asset, .. } => asset.to_specs(),
            Self::RadialMaze { asset, .. } => asset.to_specs(),
            Self::Canonical { specs, .. } => Ok(specs.clone()),
        }
    }

    pub fn replace_from_specs(&mut self, specs: &[PerimeterSpec]) -> Result<()> {
        match self {
            Self::RectangleCsv { asset, .. } => asset.replace_from_specs(specs),
            Self::CircleCsv { asset, .. } => asset.replace_from_specs(specs),
            Self::CocoJson { asset, .. } => asset.replace_from_specs(specs),
            Self::RadialMaze { asset, .. } => {
                if specs.len() != 1 {
                    return Err(BekiperError::Data(
                        "radial maze assets expect exactly one perimeter spec".into(),
                    ));
                }
                asset.replace_from_spec(&specs[0])
            }
            Self::Canonical { specs: stored, .. } => {
                *stored = specs.to_vec();
                Ok(())
            }
        }
    }

    pub fn write(&self) -> Result<()> {
        match self {
            Self::RectangleCsv { asset, .. } => asset.write(),
            Self::CircleCsv { asset, .. } => asset.write(),
            Self::CocoJson { asset, .. } => asset.write(),
            Self::RadialMaze { asset, .. } => asset.write(),
            Self::Canonical {
                perimeter_id,
                store,
                specs,
                ..
            } => {
                store.write_perimeter(perimeter_id, specs)?;
                store.write_sync_state()
            }
        }
    }
}

impl ReferenceDocument {
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(path)
            .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

        let mut rows = Vec::new();
        for record in reader.records() {
            let record =
                record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
            if record.len() < 4 {
                continue;
            }
            if record[1].trim().parse::<f64>().is_err() {
                continue;
            }
            rows.push(ReferenceRow {
                label: record[0].to_string(),
                x: record[1]
                    .trim()
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: x: {e}", path.display())))?,
                y: record[2]
                    .trim()
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: y: {e}", path.display())))?,
                image_name: record[3].to_string(),
                extras: record.iter().skip(4).map(ToOwned::to_owned).collect(),
            });
        }

        Ok(Self {
            path: path.to_path_buf(),
            rows,
        })
    }

    pub fn load_or_empty(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self {
                path: path.to_path_buf(),
                rows: Vec::new(),
            })
        }
    }

    pub fn as_reference_point(&self, index: usize) -> Option<ReferencePoint> {
        self.rows.get(index).map(|row| ReferencePoint {
            x: row.x,
            y: row.y,
            image_name: row.image_name.clone(),
        })
    }

    pub fn first_point(&self) -> Option<ReferencePoint> {
        self.as_reference_point(0)
    }

    pub fn find_row_by_stem(&self, stem: &str) -> Option<usize> {
        self.rows
            .iter()
            .position(|row| strip_png_extension(&row.image_name) == stem)
    }

    pub fn upsert_point(
        &mut self,
        image_stem: &str,
        x: f64,
        y: f64,
        template: Option<&ReferenceRow>,
    ) -> usize {
        let image_name = ensure_png_extension(image_stem);
        if let Some(index) = self.find_row_by_stem(image_stem) {
            if let Some(row) = self.rows.get_mut(index) {
                row.x = x;
                row.y = y;
                row.image_name = image_name;
            }
            return index;
        }

        self.rows.push(ReferenceRow {
            label: template
                .map(|row| row.label.clone())
                .unwrap_or_else(|| "reference".into()),
            x,
            y,
            image_name,
            extras: template.map(|row| row.extras.clone()).unwrap_or_default(),
        });
        self.rows.len() - 1
    }

    pub fn write(&self) -> Result<()> {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_path(&self.path)
            .map_err(|e| {
                BekiperError::Data(format!("failed to write {}: {e}", self.path.display()))
            })?;
        for row in &self.rows {
            let mut record = vec![
                row.label.clone(),
                format_number(row.x),
                format_number(row.y),
                row.image_name.clone(),
            ];
            record.extend(row.extras.clone());
            writer
                .write_record(&record)
                .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))?;
        }
        writer
            .flush()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))
    }
}
