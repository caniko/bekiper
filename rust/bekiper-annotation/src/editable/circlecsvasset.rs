impl CircleCsvAsset {
    pub fn load(path: &Path, kind: CircleCsvKind) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(path)
            .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

        let mut header = None;
        let mut rows = Vec::new();

        for record in reader.records() {
            let record =
                record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
            let min_fields = match kind {
                CircleCsvKind::EdgePoint => 5,
                CircleCsvKind::CenterPoint => 3,
            };
            if record.len() < min_fields {
                continue;
            }
            if record[1].trim().parse::<f64>().is_err() {
                header = Some(record.iter().map(ToOwned::to_owned).collect());
                continue;
            }

            let raw_label = record[0].trim();
            let label = raw_label
                .parse::<u32>()
                .map(|index| format!("object_{index}"))
                .unwrap_or_else(|_| raw_label.to_string());
            let center_x: f64 = record[1]
                .trim()
                .parse()
                .map_err(|e| BekiperError::Data(format!("{}: center_x: {e}", path.display())))?;
            let center_y: f64 = record[2]
                .trim()
                .parse()
                .map_err(|e| BekiperError::Data(format!("{}: center_y: {e}", path.display())))?;
            let (radius, extras) = match kind {
                CircleCsvKind::EdgePoint => {
                    let edge_x: f64 = record[3].trim().parse().map_err(|e| {
                        BekiperError::Data(format!("{}: edge_x: {e}", path.display()))
                    })?;
                    let edge_y: f64 = record[4].trim().parse().map_err(|e| {
                        BekiperError::Data(format!("{}: edge_y: {e}", path.display()))
                    })?;
                    (
                        ((edge_x - center_x).powi(2) + (edge_y - center_y).powi(2)).sqrt(),
                        record.iter().skip(5).map(ToOwned::to_owned).collect(),
                    )
                }
                CircleCsvKind::CenterPoint => {
                    (0.0, record.iter().skip(3).map(ToOwned::to_owned).collect())
                }
            };
            rows.push(CircleCsvRow {
                label,
                center_x,
                center_y,
                radius,
                extras,
            });
        }

        if rows.is_empty() {
            return Err(BekiperError::Data(format!(
                "{}: no circle rows found",
                path.display()
            )));
        }

        Ok(Self {
            path: path.to_path_buf(),
            kind,
            header,
            rows,
        })
    }

    pub fn to_specs(&self) -> Vec<PerimeterSpec> {
        self.rows
            .iter()
            .map(|row| PerimeterSpec::Circle {
                label: row.label.clone(),
                center_x: row.center_x,
                center_y: row.center_y,
                radius: row.radius,
            })
            .collect()
    }

    pub fn replace_from_specs(&mut self, specs: &[PerimeterSpec]) -> Result<()> {
        if specs.is_empty() {
            return Err(BekiperError::Data(
                "cannot replace circle CSV with no specs".into(),
            ));
        }

        let template = self.rows.first().cloned();
        self.rows = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let (label, center_x, center_y, radius) = circle_from_spec(spec);
                let extras = self
                    .rows
                    .get(index)
                    .or(template.as_ref())
                    .map(|row| row.extras.clone())
                    .unwrap_or_default();
                CircleCsvRow {
                    label,
                    center_x,
                    center_y,
                    radius,
                    extras,
                }
            })
            .collect();

        Ok(())
    }

    pub fn source_image_name(&self) -> Option<&str> {
        self.rows
            .first()
            .and_then(|row| match self.kind {
                CircleCsvKind::EdgePoint => row.extras.first(),
                CircleCsvKind::CenterPoint => row.extras.first(),
            })
            .map(String::as_str)
    }

    pub fn write(&self) -> Result<()> {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_path(&self.path)
            .map_err(|e| {
                BekiperError::Data(format!("failed to write {}: {e}", self.path.display()))
            })?;

        if let Some(header) = &self.header {
            writer
                .write_record(header)
                .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))?;
        }

        for row in &self.rows {
            let mut record = match self.kind {
                CircleCsvKind::EdgePoint => vec![
                    circle_edge_label(&row.label),
                    format_number(row.center_x),
                    format_number(row.center_y),
                    format_number(row.center_x + row.radius),
                    format_number(row.center_y),
                ],
                CircleCsvKind::CenterPoint => vec![
                    row.label.clone(),
                    format_number(row.center_x),
                    format_number(row.center_y),
                ],
            };
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

fn circle_edge_label(label: &str) -> String {
    label
        .strip_prefix("object_")
        .and_then(|value| value.parse::<u32>().ok())
        .map(|value| value.to_string())
        .unwrap_or_else(|| label.to_string())
}

impl CocoJsonAsset {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let document: Value = serde_json::from_str(&content)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        Ok(Self {
            path: path.to_path_buf(),
            document,
        })
    }

    pub fn to_specs(&self) -> Result<Vec<PerimeterSpec>> {
        parse_coco_json(&self.path)
    }

    pub fn source_image_name(&self) -> Option<&str> {
        self.document
            .get("images")
            .and_then(Value::as_array)
            .and_then(|images| images.first())
            .and_then(|image| image.get("file_name"))
            .and_then(Value::as_str)
    }

    pub fn replace_from_specs(&mut self, specs: &[PerimeterSpec]) -> Result<()> {
        if specs.is_empty() {
            return Err(BekiperError::Data(
                "cannot replace COCO JSON with no specs".into(),
            ));
        }

        let image_id = self
            .document
            .get("images")
            .and_then(Value::as_array)
            .and_then(|images| images.first())
            .and_then(|image| image.get("id"))
            .and_then(Value::as_u64)
            .unwrap_or(1);

        let existing_categories = self
            .document
            .get("categories")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let mut categories = Vec::new();
        let mut annotations = Vec::new();

        for (index, spec) in specs.iter().enumerate() {
            let (label, x, y, width, height) = bounding_rect_from_spec(spec);
            let vertices = segmentation_vertices_from_spec(spec, x, y, width, height);
            let category_id = existing_categories
                .iter()
                .find(|value| value.get("name").and_then(Value::as_str) == Some(label.as_str()))
                .and_then(|value| value.get("id").and_then(Value::as_u64))
                .unwrap_or((index + 1) as u64);

            if !categories
                .iter()
                .any(|value: &Value| value.get("id").and_then(Value::as_u64) == Some(category_id))
            {
                categories.push(json!({
                    "id": category_id,
                    "name": label,
                }));
            }

            let flat: Vec<f64> = vertices.iter().flat_map(|(vx, vy)| [*vx, *vy]).collect();

            annotations.push(json!({
                "id": index as u64,
                "iscrowd": 0,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [flat],
                "bbox": [x, y, width, height],
                "area": polygon_area(&vertices).abs().max(width * height),
            }));
        }

        self.document["annotations"] = Value::Array(annotations);
        self.document["categories"] = Value::Array(categories);
        Ok(())
    }

    pub fn write(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.document)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))?;
        std::fs::write(&self.path, content)?;
        Ok(())
    }
}
