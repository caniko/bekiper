// Both documents read the same flexible, trimmed CSV rows. Only metadata
// normalizes BOM-bearing headers; video-map header spelling is preserved.
struct CsvDocument {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn read_csv_document(path: &Path, normalize: impl Fn(&str) -> String) -> Result<CsvDocument> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|error| {
            BekiperError::Data(format!("failed to read {}: {error}", path.display()))
        })?;
    let headers = reader
        .headers()
        .map_err(|error| BekiperError::Data(format!("{}: {error}", path.display())))?
        .iter()
        .map(normalize)
        .collect();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record =
            record.map_err(|error| BekiperError::Data(format!("{}: {error}", path.display())))?;
        rows.push(record.iter().map(ToOwned::to_owned).collect());
    }
    Ok(CsvDocument { headers, rows })
}

impl MetadataDocument {
    pub fn load(path: &Path) -> Result<Self> {
        let CsvDocument { headers, rows } = read_csv_document(path, normalize_metadata_header)?;
        Ok(Self {
            path: path.to_path_buf(),
            headers,
            rows,
        })
    }

    pub fn get(&self, row: usize, column: &str) -> Option<&str> {
        let index = self
            .headers
            .iter()
            .position(|header| header_matches(header, column))?;
        self.rows
            .get(row)
            .and_then(|cells| cells.get(index))
            .map(String::as_str)
    }

    pub fn has_column(&self, column: &str) -> bool {
        self.headers
            .iter()
            .any(|header| header_matches(header, column))
    }

    pub fn set(&mut self, row: usize, column: &str, value: String) -> Result<()> {
        let index = self
            .headers
            .iter()
            .position(|header| header_matches(header, column))
            .ok_or_else(|| BekiperError::Data(format!("metadata column '{column}' not found")))?;

        let cells = self
            .rows
            .get_mut(row)
            .ok_or_else(|| BekiperError::Data(format!("metadata row {row} is out of bounds")))?;
        if cells.len() <= index {
            cells.resize(index + 1, String::new());
        }
        cells[index] = value;
        Ok(())
    }

    pub fn resolve_context_row(
        &self,
        context: Option<&IngressSessionContext>,
    ) -> Option<(usize, PerimeterSource)> {
        let source_row = context?.source_row?;
        let row_index = source_row.checked_sub(2)?;
        self.source_for_row(row_index)
            .map(|source| (row_index, source))
    }

    pub fn resolve_row(&self, ctx: &VideoContext) -> Option<(usize, PerimeterSource)> {
        let phase_part_index = self
            .headers
            .iter()
            .position(|header| matches!(normalized_header(header), "PhasePart" | "Part"));
        let animal_index = self
            .headers
            .iter()
            .position(|header| matches!(normalized_header(header), "Animal" | "Number"))?;
        let stage_index = self
            .headers
            .iter()
            .position(|header| header_matches(header, "Stage"));

        let [perimeter_index, change_reference_index, target_image_index] = self.source_indices();

        let mut current_phase_part = 0u32;
        let mut result = None;

        for (row_index, row) in self.rows.iter().enumerate() {
            if let Some(parsed) = phase_part_index
                .and_then(|index| row.get(index))
                .and_then(|value| value.trim().parse::<u32>().ok())
            {
                current_phase_part = parsed;
            }

            let animal = row
                .get(animal_index)
                .map(String::as_str)
                .map(str::trim)
                .and_then(|value| value.parse::<u32>().ok());
            if animal != Some(ctx.animal) {
                continue;
            }

            let stage = stage_index
                .and_then(|index| row.get(index))
                .map(String::as_str)
                .map(str::trim)
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(0);

            if stage != ctx.stage || current_phase_part != ctx.phase_part {
                continue;
            }

            let source = source_for_metadata_row(
                row,
                perimeter_index,
                change_reference_index,
                target_image_index,
            )?;

            result = Some((row_index, source));
        }

        result
    }

    fn source_indices(&self) -> [Option<usize>; 3] {
        ["Perimeter", "ChangeReference", "ChangeReferenceImageName"].map(|column| {
            self.headers
                .iter()
                .position(|header| header_matches(header, column))
        })
    }

    pub fn source_for_row(&self, row_index: usize) -> Option<PerimeterSource> {
        let [perimeter_index, change_reference_index, target_image_index] = self.source_indices();
        let row = self.rows.get(row_index)?;
        source_for_metadata_row(
            row,
            perimeter_index,
            change_reference_index,
            target_image_index,
        )
    }

    pub fn write(&self) -> Result<()> {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&self.path)
            .map_err(|e| {
                BekiperError::Data(format!("failed to write {}: {e}", self.path.display()))
            })?;
        writer
            .write_record(&self.headers)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))?;
        for row in &self.rows {
            writer
                .write_record(row)
                .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))?;
        }
        writer
            .flush()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))
    }
}

fn normalize_metadata_header(header: &str) -> String {
    normalized_header(header).to_string()
}

fn normalized_header(header: &str) -> &str {
    header.trim().trim_start_matches('\u{feff}')
}

fn header_matches(header: &str, expected: &str) -> bool {
    normalized_header(header) == expected
}

fn source_for_metadata_row(
    row: &[String],
    perimeter_index: Option<usize>,
    change_reference_index: Option<usize>,
    target_image_index: Option<usize>,
) -> Option<PerimeterSource> {
    let [perimeter, change_reference, target_image] =
        [perimeter_index, change_reference_index, target_image_index].map(|column| {
            column
                .and_then(|index| row.get(index))
                .map(String::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        });

    let perimeter_id = perimeter.or(change_reference)?;
    if let Some(target_image) = target_image {
        Some(PerimeterSource::ReReferenced {
            perimeter_id,
            target_image,
        })
    } else {
        Some(PerimeterSource::Direct { perimeter_id })
    }
}

impl VideoMapDocument {
    pub fn load(path: &Path) -> Result<Self> {
        let CsvDocument { headers, rows } = read_csv_document(path, str::to_owned)?;
        Ok(Self {
            path: path.to_path_buf(),
            headers,
            rows,
        })
    }

    pub fn resolve_row(&self, video_number: u32) -> Option<(usize, String)> {
        let number_index = self
            .headers
            .iter()
            .position(|header| header.trim() == "video_number")?;
        let id_index = self
            .headers
            .iter()
            .position(|header| header.trim() == "perimeter_id")?;
        self.rows.iter().enumerate().find_map(|(row_index, row)| {
            let parsed = row
                .get(number_index)
                .map(String::as_str)
                .map(str::trim)
                .and_then(|value| value.parse::<u32>().ok())?;
            if parsed != video_number {
                return None;
            }
            let perimeter_id = row
                .get(id_index)
                .map(String::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?
                .to_string();
            Some((row_index, perimeter_id))
        })
    }
}
