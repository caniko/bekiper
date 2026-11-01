pub(crate) fn load_aggregate_metadata(
    base_dir: &Path,
    cfg: &AggregateMetadataConfig,
) -> Result<AggregateMetadata> {
    let path = if cfg.path.is_absolute() {
        cfg.path.clone()
    } else {
        base_dir.join(&cfg.path)
    };
    if !path.exists() {
        return Err(BekiperError::Data(format!(
            "aggregate metadata workbook is missing: {}",
            path.display()
        )));
    }

    let table = read_xlsx_sheet(&path, &cfg.sheet)?;
    parse_metadata_table(path, table)
}

/// ZIP-directory-only preflight: no worksheet payload is decompressed here.
pub fn metadata_xml_bytes(bytes: &[u8]) -> Result<u64> {
    let mut zip = ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| BekiperError::Data(e.to_string()))?;
    (0..zip.len()).try_fold(0u64, |total, index| {
        let file = zip
            .by_index(index)
            .map_err(|e| BekiperError::Data(e.to_string()))?;
        if file.name().ends_with(".xml") || file.name().ends_with(".rels") {
            total
                .checked_add(file.size())
                .ok_or_else(|| BekiperError::Data("metadata XML size overflow".into()))
        } else {
            Ok(total)
        }
    })
}

/// Call only while holding the XML-preflight reservation. Accounts for sparse
/// worksheet columns and repeated shared strings before table materialization.
pub fn metadata_workspace_bytes(bytes: &[u8]) -> Result<u64> {
    let xml_bytes = metadata_xml_bytes(bytes)?;
    let mut zip = ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| BekiperError::Data(e.to_string()))?;
    let strings = read_shared_strings(&mut zip)?;
    let max_string = strings.iter().map(String::len).max().unwrap_or(0) as u64;
    let names: Vec<_> = zip
        .file_names()
        .filter(|n| n.starts_with("xl/worksheets/") && n.ends_with(".xml"))
        .map(str::to_string)
        .collect();
    let mut sparse_cells = 0u64;
    let mut cells = 0u64;
    for name in names {
        let xml = read_zip_string(&mut zip, &name)?;
        let doc = Document::parse(&xml).map_err(|e| BekiperError::Data(e.to_string()))?;
        for row in doc
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "row")
        {
            let mut width = 0usize;
            for cell in row
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "c")
            {
                let mut column = 0usize;
                if let Some(reference) = cell.attribute("r") {
                    for ch in reference.chars().take_while(|ch| ch.is_ascii_alphabetic()) {
                        column = column
                            .checked_mul(26)
                            .and_then(|n| {
                                n.checked_add(ch.to_ascii_uppercase() as usize - 'A' as usize + 1)
                            })
                            .ok_or_else(|| BekiperError::Data("metadata column overflow".into()))?;
                    }
                }
                width = width.max(if column == 0 { width + 1 } else { column });
                if width > 16384 {
                    return Err(BekiperError::Data(
                        "metadata column exceeds XLSX limits".into(),
                    ));
                }
                cells = cells
                    .checked_add(1)
                    .ok_or_else(|| BekiperError::Data("metadata cell overflow".into()))?;
            }
            sparse_cells = sparse_cells
                .checked_add(width as u64)
                .ok_or_else(|| BekiperError::Data("metadata row overflow".into()))?;
        }
    }
    xml_bytes
        .checked_mul(128)
        .and_then(|n| n.checked_add(sparse_cells.checked_mul(128)?))
        .and_then(|n| n.checked_add(cells.checked_mul(max_string)?.checked_mul(4)?))
        .ok_or_else(|| BekiperError::Data("metadata workspace overflow".into()))
}

fn parse_metadata_table(path: PathBuf, table: Vec<Vec<String>>) -> Result<AggregateMetadata> {
    let Some(header_row) = table.first() else {
        return Err(BekiperError::Data(format!(
            "{}: metadata workbook sheet is empty",
            path.display()
        )));
    };

    let mut header_to_index: HashMap<String, usize> = HashMap::new();
    for (idx, header) in header_row.iter().enumerate() {
        let trimmed = header.trim();
        if !trimmed.is_empty() {
            header_to_index.insert(trimmed.to_string(), idx);
        }
    }
    if !header_to_index.contains_key("Animal ID") {
        return Err(BekiperError::Data(format!(
            "{}: aggregate metadata sheet must contain an `Animal ID` column",
            path.display()
        )));
    }

    let mut rows: IndexMap<u32, IndexMap<String, CellValue>> = IndexMap::new();
    for (row_idx, raw) in table.into_iter().enumerate().skip(1) {
        let Some(animal_cell) = header_to_index
            .get("Animal ID")
            .and_then(|idx| raw.get(*idx))
            .map(String::as_str)
        else {
            continue;
        };
        let Some(animal) = parse_animal_id(animal_cell) else {
            continue;
        };
        if rows.contains_key(&animal) {
            return Err(BekiperError::Data(format!(
                "{}: duplicate Animal ID `{animal}` in metadata row {}",
                path.display(),
                row_idx + 1
            )));
        }

        let mut row = IndexMap::new();
        for &(source, target) in HEADER_MAP {
            let value = header_to_index
                .get(source)
                .and_then(|idx| raw.get(*idx))
                .map(|value| value.trim())
                .unwrap_or("");
            let cell = if target == "animal" {
                CellValue::Int(animal as i64)
            } else if value.is_empty() {
                CellValue::Null
            } else if matches!(
                target,
                "group" | "age_group_nr" | "start_age_days" | "round"
            ) {
                parse_metadata_number(value).unwrap_or_else(|| CellValue::String(value.into()))
            } else if matches!(
                target,
                "start_age_months" | "end_age_days" | "end_age_months"
            ) {
                parse_metadata_number(value).unwrap_or_else(|| CellValue::String(value.into()))
            } else {
                CellValue::String(value.replace('\n', " ").trim().to_string())
            };
            row.insert(target.to_string(), cell);
        }
        rows.insert(animal, row);
    }

    Ok(AggregateMetadata { path, rows })
}

fn parse_metadata_number(value: &str) -> Option<CellValue> {
    let parsed = value.parse::<f64>().ok()?;
    if parsed.fract() == 0.0 && parsed >= i64::MIN as f64 && parsed <= i64::MAX as f64 {
        Some(CellValue::Int(parsed as i64))
    } else {
        Some(CellValue::Float(parsed))
    }
}

pub(crate) fn parse_animal_id(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed
        .parse::<u32>()
        .ok()
        .or_else(|| trimmed.parse::<f64>().ok().map(|v| v as u32))
}

/// Workbook `(pre, post)` ages per animal: the cells under `pre_col` / `post_col`.
pub(crate) type AnimalAges = HashMap<u32, (Option<f64>, Option<f64>)>;

pub(crate) fn read_animal_ages(
    path: &Path,
    sheet: &str,
    animal_col: &str,
    pre_col: &str,
    post_col: &str,
) -> Result<AnimalAges> {
    let table = read_xlsx_sheet(path, sheet)?;
    let header = table.first().ok_or_else(|| {
        BekiperError::Data(format!("{}: sheet `{sheet}` is empty", path.display()))
    })?;
    let find = |name: &str| {
        header.iter().position(|h| h.trim() == name).ok_or_else(|| {
            BekiperError::Data(format!(
                "{}: sheet `{sheet}` has no `{name}` column",
                path.display()
            ))
        })
    };
    let (animal, pre, post) = (find(animal_col)?, find(pre_col)?, find(post_col)?);
    let number = |row: &[String], i: usize| row.get(i).and_then(|v| v.trim().parse::<f64>().ok());
    let mut ages = AnimalAges::new();
    for row in table.iter().skip(1) {
        let Some(id) = row.get(animal).and_then(|v| parse_animal_id(v)) else {
            continue;
        };
        if ages
            .insert(id, (number(row, pre), number(row, post)))
            .is_some()
        {
            return Err(BekiperError::Data(format!(
                "{}: duplicate `{animal_col}` {id} in sheet `{sheet}`",
                path.display()
            )));
        }
    }
    Ok(ages)
}
