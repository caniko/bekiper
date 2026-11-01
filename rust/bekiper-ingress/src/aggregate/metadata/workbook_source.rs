/// Read literal researcher annotations without the aggregate display normalization.
pub(crate) fn read_workbook_source(path: &Path, sheet: &str) -> Result<Vec<Value>> {
    let file = std::fs::File::open(path)?;
    let mut zip = ZipArchive::new(file).map_err(|e| BekiperError::Data(e.to_string()))?;
    let expanded = (0..zip.len()).try_fold(0u64, |total, index| {
        total
            .checked_add(
                zip.by_index(index)
                    .map_err(|e| BekiperError::Data(e.to_string()))?
                    .size(),
            )
            .ok_or_else(|| BekiperError::Data("workbook size overflow".into()))
    })?;
    if expanded > 64 * 1024 * 1024 {
        return Err(BekiperError::Data("oversized source workbook".into()));
    }
    let strings = read_shared_strings(&mut zip)?;
    let sheet_path = resolve_sheet_path(&mut zip, sheet)?;
    let xml = read_zip_string(&mut zip, &sheet_path)?;
    let mut rows = worksheet_source_rows(&xml, &strings)?.into_iter();
    let (_, headers) = rows
        .next()
        .ok_or_else(|| BekiperError::Data("empty source worksheet".into()))?;
    let mut seen = HashSet::new();
    if headers
        .iter()
        .filter(|h| !h.is_empty())
        .any(|h| h == "source_row" || !seen.insert(h))
    {
        return Err(BekiperError::Data(
            "duplicate source workbook header".into(),
        ));
    }
    rows.filter(|(_, row)| row.iter().any(|v| !v.is_empty()))
        .map(|(index, row)| {
            if row.len() > headers.len() {
                return Err(BekiperError::Data(format!(
                    "source workbook row {index} exceeds header"
                )));
            }
            let mut record = Map::new();
            for (column, header) in headers.iter().enumerate() {
                let text = row.get(column).map_or("", String::as_str);
                if header.is_empty() {
                    if !text.is_empty() {
                        return Err(BekiperError::Data(format!(
                            "populated unnamed workbook column at row {index}"
                        )));
                    }
                } else {
                    record.insert(
                        header.clone(),
                        if text.is_empty() {
                            Value::Null
                        } else {
                            Value::String(text.into())
                        },
                    );
                }
            }
            record.insert("source_row".into(), index.into());
            Ok(Value::Object(record))
        })
        .collect()
}
