fn worksheet_rows(xml: &str, shared_strings: &[String]) -> Result<Vec<Vec<String>>> {
    Ok(worksheet_source_rows(xml, shared_strings)?
        .into_iter()
        .map(|(_, row)| row)
        .collect())
}

fn worksheet_source_rows(
    xml: &str,
    shared_strings: &[String],
) -> Result<Vec<(usize, Vec<String>)>> {
    let doc = Document::parse(xml)
        .map_err(|e| BekiperError::Data(format!("invalid worksheet XML: {e}")))?;
    let mut rows = Vec::new();
    let mut seen_rows = HashSet::new();
    let mut expanded_text_bytes = 0usize;
    let mut expanded_cells = 0usize;
    for row_node in doc
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "row")
    {
        let mut row: Vec<String> = Vec::new();
        let mut seen_columns = HashSet::new();
        for cell in row_node
            .children()
            .filter(|node| node.is_element() && node.tag_name().name() == "c")
        {
            let index = match cell.attribute("r") {
                Some(reference) => cell_column_index(reference).ok_or_else(|| {
                    BekiperError::Data(format!("invalid metadata cell reference {reference}"))
                })?,
                None => row.len(),
            };
            if index >= 16384 {
                return Err(BekiperError::Data(
                    "metadata column exceeds XLSX limits".into(),
                ));
            }
            if row.len() <= index {
                row.resize(index + 1, String::new());
            }
            if !seen_columns.insert(index) {
                return Err(BekiperError::Data("duplicate metadata cell column".into()));
            }
            row[index] = cell_text(cell, shared_strings)?;
            expanded_text_bytes = expanded_text_bytes
                .checked_add(row[index].len())
                .ok_or_else(|| BekiperError::Data("metadata text expansion overflow".into()))?;
            if expanded_text_bytes > 64 * 1024 * 1024 {
                return Err(BekiperError::Data(
                    "oversized expanded metadata text".into(),
                ));
            }
        }
        let index = row_node
            .attribute("r")
            .map(str::parse::<usize>)
            .transpose()
            .map_err(|e| BekiperError::Data(format!("invalid workbook source row: {e}")))?
            .unwrap_or(rows.len() + 1);
        if index == 0 || index > 1_048_576 || !seen_rows.insert(index) {
            return Err(BekiperError::Data(
                "invalid or duplicate metadata source row".into(),
            ));
        }
        rows.push((index, row));
        expanded_cells = expanded_cells
            .checked_add(rows.last().map_or(0, |(_, r)| r.len()))
            .ok_or_else(|| BekiperError::Data("metadata cell expansion overflow".into()))?;
        if expanded_cells > 4_000_000 {
            return Err(BekiperError::Data(
                "oversized expanded metadata cells".into(),
            ));
        }
    }
    Ok(rows)
}
