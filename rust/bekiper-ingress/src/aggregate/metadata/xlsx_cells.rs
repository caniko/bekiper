fn cell_text(cell: roxmltree::Node<'_, '_>, shared_strings: &[String]) -> Result<String> {
    let raw = cell
        .descendants()
        .find(|node| node.is_element() && node.tag_name().name() == "v")
        .and_then(|node| node.text())
        .unwrap_or("");
    match cell.attribute("t") {
        Some("s") => raw
            .parse::<usize>()
            .ok()
            .and_then(|idx| shared_strings.get(idx))
            .cloned()
            .ok_or_else(|| {
                BekiperError::Data(format!("invalid metadata shared-string index {raw}"))
            }),
        Some("inlineStr") => Ok(cell
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "t")
            .filter_map(|node| node.text())
            .collect()),
        Some("e") => Err(BekiperError::Data(format!("workbook formula error {raw}"))),
        _ => Ok(raw.to_string()),
    }
}

fn cell_column_index(reference: &str) -> Option<usize> {
    let mut index = 0usize;
    let mut found = false;
    for ch in reference.chars().take_while(|ch| ch.is_ascii_alphabetic()) {
        found = true;
        index = index
            .checked_mul(26)?
            .checked_add(ch.to_ascii_uppercase() as usize - 'A' as usize + 1)?;
    }
    found.then(|| index - 1)
}

fn read_zip_string<R: std::io::Read + std::io::Seek>(
    zip: &mut ZipArchive<R>,
    name: &str,
) -> Result<String> {
    let mut file = zip.by_name(name).map_err(|e| {
        BekiperError::Data(format!("aggregate metadata XLSX missing `{name}`: {e}"))
    })?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)
        .map_err(|e| BekiperError::Data(format!("failed to read `{name}` from XLSX: {e}")))?;
    Ok(raw)
}
