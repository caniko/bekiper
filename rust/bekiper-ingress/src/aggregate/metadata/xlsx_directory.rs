fn read_xlsx_sheet(path: &Path, sheet_name: &str) -> Result<Vec<Vec<String>>> {
    let file = std::fs::File::open(path).map_err(|e| {
        BekiperError::Data(format!(
            "failed to open aggregate metadata {}: {e}",
            path.display()
        ))
    })?;
    let mut zip = ZipArchive::new(file).map_err(|e| {
        BekiperError::Data(format!(
            "failed to read aggregate metadata XLSX {}: {e}",
            path.display()
        ))
    })?;
    let shared_strings = read_shared_strings(&mut zip)?;
    let sheet_path = resolve_sheet_path(&mut zip, sheet_name)?;
    let xml = read_zip_string(&mut zip, &sheet_path)?;
    worksheet_rows(&xml, &shared_strings)
}

fn read_shared_strings<R: std::io::Read + std::io::Seek>(
    zip: &mut ZipArchive<R>,
) -> Result<Vec<String>> {
    if let Err(zip::result::ZipError::FileNotFound) = zip.by_name("xl/sharedStrings.xml") {
        return Ok(Vec::new());
    }
    let xml = read_zip_string(zip, "xl/sharedStrings.xml")?;
    let doc = Document::parse(&xml)
        .map_err(|e| BekiperError::Data(format!("invalid xl/sharedStrings.xml: {e}")))?;
    Ok(doc
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "si")
        .map(|si| {
            si.descendants()
                .filter(|node| node.is_element() && node.tag_name().name() == "t")
                .filter_map(|node| node.text())
                .collect::<String>()
        })
        .collect())
}

fn resolve_sheet_path<R: std::io::Read + std::io::Seek>(
    zip: &mut ZipArchive<R>,
    sheet_name: &str,
) -> Result<String> {
    let workbook_xml = read_zip_string(zip, "xl/workbook.xml")?;
    let workbook = Document::parse(&workbook_xml)
        .map_err(|e| BekiperError::Data(format!("invalid xl/workbook.xml: {e}")))?;
    let rel_id = workbook
        .descendants()
        .find(|node| {
            node.is_element()
                && node.tag_name().name() == "sheet"
                && node.attribute("name") == Some(sheet_name)
        })
        .and_then(|node| node.attributes().find(|attr| attr.name() == "id"))
        .map(|attr| attr.value().to_string())
        .ok_or_else(|| {
            BekiperError::Data(format!(
                "aggregate metadata workbook does not contain sheet `{sheet_name}`"
            ))
        })?;

    let rels_xml = read_zip_string(zip, "xl/_rels/workbook.xml.rels")?;
    let rels = Document::parse(&rels_xml)
        .map_err(|e| BekiperError::Data(format!("invalid workbook relationships: {e}")))?;
    let target = rels
        .descendants()
        .find(|node| {
            node.is_element()
                && node.tag_name().name() == "Relationship"
                && node.attribute("Id") == Some(rel_id.as_str())
        })
        .and_then(|node| node.attribute("Target"))
        .ok_or_else(|| {
            BekiperError::Data(format!(
                "aggregate metadata workbook sheet `{sheet_name}` has no worksheet relationship"
            ))
        })?;
    Ok(if target.starts_with("xl/") {
        target.to_string()
    } else {
        format!("xl/{}", target.trim_start_matches('/'))
    })
}
