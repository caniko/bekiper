//! Bounded OpenDocument sheet decoding.
use super::*;

pub fn read_ods(path: &Path, sheet: Option<&str>) -> Result<Vec<Record>> {
    snapshot::read_pinned(path, |file| decode_ods(file, sheet)).map(|(rows, _)| rows)
}

/// Literal sheet evidence, including physical row numbers and empty cells.
/// Unnamed trailing columns are omitted; populated unnamed columns are rejected.
pub fn read_ods_source(path: &Path, sheet: &str) -> Result<Vec<Record>> {
    snapshot::read_pinned(path, |file| decode(file, Some(sheet), true)).map(|(rows, _)| rows)
}

pub(super) fn decode_ods(file: &mut File, sheet: Option<&str>) -> Result<Vec<Record>> {
    decode(file, sheet, false)
}

fn decode(file: &mut File, sheet: Option<&str>, source: bool) -> Result<Vec<Record>> {
    let sheet = sheet
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("ODS requires an explicit sheet"))?;
    let mut archive = ::zip::ZipArchive::new(file)?;
    let mut content = String::new();
    archive
        .by_name("content.xml")?
        .take(64 * 1024 * 1024 + 1)
        .read_to_string(&mut content)?;
    ensure!(content.len() <= 64 * 1024 * 1024, "Oversized ODS content");
    let document = roxmltree::Document::parse(&content)?;
    const T: &str = "urn:oasis:names:tc:opendocument:xmlns:table:1.0";
    const O: &str = "urn:oasis:names:tc:opendocument:xmlns:office:1.0";
    let mut matches = document
        .descendants()
        .filter(|n| n.has_tag_name((T, "table")) && n.attribute((T, "name")) == Some(sheet));
    let table = matches
        .next()
        .ok_or_else(|| anyhow!("ODS sheet {sheet:?} not present"))?;
    ensure!(
        matches.next().is_none(),
        "Duplicate ODS sheet name: {sheet}"
    );
    let mut rows: Vec<(usize, Vec<Value>)> = Vec::new();
    let mut physical_row = 1usize;
    let mut expanded_cells = 0usize;
    let mut expanded_text_bytes = 0usize;
    for row in table
        .descendants()
        .filter(|n| n.has_tag_name((T, "table-row")))
    {
        let mut cells = Vec::new();
        let mut text_bytes = 0usize;
        for cell in row.children().filter(|n| n.is_element()) {
            ensure!(
                cell.has_tag_name((T, "table-cell"))
                    || cell.has_tag_name((T, "covered-table-cell")),
                "Unsupported ODS row element"
            );
            let repeat = cell
                .attribute((T, "number-columns-repeated"))
                .unwrap_or("1")
                .parse::<usize>()?;
            ensure!(
                repeat > 0 && repeat <= 16384 && cells.len() + repeat <= 16384,
                "Oversized ODS cell repeat"
            );
            let text = cell
                .descendants()
                .filter(|n| n.is_text())
                .filter_map(|n| n.text())
                .collect::<String>();
            let text = if source {
                let paragraphs: Vec<_> = cell
                    .descendants()
                    .filter(|n| n.is_element() && n.tag_name().name() == "p")
                    .map(|p| {
                        p.descendants()
                            .filter(|n| n.is_text())
                            .filter_map(|n| n.text())
                            .collect::<String>()
                    })
                    .collect();
                if paragraphs.is_empty() {
                    text
                } else {
                    paragraphs.join("\n")
                }
            } else {
                text
            };
            let value = match cell.attribute((O, "value-type")) {
                Some("float") => numeric_text(
                    cell.attribute((O, "value"))
                        .ok_or_else(|| anyhow!("Missing ODS numeric value"))?,
                    "ODS numeric value",
                )?,
                Some("boolean") => match cell.attribute((O, "boolean-value")) {
                    Some("true") => json!(true),
                    Some("false") => json!(false),
                    _ => bail!("Invalid ODS boolean"),
                },
                None | Some("string")
                    if if source {
                        text.is_empty()
                    } else {
                        text.trim().is_empty()
                    } =>
                {
                    Value::Null
                }
                None | Some("string") => json!(if source { &text } else { text.trim() }),
                Some(kind) => bail!(
                    "Unsupported ODS value type: {kind}; export canonical numeric units explicitly"
                ),
            };
            let bytes = value
                .as_str()
                .map_or(0, str::len)
                .checked_mul(repeat)
                .context("Oversized ODS text repeat")?;
            text_bytes = text_bytes
                .checked_add(bytes)
                .context("Oversized ODS text repeat")?;
            ensure!(
                text_bytes <= 64 * 1024 * 1024,
                "Oversized expanded ODS text"
            );
            cells.extend(std::iter::repeat_n(value, repeat));
        }
        while cells.last().is_some_and(Value::is_null) {
            cells.pop();
        }
        let repeat = row
            .attribute((T, "number-rows-repeated"))
            .unwrap_or("1")
            .parse::<usize>()?;
        ensure!(repeat > 0, "Invalid ODS row repeat");
        let next_row = physical_row
            .checked_add(repeat)
            .context("ODS row overflow")?;
        if cells.iter().any(|v| !v.is_null()) {
            ensure!(
                repeat > 0 && repeat <= 100000 && rows.len() + repeat <= 100000,
                "Oversized ODS row repeat"
            );
            expanded_cells = expanded_cells
                .checked_add(
                    cells
                        .len()
                        .checked_mul(repeat)
                        .context("Oversized ODS cell expansion")?,
                )
                .context("Oversized ODS cell expansion")?;
            expanded_text_bytes = expanded_text_bytes
                .checked_add(
                    text_bytes
                        .checked_mul(repeat)
                        .context("Oversized ODS text expansion")?,
                )
                .context("Oversized ODS text expansion")?;
            ensure!(
                expanded_cells <= 4_000_000 && expanded_text_bytes <= 64 * 1024 * 1024,
                "Oversized expanded ODS table"
            );
            rows.extend((physical_row..next_row).map(|index| (index, cells.clone())));
        }
        physical_row = next_row;
    }
    ensure!(!rows.is_empty(), "Empty ODS sheet");
    let headers = rows
        .remove(0)
        .1
        .into_iter()
        .map(|v| {
            if source && v.is_null() {
                return Ok(String::new());
            }
            v.as_str()
                .filter(|s| !s.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("Invalid ODS header"))
        })
        .collect::<Result<Vec<_>>>()?;
    let named: Vec<_> = headers.iter().filter(|h| !h.is_empty()).collect();
    ensure!(
        named.iter().collect::<BTreeSet<_>>().len() == named.len()
            && (!source || !headers.iter().any(|h| h == "source_row")),
        "duplicate ODS headers"
    );
    rows.into_iter()
        .map(|(index, mut row)| {
            ensure!(row.len() <= headers.len(), "ODS row/header width mismatch");
            row.resize(headers.len(), Value::Null);
            let mut record = Map::new();
            for (header, value) in headers.iter().zip(row) {
                if header.is_empty() {
                    ensure!(
                        value.is_null(),
                        "Populated unnamed ODS column at row {index}"
                    );
                } else {
                    record.insert(header.clone(), value);
                }
            }
            if source {
                record.insert("source_row".into(), json!(index));
            }
            Ok(Value::Object(record))
        })
        .collect()
}
