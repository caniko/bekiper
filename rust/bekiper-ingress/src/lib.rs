pub mod aggregate;
pub mod analysis;
mod analysis_exclusion;
pub mod annotation;
pub mod audit;
pub mod config;
mod coordinate_diagnostics;
pub mod export;
pub mod feature_inspect;
pub mod inspection;
#[cfg(feature = "pose-video")]
pub mod inspection_video;
pub mod overlay;
pub mod parallel;
pub mod pose_import;
pub mod preparation;
mod spreadsheet;
pub mod stage;
pub mod stage_share;
#[cfg(feature = "pose-gpu")]
pub mod workflow;

/// Run `f` with a TRACE subscriber so log-field expressions execute in tests.
#[cfg(test)]
pub(crate) fn traced<T>(f: impl FnOnce() -> T) -> T {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(std::io::sink)
        .finish();
    tracing::subscriber::with_default(subscriber, f)
}

/// One-sheet XLSX (`Updated_Mice-overview`) with inline-string cells.
#[cfg(test)]
pub(crate) fn write_test_xlsx(path: &std::path::Path, rows: &[&[&str]]) {
    use std::io::Write;
    let sheet = rows
        .iter()
        .map(|row| {
            let cells = row
                .iter()
                .map(|cell| format!("<c t=\"inlineStr\"><is><t>{cell}</t></is></c>"))
                .collect::<String>();
            format!("<row>{cells}</row>")
        })
        .collect::<String>();
    let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, body) in [
        (
            "xl/workbook.xml",
            r#"<workbook xmlns:r="r"><sheets><sheet name="Updated_Mice-overview" r:id="rId1"/></sheets></workbook>"#.to_string(),
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#.to_string(),
        ),
        ("xl/worksheets/sheet1.xml", format!("<worksheet>{sheet}</worksheet>")),
    ] {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(body.as_bytes()).unwrap();
    }
    zip.finish().unwrap();
}
