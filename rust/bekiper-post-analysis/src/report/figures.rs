//! Validated vector figure inputs.
use super::*;

pub(super) fn report_figures(directory: &Path, figures: &[Record]) -> Result<Vec<Figure>> {
    let mut result = Vec::new();
    let mut names = BTreeSet::new();
    for figure in figures {
        let name = figure["name"]
            .as_str()
            .context("report figure requires name")?;
        ensure!(
            !name.is_empty()
                && name.len() <= 128
                && name != "report"
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
            "unsafe report figure name"
        );
        ensure!(names.insert(name), "duplicate report figure");
        if matches!(figure["status"].as_str(), Some("blocked" | "unavailable")) {
            continue;
        }
        let path = crate::export::input_path(directory, &format!("{name}.pdf"));
        validate_pdf(&path)
            .with_context(|| format!("selected report figure {name} needs its vector PDF"))?;
        let captions = figure["panels"]
            .as_array()
            .map(|panels| {
                panels
                    .iter()
                    .map(|panel| {
                        format!(
                            "{}; {}; status {}",
                            text(&panel["label"]),
                            if panel["reference_line"].is_null() {
                                "no reference line".into()
                            } else {
                                format!("reference {}", text(&panel["reference_line"]))
                            },
                            text(&panel["status"])
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        result.push(Figure {
            name: name.into(),
            path: path
                .strip_prefix(directory)?
                .to_str()
                .context("figure path must be UTF-8")?
                .into(),
            status: text(&figure["status"]),
            captions,
            fingerprint: fingerprint(&path)?,
        });
    }
    Ok(result)
}
