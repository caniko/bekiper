//! Poppler extraction/rendering is driven by the native verifier with immutable input inventories.
use super::*;
use std::{path::PathBuf, process::Command};

/// Publish source-bound text bounds and representative page images for visual review.
pub fn review_pdfs(root: &Path, output: &Path) -> Result<Record> {
    let verification = verify_run(root)?;
    let before = pipeline::verify_artifacts(root)?;
    let root = root.canonicalize()?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let output = parent.join(output.file_name().context("PDF review output")?);
    ensure!(
        !output.starts_with(&root) && !root.starts_with(&output),
        "PDF review output overlaps the input package"
    );
    let mut publication = pipeline::Publication::claim(&output)?;
    let mut reports = Vec::new();
    for member in before["files"].as_array().context("artifact inventory")? {
        let relative = member["path"].as_str().context("artifact path")?;
        if relative != "report.pdf" && !relative.ends_with("/report.pdf") {
            continue;
        }
        let path = root.join(relative);
        let prefix = relative
            .trim_end_matches("/report.pdf")
            .trim_end_matches("report.pdf")
            .replace('/', "-");
        let prefix = if prefix.is_empty() {
            "report".into()
        } else {
            prefix
        };
        let text_path = publication.directory().join(format!("{prefix}.bbox.html"));
        let status = Command::new("pdftotext")
            .args(["-bbox", "-enc", "UTF-8"])
            .arg(&path)
            .arg(&text_path)
            .status()
            .context("PDF review requires Poppler pdftotext on PATH")?;
        ensure!(
            status.success(),
            "Poppler text extraction failed: {relative}"
        );
        let text = std::fs::read_to_string(&text_path)?;
        let (pages, outside) = text_bounds(&text)?;
        let mut selected: BTreeSet<_> = std::iter::once(1)
            .chain(
                pages
                    .iter()
                    .filter(|p| p["policy_section"] == true)
                    .take(2)
                    .filter_map(|p| p["page"].as_u64()),
            )
            .collect();
        for label in ["HCAR1 B:", "HCAR1 C:"] {
            if let Some(page) = pages
                .iter()
                .find(|p| p["text"].as_str().is_some_and(|text| text.contains(label)))
            {
                selected.insert(page["page"].as_u64().context("policy page identity")?);
            }
        }
        let mut images = Vec::new();
        for page in selected {
            let image = publication
                .directory()
                .join(format!("{prefix}-page-{page:03}"));
            let status = Command::new("pdftoppm")
                .args(["-png", "-singlefile", "-r", "120", "-f"])
                .arg(page.to_string())
                .arg("-l")
                .arg(page.to_string())
                .arg(&path)
                .arg(&image)
                .status()
                .context("PDF review requires Poppler pdftoppm on PATH")?;
            ensure!(
                status.success(),
                "PDF representative page render failed: {relative}:{page}"
            );
            let image = PathBuf::from(format!("{}.png", image.display()));
            verify_png(&image, 120)?;
            images.push(
                image
                    .file_name()
                    .context("PDF preview filename")?
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        reports.push(json!({"source":relative,"sha256":member["sha256"],"page_count":pages.len(),
            "policy_section_pages":pages.into_iter().filter(|p|p["policy_section"]==true).collect::<Vec<_>>(),
            "words_outside_page":outside,"representative_pages":images}));
    }
    ensure!(!reports.is_empty(), "no report PDFs in the verified export");
    ensure!(
        pipeline::verify_artifacts(&root)? == before,
        "input artifacts changed during PDF review"
    );
    let receipt = json!({"schema_version":1,"scope":"source_bound_pdf_text_and_visual_review","code":verification["code"],"verifier_code":verification["verifier_code"],
        "source_inventory":inputs::fingerprint(&root.join("artifact_manifest.json"))?,"reports":reports,"analysis_admitted":false,
        "status":if reports.iter().all(|r|r["words_outside_page"].as_array().is_some_and(Vec::is_empty)) {"bounds_verified_visual_review_pending"} else {"text_bounds_failed"}});
    pipeline::write_json(&publication.directory().join("receipt.json"), &receipt)?;
    pipeline::write_artifact_manifest(publication.directory())?;
    publication.publish()?;
    ensure!(
        receipt["status"] != "text_bounds_failed",
        "PDF text extends beyond the page; see {}/receipt.json",
        output.display()
    );
    Ok(receipt)
}

fn text_bounds(text: &str) -> Result<(Vec<Record>, Vec<Record>)> {
    // Poppler emits XHTML with a document-type declaration. roxmltree never
    // fetches the external DTD; permit the declaration in this generated output.
    let document = roxmltree::Document::parse_with_options(
        text,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )?;
    let mut pages = Vec::new();
    let mut outside = Vec::new();
    for (index, page) in document
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "page")
        .enumerate()
    {
        let attribute = |node: roxmltree::Node<'_, '_>, key: &str| -> Result<f64> {
            let value = node
                .attribute(key)
                .with_context(|| format!("PDF {key}"))?
                .parse::<f64>()?;
            ensure!(value.is_finite(), "nonfinite PDF coordinate");
            Ok(value)
        };
        let width = attribute(page, "width")?;
        let height = attribute(page, "height")?;
        ensure!(width > 0.0 && height > 0.0, "invalid PDF page dimensions");
        let mut words = Vec::new();
        for word in page
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "word")
        {
            let bounds = ["xMin", "yMin", "xMax", "yMax"].map(|key| attribute(word, key));
            let [x0, y0, x1, y1] = bounds
                .into_iter()
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| anyhow::anyhow!("PDF bounds"))?;
            if x0 < -1.0 || y0 < -1.0 || x1 > width + 1.0 || y1 > height + 1.0 || x1 < x0 || y1 < y0
            {
                outside.push(json!({"page":index+1,"bbox":[x0,y0,x1,y1],"text":word.text()}));
            }
            words.push(word.text().unwrap_or(""));
        }
        let text = words.join(" ");
        let policy_section = [
            "HCAR1 handling:",
            "Policy sensitivity:",
            "Sensitivity policy",
        ]
        .iter()
        .any(|s| text.contains(s));
        pages.push(
            json!({"page":index+1,"width":width,"height":height,"text":text,
            "policy_section":policy_section}),
        );
    }
    ensure!(!pages.is_empty(), "PDF extraction has no pages");
    Ok((pages, outside))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_and_policy_sections_are_measured_per_page() {
        let text = r#"<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"><html><body><page width="100" height="200"><word xMin="2" yMin="2" xMax="30" yMax="10">HCAR1</word><word xMin="31" yMin="2" xMax="50" yMax="10">handling:</word><word xMin="10" yMin="-2" xMax="30" yMax="10">overflow</word></page></body></html>"#;
        let (pages, outside) = text_bounds(text).unwrap();
        assert_eq!(pages[0]["policy_section"], true);
        assert_eq!(outside.len(), 1);
        assert_eq!(outside[0]["text"], "overflow");
        assert!(text_bounds(&text.replace("xMin=\"2\"", "xMin=\"NaN\"")).is_err());
    }
}
