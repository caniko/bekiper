//! Source identity and package-path descriptions.
use super::*;

pub(super) fn provenance_section(manifest: &Record) -> Result<Section> {
    let mut provenance = Section::new("Provenance and reproducibility");
    let resolved_hash = if manifest["resolved_config"].is_object() {
        record_fingerprint("resolved_config", &manifest["resolved_config"])?["sha256"].clone()
    } else {
        Value::Null
    };
    provenance.paragraph(format!(
        "Run-manifest schema: {}. Package: {}; version: {}; code revision: {}; dirty status: {}.",
        text(&manifest["schema_version"]),
        text(&manifest["code"]["package"]),
        text(&manifest["code"]["version"]),
        text(&manifest["code"]["revision"]),
        text(&manifest["code"]["dirty"])
    ));
    provenance.table(
        &["Identity", "SHA-256"],
        vec![
            vec![
                "Scientific recipe".into(),
                text(&manifest["source_recipe"]["sha256"]),
            ],
            vec![
                "Y-maze amendment".into(),
                text(&manifest["ymaze_amendment"]["sha256"]),
            ],
            vec![
                "Configuration plan input".into(),
                text(config_input_hash(manifest)),
            ],
            vec![
                "Resolved configuration snapshot".into(),
                text(&resolved_hash),
            ],
            vec![
                "Native engine source".into(),
                text(&manifest["code"]["source_sha256"]),
            ],
        ],
    );
    if let Some(inputs) = manifest["inputs"].as_array() {
        provenance.table(
            &["Input basename / role", "Bytes", "SHA-256"],
            inputs
                .iter()
                .map(|input| {
                    vec![
                        format!(
                            "{} / {}",
                            basename(&text(&input["name"])),
                            text(&input["role"])
                        ),
                        text(&input["bytes"]),
                        text(&input["sha256"]),
                    ]
                })
                .collect(),
        );
    }
    if let Some(versions) = manifest["dependencies"].as_object() {
        provenance.table(
            &["Dependency", "Version / identity"],
            versions
                .iter()
                .map(|(name, value)| {
                    let mut identity = value.clone();
                    if let Some(object) = identity.as_object_mut()
                        && let Some(files) = object.get("files").and_then(Value::as_array)
                    {
                        // The immutable manifest retains every path/hash. Expanding
                        // thousands of artifacts into one table cell exceeds TeX's
                        // line buffer and produces an unreadable provenance row.
                        object.insert(
                            "files".into(),
                            format!(
                                "{} artifact files; full inventory in run_manifest.json",
                                files.len()
                            )
                            .into(),
                        );
                    }
                    vec![name.clone(), text(&identity)]
                })
                .collect(),
        );
    }
    provenance.paragraph(format!("Random seed (where used): {}. Report compiler versions, invocation arguments, source and vector-figure fingerprints are in report_build.json when PDF reporting is enabled. artifact_manifest.json binds the final package members; scientific decisions and input verification are distinct from software completion.",text(&manifest["random_seed"])));
    Ok(provenance)
}

pub(super) fn organize_section_paths(sections: &mut [Section]) {
    for section in sections {
        for paragraph in &mut section.paragraphs {
            for name in [
                "inclusion_ledger.json",
                "pairing_ledger.json",
                "summaries.json",
                "models.json",
                "plotting_data.json",
                "report_build.json",
            ] {
                *paragraph = paragraph.replace(name, &crate::export::relative_path(name));
            }
        }
    }
}
