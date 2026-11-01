//! One-time presentation exports preserve the original scientific run and
//! independently identify the exporter; no scoring or inference is performed.
use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde_json::json;

use crate::{
    Record,
    pipeline::{
        self,
        publication::{Publication, file_fingerprint},
    },
};

pub fn export_existing(input: &Path, output: &Path) -> Result<Record> {
    let input = input.canonicalize()?;
    // Registered project exports share the same archive-cleanup lease as fresh
    // cohort publication. Standalone/retained exports do not have a project root.
    let project = input
        .parent()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name == "_post_analysis")
        })
        .and_then(Path::parent)
        .filter(|path| path.file_name().is_some_and(|name| name == "data"))
        .and_then(Path::parent);
    let _lease = project
        .map(pipeline::ProjectDataLease::export)
        .transpose()?;
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    ensure!(
        !parent.starts_with(&input),
        "presentation export must be outside the immutable source run"
    );
    let inventory = pipeline::verify_artifacts(&input)?;
    let source_inventory_bytes = fs::read(input.join("artifact_manifest.json"))?;
    let source_inventory = file_fingerprint(&input.join("artifact_manifest.json"))?;
    let source_manifest_bytes = fs::read(input.join("run_manifest.json"))?;
    let mut manifest: Record = serde_json::from_slice(&source_manifest_bytes)?;
    ensure!(
        manifest["schema_version"] == "2.0"
            && matches!(
                manifest["status"].as_str(),
                Some("complete" | "partial" | "blocked")
            ),
        "source must be a completed post-analysis run"
    );
    let rows: Vec<Record> =
        serde_json::from_slice(&fs::read(super::input_path(&input, "metrics.json"))?)?;
    let source_prefix = format!(
        "provenance/presentation_source/{}",
        &source_inventory["sha256"]
            .as_str()
            .context("source inventory hash")?[..12]
    );
    let mut publication = Publication::claim(output)?;
    let result = (|| -> Result<Record> {
        let directory = publication.directory();
        let mut members = serde_json::Map::new();
        for member in inventory["files"]
            .as_array()
            .context("source artifact inventory")?
        {
            let name = member["path"].as_str().context("source artifact path")?;
            let destination = if [
                "README.md",
                "metrics.toml",
                "parameters.toml",
                "run_manifest.json",
            ]
            .contains(&name)
            {
                format!("{source_prefix}/{name}")
            } else {
                super::relative_path(name)
            };
            let target = directory.join(&destination);
            fs::create_dir_all(target.parent().context("artifact parent")?)?;
            ensure!(
                !target.try_exists()?,
                "duplicate presentation export artifact"
            );
            fs::copy(input.join(name), &target)?;
            let copied = file_fingerprint(&target)?;
            ensure!(
                copied["sha256"] == member["sha256"] && copied["bytes"] == member["bytes"],
                "source artifact changed while exporting: {name}"
            );
            members.insert(name.into(), json!(destination));
        }
        let inventory_path = format!("{source_prefix}/inventory.json");
        fs::write(directory.join(&inventory_path), &source_inventory_bytes)?;
        // A source run always has its manifest in its artifact inventory.
        ensure!(
            fs::read(directory.join(format!("{source_prefix}/run_manifest.json")))?
                == source_manifest_bytes,
            "source run manifest changed"
        );
        manifest["presentation_export"] = json!({"schema_version":"1.0","scientific_results_recomputed":false,
            "generator":crate::code_identity(),"source_artifact_manifest":source_inventory,
            "source_inventory_path":inventory_path,"source_member_paths":members,
            "regenerated_files":["README.md","metrics.toml","parameters.toml","run_manifest.json","artifact_manifest.json"]});
        let mut figures = manifest["figures"].as_array().cloned().unwrap_or_default();
        super::organize_figures(directory, &mut figures)?;
        manifest["figures"] = json!(figures);
        super::finalize(directory, &mut manifest, &rows)?;
        pipeline::write_json(&directory.join("run_manifest.json"), &manifest)?;
        pipeline::write_artifact_manifest(directory)?;
        pipeline::verify_artifacts(directory)?;
        ensure!(
            pipeline::verify_artifacts(&input)? == inventory
                && fs::read(input.join("artifact_manifest.json"))? == source_inventory_bytes,
            "source run changed during presentation export"
        );
        Ok(manifest.clone())
    })();
    match result {
        Ok(manifest) => {
            if let Err(error) = publication.publish() {
                let evidence = publication.fail("presentation_publication", &error)?;
                return Err(error.context(format!(
                    "failure evidence retained in {}",
                    evidence.display()
                )));
            }
            Ok(manifest)
        }
        Err(error) => {
            let evidence = publication.fail("presentation_export", &error)?;
            Err(error.context(format!(
                "failure evidence retained in {}",
                evidence.display()
            )))
        }
    }
}
