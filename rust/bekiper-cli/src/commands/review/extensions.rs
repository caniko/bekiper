//! Exploratory "Research extensions" page: figures drawn by the Python backend
//! (`bekiper-inspect extension-figures`) from outputs the run already wrote, placed
//! between the prose of `[review].extensions` at its `Figure: <id>` lines. Nothing is
//! rescored and nothing here is part of the registered analysis.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use super::{
    Cancel, PackageArgs,
    html::{escape, img, markdown},
    index::{self, Recording},
    sha256_file,
};

/// Bump with any change to the figures of `bekiper_inspect/plot/extensions.py`.
const EXTENSIONS_VERSION: &str = "2";

/// Package-relative directory of the rendered figures and their `index.json`.
pub(super) const DIR: &str = "figures/extensions";

/// A `[review]` file named relative to the config: `None` when the key is unset, an error
/// naming the path when it is set but cannot be read.
pub(super) fn read_configured(config: &Path, name: Option<PathBuf>) -> Result<Option<String>> {
    let Some(name) = name else { return Ok(None) };
    let path = config.parent().unwrap_or(Path::new("")).join(name);
    fs::read_to_string(&path)
        .map(Some)
        .with_context(|| format!("[review] file {} is unreadable", path.display()))
}

/// The configured markdown, resolved relative to the config like the decision log.
fn source(config: Option<&Path>, set: &[String]) -> Result<Option<String>> {
    let Some(config) = config else {
        return Ok(None);
    };
    let (review, _) = super::load_review_config(Some(config), None, set)?;
    read_configured(config, review.extensions)
}

/// Everything the figures consume: version, colour mode, group names and the trial inputs.
fn reuse_key(
    recordings: &[Recording],
    groups: &BTreeMap<String, String>,
    mode: themekit::Mode,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(EXTENSIONS_VERSION);
    hasher.update(mode.as_str());
    hasher.update(serde_json::to_vec(groups)?);
    for trial in recordings.iter().flat_map(|r| &r.trials) {
        hasher.update(trial.id.as_bytes());
        for path in index::inputs(&trial.dir)? {
            hasher.update(sha256_file(&path)?);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Draws the figures into `<package>/figures/extensions`; `Ok(false)` when `[review] extensions`
/// is unset. A finished directory is reused only under the same `reuse_key` (not in a preview).
pub(super) fn render(
    args: &PackageArgs,
    recordings: &[Recording],
    groups: &BTreeMap<String, String>,
    mode: themekit::Mode,
    cancel: &Cancel,
) -> Result<bool> {
    if source(args.config.as_deref(), &args.set)?.is_none() {
        return Ok(false);
    }
    let target = args.output.join(DIR);
    let index_path = target.join("index.json");
    let key = reuse_key(recordings, groups, mode)?;
    let cached_key = fs::read(&index_path)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .map(|index| index["key"].clone());
    if args.preview.is_some() || cached_key != Some(json!(key)) {
        let _ = fs::remove_dir_all(&target);
        let work = args.output.join("logs");
        fs::create_dir_all(&work)?;
        let groups_path = work.join("extension_groups.json");
        fs::write(&groups_path, serde_json::to_vec(groups)?)?;
        let mut command = Command::new(&args.renderer);
        command
            .arg("extension-figures")
            .arg("--run")
            .arg(&args.run)
            .arg("--output")
            .arg(&target)
            .arg("--groups")
            .arg(&groups_path)
            .arg("--theme")
            .arg(mode.as_str());
        let status = bekiper_post_analysis::pipeline::run_child(
            &mut command,
            &work.join("extensions.stdout.log"),
            &work.join("extensions.stderr.log"),
            &|| cancel.cancelled(),
        )?;
        ensure!(status.success(), "extension figures failed; see logs/");
        ensure!(
            index_path.is_file(),
            "extension figures wrote no index.json"
        );
        let mut index: Value = serde_json::from_slice(&fs::read(&index_path)?)?;
        index["key"] = json!(key);
        fs::write(&index_path, serde_json::to_vec_pretty(&index)?)?;
    }
    Ok(true)
}

/// Body of `extensions.html`; `None` (no page, no link) when `[review] extensions` is unset.
pub(super) fn page_body(args: &PackageArgs) -> Result<Option<String>> {
    body_for(args.config.as_deref(), &args.set, &args.output)
}

fn body_for(config: Option<&Path>, set: &[String], out: &Path) -> Result<Option<String>> {
    let Some(text) = source(config, set)? else {
        return Ok(None);
    };
    let index: Value = fs::read(out.join(DIR).join("index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    Ok(Some(body(&text, &index)))
}

/// The markdown with each `Figure: <id>` line replaced by that figure, or by a muted note
/// naming why it could not be drawn ("needs cohort: ...").
fn body(text: &str, index: &Value) -> String {
    let mut html = String::from(
        "<p class=\"preview\"><b>Exploratory — not part of the registered analysis.</b> \
Hypothesis-generating views of outputs the run already wrote; nothing was rescored and no P value here supports a claim.</p>",
    );
    let mut chunk = String::new();
    for line in text.lines() {
        if let Some(id) = line.trim().strip_prefix("Figure:") {
            html.push_str(&markdown(&std::mem::take(&mut chunk)));
            html.push_str(&figure(id.trim(), index));
        } else {
            chunk.push_str(line);
            chunk.push('\n');
        }
    }
    html.push_str(&markdown(&chunk));
    html
}

fn figure(id: &str, index: &Value) -> String {
    let entry = index["figures"]
        .as_array()
        .and_then(|all| all.iter().find(|f| f["id"] == id));
    let Some(entry) = entry else {
        return format!("<p class=\"muted\">{}: not rendered.</p>", escape(id));
    };
    let title = entry["title"].as_str().unwrap_or(id);
    match (entry["status"].as_str(), entry["file"].as_str()) {
        (Some("ok"), Some(file)) => {
            let size = entry["width"].as_u64().zip(entry["height"].as_u64());
            format!(
                "<div class=\"figure\" id=\"{}\">{}<p class=\"muted\">{}</p></div>",
                escape(id),
                img(&format!("{DIR}/{file}"), title, size),
                escape(title)
            )
        }
        (status, _) => format!(
            "<p class=\"muted\" id=\"{}\">{}: {}</p>",
            escape(id),
            escape(status.unwrap_or("not rendered")),
            escape(entry["reason"].as_str().unwrap_or(""))
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "# Research extensions\n\n**Exploratory, not part of the registered analysis.**\n\n\
## First idea\n\n- **Hypothesis.** One.\n\nFigure: a\n\n## Second idea\n\nFigure: b\n\nFigure: missing\n";

    fn index() -> Value {
        json!({"color_mode": "dark", "figures": [
            {"id": "a", "title": "Fig A", "status": "ok", "reason": "", "file": "a.png", "width": 10, "height": 20},
            {"id": "b", "title": "Fig B", "status": "needs cohort", "reason": "One animal only.", "file": null}]})
    }

    #[test]
    fn banner_then_figures_sit_at_their_markers() {
        let html = body(TEXT, &index());
        assert!(html.starts_with(
            "<p class=\"preview\"><b>Exploratory — not part of the registered analysis."
        ));
        let (first, figure, second) = (
            html.find("First idea").unwrap(),
            html.find("src=\"figures/extensions/a.png\"").unwrap(),
            html.find("Second idea").unwrap(),
        );
        assert!(first < figure && figure < second);
        assert!(html.contains("width=\"10\" height=\"20\""));
        assert!(!html.contains("Figure: a"));
    }

    #[test]
    fn unrendered_figures_become_a_muted_note() {
        let html = body(TEXT, &index());
        assert!(html.contains("<p class=\"muted\" id=\"b\">needs cohort: One animal only.</p>"));
        assert!(!html.contains("b.png"));
        assert!(html.contains("missing: not rendered."));
        assert!(body(TEXT, &Value::Null).contains("a: not rendered."));
    }

    #[test]
    fn none_unless_the_config_names_a_file() {
        let root = tempfile::tempdir().unwrap();
        assert!(body_for(None, &[], root.path()).unwrap().is_none());
        let config = root.path().join("config.toml");
        fs::write(&config, "[review]\nworkers = 1\n").unwrap();
        assert!(body_for(Some(&config), &[], root.path()).unwrap().is_none());
        fs::write(&config, "[review]\nextensions = \"ext.md\"\n").unwrap();
        let error = body_for(Some(&config), &[], root.path()).unwrap_err();
        assert!(
            format!("{error:#}").contains("ext.md"),
            "a configured but unreadable file is an error naming it: {error:#}"
        );
        fs::write(root.path().join("ext.md"), TEXT).unwrap();
        fs::create_dir_all(root.path().join(DIR)).unwrap();
        fs::write(
            root.path().join(DIR).join("index.json"),
            index().to_string(),
        )
        .unwrap();
        let html = body_for(Some(&config), &[], root.path()).unwrap().unwrap();
        assert!(html.contains("figures/extensions/a.png") && html.contains("needs cohort"));
    }
}
