//! Parameters page: the effective settings per assay context (from each trial's
//! `result.toml`), preview figures reused from the run, and the audit sources.
use anyhow::Result;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use super::{
    PackageArgs,
    html::{blk, escape, img, short},
    index::{self, Recording, Trial},
    link_or_copy,
};

struct Entry<'a> {
    trial: &'a Trial,
    recipe: Option<String>,
    /// `effective_parameters` were recorded (else `values` hold the legacy provenance scalars).
    recorded: bool,
    values: BTreeMap<String, String>,
    sources: BTreeMap<String, String>,
}

fn text_of(value: &toml::Value) -> String {
    match value {
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Lenient: an unreadable or old `result.toml` yields an entry without values.
fn load(trial: &Trial) -> Entry<'_> {
    let table: toml::Table = fs::read_to_string(trial.dir.join("result.toml"))
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or_default();
    let detailed = table.get("detailed");
    let flat = |v: Option<&toml::Value>| -> BTreeMap<String, String> {
        v.and_then(|v| v.as_table())
            .into_iter()
            .flatten()
            .map(|(k, v)| (k.clone(), text_of(v)))
            .collect()
    };
    let effective = detailed.and_then(|d| d.get("effective_parameters"));
    let mut values = flat(effective.and_then(|e| e.get("values")));
    let mut sources = flat(effective.and_then(|e| e.get("source")));
    let recorded = !values.is_empty();
    if !recorded {
        // Old run: show the scalar scoring provenance instead.
        for (k, v) in detailed
            .and_then(|d| d.get("scoring_provenance"))
            .and_then(|p| p.as_table())
            .into_iter()
            .flatten()
            .filter(|(_, v)| !v.is_table() && !v.is_array())
        {
            values.insert(format!("scoring_provenance.{k}"), text_of(v));
            sources.insert(
                format!("scoring_provenance.{k}"),
                "scoring_provenance".into(),
            );
        }
    }
    Entry {
        trial,
        recipe: table
            .get("experiment")
            .and_then(|v| v.as_str())
            .and_then(bekiper_ingress::aggregate::parse_recipe_from_experiment),
        recorded,
        values,
        sources,
    }
}

/// value → trial count, most common first.
fn tally<'a>(values: impl Iterator<Item = &'a str>) -> Vec<(&'a str, usize)> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for v in values {
        *counts.entry(v).or_default() += 1;
    }
    let mut list: Vec<_> = counts.into_iter().collect();
    list.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    list
}

fn context_section(
    out: &Path,
    label: &str,
    slug: &str,
    entries: &[&Entry<'_>],
    body: &mut String,
) -> Result<()> {
    let _ = write!(
        body,
        "<h2 id=\"{slug}\">{} <span class=\"muted\">({} trials)</span></h2>",
        escape(label),
        entries.len()
    );
    let missing = entries.iter().filter(|e| !e.recorded).count();
    if missing > 0 {
        let _ = write!(
            body,
            "<p class=\"muted\">Effective parameters not recorded for this run{}.{}</p>",
            if missing < entries.len() {
                format!(" ({missing} of {} trials)", entries.len())
            } else {
                String::new()
            },
            if entries.iter().any(|e| !e.values.is_empty() && !e.recorded) {
                " Showing the scalar scoring provenance instead."
            } else {
                ""
            }
        );
    }
    let keys: BTreeSet<&String> = entries.iter().flat_map(|e| e.values.keys()).collect();
    let mut rows = String::new();
    let mut varying: Vec<&String> = Vec::new();
    for key in keys {
        let per_trial = || {
            entries
                .iter()
                .map(|e| e.values.get(key).map_or("(absent)", |v| v))
        };
        let values = tally(per_trial());
        let sources = tally(
            entries
                .iter()
                .filter_map(|e| e.sources.get(key).map(String::as_str)),
        );
        let source = sources
            .iter()
            .map(|(s, _)| *s)
            .collect::<Vec<_>>()
            .join(" / ");
        let value = if let [(only, _)] = values.as_slice() {
            escape(only)
        } else {
            varying.push(key);
            let counts: Vec<String> = values
                .iter()
                .map(|(v, n)| format!("{} ×{n}", escape(v)))
                .collect();
            format!("<b>varies</b>: {}", counts.join(", "))
        };
        let _ = write!(
            rows,
            "<tr><td>{}</td><td>{value}</td><td>{}</td></tr>",
            escape(key),
            escape(&source)
        );
    }
    if !rows.is_empty() {
        let state = if missing == 0 { "ok" } else { "warn" };
        let _ = write!(
            body,
            "<div class=\"scroll {}\"><table><tr><th>Key</th><th>Value</th><th>Source</th></tr>{rows}</table></div>",
            blk("parameter_table", &[("state", state)])
        );
    }
    if !varying.is_empty() {
        let _ = write!(
            body,
            "<details><summary>Per-trial values of {} varying keys</summary><div class=\"scroll\"><table><tr><th>Trial</th>",
            varying.len()
        );
        for key in &varying {
            let _ = write!(body, "<th>{}</th>", escape(key));
        }
        body.push_str("</tr>");
        for entry in entries {
            let _ = write!(body, "<tr><td>{}</td>", escape(&entry.trial.id));
            for key in &varying {
                let _ = write!(
                    body,
                    "<td>{}</td>",
                    escape(entry.values.get(*key).map_or("—", |v| v))
                );
            }
            body.push_str("</tr>");
        }
        body.push_str("</table></div></details>");
    }
    previews(out, slug, entries, body)
}

/// The figure that draws feature `kind` in a trial's `figures/`: its own
/// `feature_<kind>_*.png`, else the story panel that groups it, as listed in
/// `render_receipt.json` (e.g. arm sections render inside `story_arm_sequence`).
fn figure_for(figures: &Path, kind: &str) -> Option<std::path::PathBuf> {
    let prefix = format!("feature_{kind}_");
    let mut own: Vec<_> = fs::read_dir(figures)
        .ok()?
        .filter_map(|f| f.ok().map(|f| f.path()))
        .filter(|p| {
            p.extension().is_some_and(|x| x == "png")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with(&prefix))
        })
        .collect();
    own.sort();
    let receipt: serde_json::Value = fs::read_to_string(figures.join("render_receipt.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let stories = receipt["inventory"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| {
            item["selected"] != false
                && item["features"]
                    .as_array()
                    .is_some_and(|f| f.iter().any(|label| label == kind))
        })
        .filter_map(|item| {
            item["stem"]
                .as_str()
                .map(|s| figures.join(format!("{s}.png")))
        });
    own.into_iter().chain(stories).find(|p| p.is_file())
}

/// One representative trial per figure kind: the first (by ID) that has it.
fn previews(out: &Path, slug: &str, entries: &[&Entry<'_>], body: &mut String) -> Result<()> {
    let mut found = String::new();
    let mut absent = Vec::new();
    // Preview figures by feature kind (descriptor `features.previews`); see [`figure_for`].
    let kinds = entries
        .first()
        .map(|e| e.trial.descriptor().features.previews.as_slice())
        .unwrap_or_default();
    for preview in kinds {
        let (kind, title) = (preview.kind.as_str(), preview.title.as_str());
        let hit = entries
            .iter()
            .find_map(|e| figure_for(&e.trial.dir.join("figures"), kind).map(|p| (e.trial, p)));
        let Some((trial, path)) = hit else {
            // Only assays that emit the figure are expected to have it; stay quiet otherwise.
            absent.push(title);
            continue;
        };
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let dir = out.join("figures/parameters").join(slug);
        fs::create_dir_all(&dir)?;
        link_or_copy(&path, &dir.join(&*name))?;
        let _ = write!(
            found,
            "<div class=\"figure\">{}<p class=\"muted\">{} — from trial {}</p></div>",
            img(&format!("figures/parameters/{slug}/{name}"), title, None),
            escape(title),
            escape(&trial.id)
        );
    }
    if found.is_empty() {
        body.push_str("<p class=\"muted\">Preview figures not available for this context.</p>");
    } else {
        let archetype = entries
            .first()
            .map_or("generic", |e| e.trial.descriptor().archetype.as_str());
        let _ = write!(
            body,
            "<div class=\"grid {}\">{found}</div>",
            blk("figure_gallery", &[("archetype", archetype)])
        );
    }
    Ok(())
}

fn has_date(text: &str) -> bool {
    text.as_bytes().windows(10).any(|w| {
        w.iter().enumerate().all(|(i, b)| match i {
            4 | 7 => *b == b'-',
            _ => b.is_ascii_digit(),
        })
    })
}

/// Dated headings of `[review].requirements` (a markdown file relative to the config).
// ponytail: plain-text headings, no links; the requirements file is not packaged.
pub(super) fn decision_log(args: &PackageArgs) -> Result<Option<String>> {
    let Some(config) = args.config.as_ref() else {
        return Ok(None);
    };
    let (review, _) = super::load_review_config(Some(config), None, &args.set)?;
    let name = review
        .requirements
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let Some(text) = super::extensions::read_configured(config, review.requirements)? else {
        return Ok(None);
    };
    let items: String = text
        .lines()
        .filter(|l| l.starts_with('#') && has_date(l))
        .map(|l| format!("<li>{}</li>", escape(l.trim_start_matches('#').trim())))
        .collect();
    Ok((!items.is_empty()).then(|| {
        format!(
            "<section class=\"{}\"><h2>Decision log</h2><p class=\"muted\">Dated decisions recorded in {}.</p><ul>{items}</ul></section>",
            blk("decision_log", &[]),
            escape(&name)
        )
    }))
}

pub(super) fn page(
    out: &Path,
    args: &PackageArgs,
    recordings: &[Recording],
    presentation: &bekiper_core::experiment::ResolvedPresentation,
) -> Result<String> {
    let _ = fs::remove_dir_all(out.join("figures/parameters"));
    let mut entries: Vec<Entry> = recordings
        .iter()
        .flat_map(|r| &r.trials)
        .map(load)
        .collect();
    entries.sort_by(|a, b| a.trial.id.cmp(&b.trial.id));
    // Context = assay + variant; split further by recipe when the protocol differs.
    let mut recipes: BTreeMap<(&str, &str), BTreeSet<&Option<String>>> = BTreeMap::new();
    for e in &entries {
        recipes
            .entry((&e.trial.assay, e.trial.variant.as_deref().unwrap_or("")))
            .or_default()
            .insert(&e.recipe);
    }
    let mut contexts: BTreeMap<(String, String), Vec<&Entry>> = BTreeMap::new();
    for e in &entries {
        let variant = e.trial.variant.as_deref().unwrap_or("");
        let split = recipes[&(e.trial.assay.as_str(), variant)].len() > 1;
        let mut label = short(presentation, &e.trial.assay).to_string();
        for part in [
            Some(variant),
            split.then(|| e.recipe.as_deref().unwrap_or("no recipe")),
        ]
        .into_iter()
        .flatten()
        .filter(|p| !p.is_empty())
        {
            label.push_str(" · ");
            label.push_str(part);
        }
        contexts
            .entry((e.trial.assay.clone(), label))
            .or_default()
            .push(e);
    }

    super::index::ensure_unique_slugs("context", contexts.keys().map(|(_, label)| label.as_str()))?;

    let mut body = String::from(
        "<p class=\"muted\">The effective settings of every analysed trial, by assay context. A key \
with one value shows it once; keys that differ between trials are marked <b>varies</b> with a \
value × trial count list and a per-trial table. Source is the layer that set the value. Preview \
figures are reused from one representative trial of the run. The resolved post-analysis parameters \
per context are in data/cohort/&lt;context&gt;/parameters.toml.</p>",
    );
    for ((_, label), group) in &contexts {
        context_section(out, label, &index::slug(label), group, &mut body)?;
    }
    if let Some(log) = decision_log(args)? {
        body.push_str(&log);
    }
    body.push_str("<h2>Source files (audit)</h2>");
    for (title, path) in [
        ("Project configuration", &args.config),
        ("Analysis plan", &args.plan),
    ] {
        if let Some(path) = path {
            let _ = write!(
                body,
                "<h3>{}</h3><pre>{}</pre>",
                escape(title),
                escape(&fs::read_to_string(path)?)
            );
        }
    }
    Ok(body)
}
