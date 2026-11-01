//! Offline HTML pages (no JavaScript): index, assays, animals, issues,
//! parameters and the reading guide. Colours come from the themekit palette.
use anyhow::Result;
use serde_json::Value;
use std::{collections::BTreeMap, fmt::Write as _, fs, path::Path};

use bekiper_core::experiment::{self, ResolvedExperiment, ResolvedPresentation};

use super::{
    PackageArgs,
    index::{self, Recording},
};

/// An experiment's nav entry: (assay key, page title, short name).
type Assay<'a> = (&'a str, &'a str, &'a str);

/// The presentation blocks of `design/review-package/contract.toml`, in contract
/// order; `design_contract_matches_blocks` keeps the two equal.
pub(crate) const BLOCKS: &[&str] = &[
    "header",
    "nav",
    "context_card",
    "endpoint_table",
    "parameter_table",
    "figure_gallery",
    "protocol_timeline",
    "animal_step_grid",
    "video_embed",
    "object_role_legend",
    "threshold_comparison",
    "qc_flags",
    "callout",
    "decision_log",
];

/// Class list of a block and its variants: `blk-<id> blk-<id>--<axis>-<value> ...`.
pub(crate) fn blk(id: &str, variants: &[(&str, &str)]) -> String {
    debug_assert!(BLOCKS.contains(&id), "unknown block {id}");
    let mut classes = format!("blk-{id}");
    for (axis, value) in variants {
        let _ = write!(classes, " blk-{id}--{axis}-{value}");
    }
    classes
}

/// `ok` when every item is there, `missing` when none is, else `warn`.
fn coverage(have: usize, total: usize) -> &'static str {
    match have {
        n if n == total => "ok",
        0 => "missing",
        _ => "warn",
    }
}

/// Layout only; every colour is a `themekit` custom property.
const STYLE: &str = r#"
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;margin:0;color:var(--ink);background:var(--background);line-height:1.55;-webkit-text-size-adjust:100%}
a{color:var(--link)}a:focus-visible,summary:focus-visible{outline:2px solid var(--highlight);outline-offset:2px;border-radius:3px}
header{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--border);padding:10px 32px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:6px 24px}
header h1{margin:0;font-size:16px;font-weight:700;letter-spacing:-.005em;white-space:nowrap}
header nav{display:flex;align-items:center;gap:2px;min-width:0;max-width:100%;overflow-x:auto;scrollbar-width:none}
header nav a{margin:0;padding:6px 12px;border:1px solid transparent;border-radius:6px;text-decoration:none;font-weight:500;font-size:14px;white-space:nowrap}
header nav a:hover{background:var(--surface-alt)}header nav a[aria-current=page]{background:var(--surface-alt);border-color:var(--accent);color:var(--ink);font-weight:600}
header nav a[href$="issues.html"]{margin-left:14px}
main{padding:28px 32px 56px;max-width:1280px;margin:0 auto}main>:first-child{margin-top:0}
h2{margin:36px 0 12px;font-size:22px;font-weight:650;line-height:1.25;letter-spacing:-.01em;scroll-margin-top:90px}h3{margin:22px 0 8px;font-size:17px;font-weight:600;line-height:1.3;scroll-margin-top:90px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:16px}.card{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:14px 18px;min-width:0}
.card a{font-weight:600;text-decoration:none}.card a:hover{text-decoration:underline}.scroll{overflow-x:auto}
table{display:block;width:max-content;max-width:100%;overflow-x:auto;border-collapse:collapse;background:var(--surface);margin:8px 0 18px;font-size:13px;border:1px solid var(--border);border-radius:6px}
th,td{border:0;border-bottom:1px solid var(--border);padding:6px 12px;text-align:left}tr:nth-child(even) td{background:var(--surface-alt)}
th{background:var(--surface-alt);color:var(--muted);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
img{max-width:100%;height:auto;border:1px solid var(--border);border-radius:4px;background:var(--background)}
.figure{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin:14px 0;scroll-margin-top:90px}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:var(--on-status);margin:0 4px 4px 0}
.good{background:var(--positive)}.bad{background:var(--negative)}.pick{background:var(--highlight)}
.preview{background:var(--surface-alt);border:1px solid var(--highlight);border-radius:8px;padding:8px 12px}
.muted{color:var(--muted);font-size:13px}details{margin:6px 0}summary{cursor:pointer;font-weight:600}
details details{margin-left:18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),1fr));gap:12px}
ul.animals{columns:150px;list-style:none;padding:0}ul.toc{columns:240px}.pager{display:flex;justify-content:space-between;margin:8px 0}
pre{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:12px;overflow:auto;font-size:12px}
.navlabel{color:var(--muted);margin:0 6px 0 14px;padding-left:14px;border-left:1px solid var(--border);font-size:11px;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:8px 0 14px}.flow .arrow{align-self:center;color:var(--muted);font-size:20px}
.step:hover{border-color:var(--accent)}.step{background:var(--surface-alt);border:1px solid var(--border);border-radius:8px;padding:8px 12px;min-width:180px;text-decoration:none;color:var(--ink)}
.step b{display:block}.step span{display:block;color:var(--muted);font-size:12px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,280px),1fr));gap:16px}.gallery a{text-decoration:none}.gallery img{width:100%;aspect-ratio:16/10;object-fit:contain}
video{width:100%;max-width:960px;aspect-ratio:16/9;height:auto;object-fit:contain;border:1px solid var(--border);border-radius:6px;background:var(--background);display:block;margin:8px 0}
.rows{font-size:12px;color:var(--muted)}
@media (max-width:760px){header{flex-direction:column;align-items:stretch;padding:8px 16px}header nav{flex-wrap:wrap;overflow:visible;row-gap:4px}h2,h3,.figure,.card{overflow-wrap:anywhere}.navlabel{display:none}header nav a[href$="issues.html"]{margin-left:0}main{padding:16px 16px 40px}h2{font-size:20px;margin-top:28px}.grid{grid-template-columns:1fr}}
"#;

/// What every page needs besides the recordings and cohort indexes.
pub(crate) struct Site<'a> {
    pub review: &'a bekiper_ingress::config::ReviewConfig,
    /// Resolved experiments, steps, contexts and blocks.
    pub presentation: &'a bekiper_core::experiment::ResolvedPresentation,
    pub groups: &'a BTreeMap<String, String>,
    pub preview: Option<&'a str>,
    pub mode: themekit::Mode,
    /// Per-animal summary figure sidecars (`{path, width, height}`) by animal ID.
    pub animal_figures: &'a BTreeMap<String, Value>,
}

pub(crate) fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// An assay page's path; the key may be a recorded task type, so it is a slug.
fn assay_href(assay: &str) -> String {
    format!("assays/{}.html", index::slug(assay))
}

/// `here` is the nav target of the page itself.
fn page(
    here: &str,
    title: &str,
    depth: usize,
    assays: &[Assay<'_>],
    body: &str,
    extensions: bool,
) -> String {
    let up = "../".repeat(depth);
    let link = |href: &str, label: &str| {
        let current = if href == here {
            " aria-current=\"page\""
        } else {
            ""
        };
        format!("<a href=\"{up}{href}\"{current}>{label}</a>")
    };
    let links: String = assays
        .iter()
        .map(|(assay, _, short)| link(&assay_href(assay), &escape(short)))
        .collect();
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
<title>{t}</title><link rel=\"stylesheet\" href=\"{up}style.css\"></head><body><header class=\"{header}\"><h1>{t}</h1>\
<nav class=\"{nav}\">{overview}{animals}\
<span class=\"navlabel\">Experiment analytics:</span>{links}{issues}\
{parameters}{extensions}{guide}</nav></header><main>{body}</main></body></html>",
        t = escape(title),
        header = blk("header", &[]),
        nav = blk("nav", &[]),
        overview = link("index.html", "Overview"),
        animals = link("animals/index.html", "Animals"),
        issues = link("issues.html", "Issues"),
        parameters = link("parameters.html", "Parameters"),
        extensions = if extensions {
            link("extensions.html", "Extensions")
        } else {
            String::new()
        },
        guide = link("guide.html", "Reading guide")
    )
}

/// Short experiment name (`OFT`, `NORT`, `Y-maze`), else the assay key.
pub(super) fn short<'a>(
    presentation: &'a bekiper_core::experiment::ResolvedPresentation,
    assay: &'a str,
) -> &'a str {
    index::experiment(presentation, assay).map_or(assay, |e| e.short_name.as_str())
}

fn number(value: &Value) -> String {
    match value {
        Value::Number(n) => n
            .as_f64()
            .map(|v| {
                if v.fract() == 0.0 && v.abs() < 1e9 {
                    format!("{v:.0}")
                } else {
                    format!("{v:.4}")
                }
            })
            .unwrap_or_default(),
        Value::Null => "NA".into(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => escape(s),
        other => escape(&other.to_string()),
    }
}

/// Mean ± SEM with 3 significant digits, for compact tables.
pub(crate) fn mean_sem(group: &Value) -> String {
    let fmt = |v: f64| {
        // Zero has no magnitude (log10 = -inf); show it with two decimals.
        let magnitude = if v != 0.0 {
            v.abs().log10().floor() as i32
        } else {
            0
        };
        let digits = (2 - magnitude).clamp(0, 6) as usize;
        format!("{v:.digits$}")
    };
    match (group["mean"].as_f64(), group["sem"].as_f64()) {
        (Some(mean), Some(sem)) => format!("{} ± {}", fmt(mean), fmt(sem)),
        (Some(mean), None) => fmt(mean),
        _ => "NA".into(),
    }
}

fn badges(flags: &BTreeMap<String, bool>) -> String {
    if flags.is_empty() {
        return String::new();
    }
    let state = if flags.values().all(|good| *good) {
        "ok"
    } else {
        "warn"
    };
    let spans: String = flags
        .iter()
        .map(|(name, good)| {
            format!(
                "<span class=\"badge {}\">{}</span>",
                if *good { "good" } else { "bad" },
                escape(name)
            )
        })
        .collect();
    format!(
        "<span class=\"{}\">{spans}</span>",
        blk("qc_flags", &[("state", state)])
    )
}

/// A lazily decoded image; known dimensions reserve its box so pages do not reflow.
pub(super) fn img(src: &str, alt: &str, size: Option<(u64, u64)>) -> String {
    let dims = size.map_or(String::new(), |(w, h)| {
        format!(" width=\"{w}\" height=\"{h}\"")
    });
    format!(
        "<a href=\"{src}\"><img loading=\"lazy\" decoding=\"async\" src=\"{src}\"{dims} alt=\"{}\"></a>",
        escape(alt),
        src = escape(src)
    )
}

fn size_of(value: &Value) -> Option<(u64, u64)> {
    Some((value["width"].as_u64()?, value["height"].as_u64()?))
}

/// Animal IDs in numeric order where they are numbers.
fn animal_order(id: &str) -> (u64, String) {
    (id.parse().unwrap_or(u64::MAX), id.to_string())
}

pub(crate) fn write_site(
    out: &Path,
    recordings: &[Recording],
    cohort: &[Value],
    site: &Site<'_>,
    args: &PackageArgs,
) -> Result<usize> {
    let palette = themekit::Palette::for_mode(site.mode);
    fs::write(
        out.join("style.css"),
        format!("{}\n{}", palette.css(), STYLE.trim_start()),
    )?;
    let extensions = super::extensions::page_body(args)?;
    let home = index_title(site);
    let page = |here: &str, title: &str, depth: usize, assays: &[Assay<'_>], body: &str| {
        page(here, title, depth, assays, body, extensions.is_some())
    };
    // Pages are rebuilt every run; drop those of recordings no longer packaged.
    for dir in ["assays", "animals"] {
        let _ = fs::remove_dir_all(out.join(dir));
        fs::create_dir_all(out.join(dir))?;
    }
    let assays: Vec<Assay<'_>> = site
        .presentation
        .experiments
        .iter()
        .map(|e| (e.key.as_str(), e.title.as_str(), e.short_name.as_str()))
        .filter(|(assay, _, _)| args.preview.as_deref().is_none_or(|p| p == *assay))
        .collect();
    let mut animals: BTreeMap<(u64, String), Vec<&Recording>> = BTreeMap::new();
    for recording in recordings {
        animals
            .entry(animal_order(
                recording.animal.as_deref().unwrap_or("unassigned"),
            ))
            .or_default()
            .push(recording);
    }

    fs::write(
        out.join("index.html"),
        page(
            "index.html",
            &home,
            0,
            &assays,
            &index_page(
                out,
                recordings,
                cohort,
                site,
                &assays,
                &animals,
                extensions.is_some(),
            )?,
        ),
    )?;
    for (assay, title, _) in &assays {
        fs::write(
            out.join(assay_href(assay)),
            page(
                &assay_href(assay),
                title,
                1,
                &assays,
                &assay_page(assay, cohort, site),
            ),
        )?;
    }
    fs::write(
        out.join("animals/index.html"),
        page(
            "animals/index.html",
            "Animals",
            1,
            &assays,
            &animal_gallery(&animals, site),
        ),
    )?;
    let order: Vec<&String> = animals.keys().map(|(_, id)| id).collect();
    for (n, ((_, animal), recs)) in animals.iter().enumerate() {
        let previous = n.checked_sub(1).and_then(|i| order.get(i)).copied();
        let body = animal_page(animal, recs, site, previous, order.get(n + 1).copied());
        fs::write(
            out.join(format!("animals/{}.html", index::slug(animal))),
            page(
                "animals/index.html",
                &format!("Animal {animal}"),
                1,
                &assays,
                &body,
            ),
        )?;
    }
    fs::write(
        out.join("issues.html"),
        page(
            "issues.html",
            "Open issues",
            0,
            &assays,
            &issues_page(recordings, site.presentation),
        ),
    )?;
    fs::write(
        out.join("parameters.html"),
        page(
            "parameters.html",
            "Parameters",
            0,
            &assays,
            &super::parameters::page(out, args, recordings, site.presentation)?,
        ),
    )?;
    if let Some(body) = &extensions {
        fs::write(
            out.join("extensions.html"),
            page("extensions.html", "Research extensions", 0, &assays, body),
        )?;
    }
    let readme = readme(site.presentation);
    fs::write(
        out.join("guide.html"),
        page(
            "guide.html",
            "Reading guide",
            0,
            &assays,
            &markdown(&readme),
        ),
    )?;
    fs::write(out.join("README.md"), &readme)?;
    fs::write(out.join("DATA_DICTIONARY.md"), dictionary(cohort))?;
    Ok(2 + assays.len() + animals.len() + 3 + usize::from(extensions.is_some()))
}

/// `[review] title` (default `Reviewer package`), marked in a preview.
fn index_title(site: &Site<'_>) -> String {
    let title = site.review.title.as_deref().unwrap_or("Reviewer package");
    if site.preview.is_some() {
        format!("{title} (preview)")
    } else {
        title.to_string()
    }
}

fn index_page(
    out: &Path,
    recordings: &[Recording],
    cohort: &[Value],
    site: &Site<'_>,
    assays: &[Assay<'_>],
    animals: &BTreeMap<(u64, String), Vec<&Recording>>,
    extensions: bool,
) -> Result<String> {
    let mut body = String::new();
    if let Some(note) = site.preview {
        write!(
            body,
            "<p class=\"preview {}\">{}</p>",
            blk("callout", &[("state", "warn")]),
            escape(note)
        )?;
    }
    let trials: usize = recordings.iter().map(|r| r.trials.len()).sum();
    let failed = recordings.iter().filter(|r| r.failed.is_some()).count();
    let flagged = recordings
        .iter()
        .filter(|r| {
            r.trials
                .iter()
                .any(|t| index::flags(t.row.as_ref()).values().any(|good| !good))
        })
        .count();
    write!(
        body,
        "<p>{} animals · {} recordings · {trials} scored trials · one composite figure per recording · \
<a href=\"issues.html\">{flagged} recordings with a QC flag</a>{}.</p>",
        animals.len(),
        recordings.len(),
        if failed > 0 {
            format!(" · <b>{failed} render failures</b>")
        } else {
            String::new()
        }
    )?;
    body.push_str("<p class=\"muted\">Descriptive package: recorded measurements and figures only. Registered models, \
corrected tests and contrasts require the hash-bound session metadata and are not included. Start with an assay page, \
then open animals to check individual trials. See the <a href=\"guide.html\">reading guide</a> for how to read a composite.</p>\
<h2>Assays</h2><div class=\"cards\">");
    for (assay, title, _) in assays {
        let count = recordings.iter().filter(|r| r.assay == *assay).count();
        let card = blk(
            "context_card",
            &[
                (
                    "archetype",
                    index::experiment(site.presentation, assay)
                        .map_or("generic", |e| e.archetype.as_str()),
                ),
                ("state", if count > 0 { "ok" } else { "missing" }),
            ],
        );
        write!(
            body,
            "<div class=\"card {card}\"><a href=\"{}\">{}</a><br><span class=\"muted\">{count} recordings</span></div>",
            assay_href(assay),
            escape(title)
        )?;
    }
    if extensions {
        body.push_str(
            "<div class=\"card\"><a href=\"extensions.html\">Research extensions</a><br>\
<span class=\"muted\">Exploratory, not part of the registered analysis</span></div>",
        );
    }
    body.push_str("</div><h2>Criteria at a glance</h2><p class=\"muted\">Primary criteria endpoint per assay, group mean ± SEM \
per session (n animals in the assay tables). Every criteria endpoint is on its assay page.</p>");
    for (assay, title, _) in assays {
        let criteria = index::criteria(site.presentation, assay);
        let Some(endpoint) = criteria.first() else {
            continue;
        };
        for (index, entry) in
            endpoints(cohort, assay).filter(|(_, e)| e["endpoint"] == endpoint.as_str())
        {
            write!(
                body,
                "<h3><a href=\"{href}#{e}\">{title}: {e}</a> <span class=\"muted\">({c})</span></h3>{t}",
                href = assay_href(assay),
                title = escape(title),
                e = escape(endpoint),
                c = escape(index["context"].as_str().unwrap_or("")),
                t = wide_table(entry),
            )?;
        }
    }
    if site.preview.is_some() {
        body.push_str("<h2>Sampled recordings</h2><table><tr><th>Recording</th><th>Why</th></tr>");
        for recording in recordings {
            write!(
                body,
                "<tr><td>{}</td><td>{}</td></tr>",
                recording_link(recording, ""),
                escape(&recording.reasons.join("; "))
            )?;
        }
        body.push_str("</table>");
    }
    body.push_str("<h2>Groups</h2><table><tr><th>Code</th><th>Group</th></tr>");
    for (code, name) in site.groups {
        write!(
            body,
            "<tr><td>{}</td><td>{}</td></tr>",
            escape(code),
            escape(name)
        )?;
    }
    body.push_str("</table><h2>Animals</h2><ul class=\"animals\">");
    for ((_, animal), recs) in animals {
        write!(
            body,
            "<li><a href=\"animals/{}.html\">Animal {}</a> <span class=\"muted\">({})</span></li>",
            index::slug(animal),
            escape(animal),
            recs.len()
        )?;
    }
    body.push_str("</ul><h2>Data</h2><ul>");
    let mut names: Vec<String> = fs::read_dir(out.join("data"))?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|name| name != "cohort")
        .collect();
    names.sort();
    for name in names {
        write!(body, "<li><a href=\"data/{0}\">{0}</a></li>", escape(&name))?;
    }
    for index in cohort {
        let context = index["context"].as_str().unwrap_or("");
        let files: Vec<String> = ["report.pdf", "metrics.toml", "parameters.toml", "README.md"]
            .iter()
            .filter(|f| out.join("data/cohort").join(context).join(f).is_file())
            .map(|f| format!("<a href=\"data/cohort/{}/{f}\">{f}</a>", escape(context)))
            .collect();
        write!(
            body,
            "<li>{} (recorded run originals): {}</li>",
            escape(context),
            files.join(" · ")
        )?;
    }
    body.push_str("</ul>");
    Ok(body)
}

fn endpoints<'a>(
    cohort: &'a [Value],
    assay: &'a str,
) -> impl Iterator<Item = (&'a Value, &'a Value)> + 'a {
    cohort
        .iter()
        .flat_map(|index| {
            index["endpoints"]
                .as_array()
                .into_iter()
                .flatten()
                .map(move |e| (index, e))
        })
        .filter(move |(_, e)| e["assay"] == assay)
}

/// Sessions as rows, groups as columns: `mean ± SEM` with n in the header.
fn wide_table(entry: &Value) -> String {
    let partitions = entry["partitions"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut columns: Vec<&str> = Vec::new();
    for group in partitions
        .iter()
        .flat_map(|p| p["groups"].as_array().into_iter().flatten())
    {
        let name = group["group"].as_str().unwrap_or("");
        if !columns.contains(&name) {
            columns.push(name);
        }
    }
    let mut html = format!(
        "<div class=\"scroll {}\"><table><tr><th>Session</th>",
        blk(
            "endpoint_table",
            &[(
                "state",
                if partitions.is_empty() {
                    "missing"
                } else {
                    "ok"
                }
            )]
        )
    );
    for column in &columns {
        let _ = write!(html, "<th>{}</th>", escape(column));
    }
    html.push_str("</tr>");
    for partition in partitions {
        let _ = write!(
            html,
            "<tr><td>{}</td>",
            escape(partition["label"].as_str().unwrap_or(""))
        );
        let groups = partition["groups"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        for column in &columns {
            match groups.iter().find(|g| g["group"] == *column) {
                Some(group) => {
                    let _ = write!(
                        html,
                        "<td class=\"num\">{} <span class=\"muted\">n={}</span></td>",
                        mean_sem(group),
                        number(&group["n_animals"])
                    );
                }
                None => html.push_str("<td></td>"),
            }
        }
        html.push_str("</tr>");
    }
    html.push_str("</table></div>");
    html
}

fn figures(index: &Value, entry: &Value) -> String {
    let path = index["path"].as_str().unwrap_or("");
    entry["webp"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|figure| {
            let src = format!("../{path}/{}", figure["path"].as_str().unwrap_or(""));
            img(
                &src,
                entry["endpoint"].as_str().unwrap_or(""),
                size_of(figure),
            )
        })
        .collect()
}

/// Appendix families: the endpoint's first word (`quadrant_*`, `arm_*`, ...); singletons go to "other".
fn family(endpoint: &str) -> &str {
    endpoint.split('_').next().unwrap_or(endpoint)
}

/// The assay page's note on threshold contexts, from the descriptors' prose: an
/// experiment with contexts explains them, the others say where they are shown from.
fn contexts_note(presentation: &ResolvedPresentation, assay: &str) -> Option<String> {
    let descriptor = |e: &ResolvedExperiment| experiment::descriptor(&e.id);
    let this = descriptor(index::experiment(presentation, assay)?)?;
    let varies =
        |d: &experiment::ExperimentDescriptor| d.contexts.as_ref().is_some_and(|c| c.varies());
    let (class, text) = if varies(this) {
        let note = this.prose.contexts_note.as_deref()?;
        (
            blk("threshold_comparison", &[("state", "ok")]),
            note.to_string(),
        )
    } else {
        let with = presentation.experiments.iter().find(|e| {
            descriptor(e).is_some_and(|d| {
                varies(d)
                    && d.contexts
                        .as_ref()
                        .is_some_and(|c| c.principal_cohort.is_some())
            })
        })?;
        let d = descriptor(with)?;
        let (threshold, cohort) = (
            d.prose.threshold.as_deref()?,
            d.contexts.as_ref()?.principal_cohort.as_deref()?,
        );
        let (verb, names) = no_context_names(presentation);
        (
            blk("callout", &[("state", "ok")]),
            format!(
                "{names} {verb} not depend on the {} {threshold}; shown from the {cohort} context.",
                with.short_name
            ),
        )
    };
    Some(format!("<p class=\"muted {class}\">{}</p>", escape(&text)))
}

/// Short names of the experiments the principal-cohort contexts do not vary (`OFT and NORT`) with the verb
/// that agrees (`do`, `does`).
fn no_context_names(presentation: &ResolvedPresentation) -> (&'static str, String) {
    let names: Vec<&str> = presentation
        .experiments
        .iter()
        .filter(|e| {
            experiment::descriptor(&e.id).is_some_and(|d| {
                !d.contexts
                    .as_ref()
                    .is_some_and(|c| c.varies() && c.principal_cohort.is_some())
            })
        })
        .map(|e| e.short_name.as_str())
        .collect();
    match names.split_last() {
        Some((last, [])) => ("does", (*last).to_string()),
        Some((last, rest)) => ("do", format!("{} and {last}", rest.join(", "))),
        None => ("do", String::new()),
    }
}

fn assay_page(assay: &str, cohort: &[Value], site: &Site<'_>) -> String {
    let mut body = String::new();
    let order = index::criteria(site.presentation, assay);
    let entries: Vec<(&Value, &Value)> = endpoints(cohort, assay).collect();
    let class = |e: &Value, c: &str| e["class"] == c;
    let archetype =
        index::experiment(site.presentation, assay).map_or("generic", |e| e.archetype.as_str());
    body.push_str(&contexts_note(site.presentation, assay).unwrap_or_default());
    let mut criteria: Vec<_> = entries
        .iter()
        .filter(|(_, e)| class(e, "criteria"))
        .collect();
    criteria.sort_by_key(|(_, e)| {
        order
            .iter()
            .position(|k| e["endpoint"] == k.as_str())
            .unwrap_or(usize::MAX)
    });
    let mut families: BTreeMap<&str, Vec<&(&Value, &Value)>> = BTreeMap::new();
    for item in entries.iter().filter(|(_, e)| class(e, "appendix")) {
        families
            .entry(family(item.1["endpoint"].as_str().unwrap_or("")))
            .or_default()
            .push(item);
    }
    let (mut grouped, mut other): (BTreeMap<&str, Vec<_>>, Vec<_>) = (BTreeMap::new(), Vec::new());
    for (name, items) in families {
        if items.len() > 1 {
            grouped.insert(name, items);
        } else {
            other.extend(items);
        }
    }
    if !other.is_empty() {
        grouped.insert("other", other);
    }

    body.push_str("<h2>Contents</h2><ul class=\"toc\">");
    for (index, entry) in &criteria {
        let _ = write!(
            body,
            "<li><a href=\"#{e}-{c}\">{e}</a> <span class=\"muted\">{c}</span></li>",
            e = escape(entry["endpoint"].as_str().unwrap_or("")),
            c = escape(index["context"].as_str().unwrap_or(""))
        );
    }
    body.push_str("<li><a href=\"#appendix\">Appendix: other recorded endpoints</a></li><li><a href=\"#constant\">Constant endpoints (QC)</a></li></ul>");

    body.push_str("<h2>Criteria endpoints</h2>");
    if criteria.is_empty() {
        let _ = write!(
            body,
            "<p class=\"muted {}\">No criteria endpoints configured ([review.primary_endpoints]).</p>",
            blk("callout", &[("state", "warn")])
        );
    }
    let gallery = blk("figure_gallery", &[("archetype", archetype)]);
    for (index, entry) in &criteria {
        let endpoint = escape(entry["endpoint"].as_str().unwrap_or(""));
        let context = escape(index["context"].as_str().unwrap_or(""));
        let _ = write!(
            body,
            "<div class=\"figure {gallery}\" id=\"{endpoint}-{context}\"><h3 id=\"{endpoint}\">{endpoint} <span class=\"muted\">({context})</span></h3>{}{}</div>",
            figures(index, entry),
            wide_table(entry),
        );
    }
    let _ = write!(
        body,
        "<h2 id=\"appendix\">Appendix: other recorded endpoints</h2><p class=\"muted {}\">Grouped by endpoint family; \
each endpoint shows its group figure, its recorded distributions and the group table.</p>",
        blk("callout", &[("state", "ok")])
    );
    for (name, items) in &grouped {
        let _ = write!(
            body,
            "<details><summary>{} <span class=\"muted\">({} endpoints)</span></summary>",
            escape(name),
            items.len()
        );
        for (index, entry) in items {
            let _ = write!(
                body,
                "<details><summary>{} <span class=\"muted\">({})</span></summary><div class=\"grid\">{}</div>{}</details>",
                escape(entry["endpoint"].as_str().unwrap_or("")),
                escape(index["context"].as_str().unwrap_or("")),
                figures(index, entry),
                wide_table(entry),
            );
        }
        body.push_str("</details>");
    }
    body.push_str("<h2 id=\"constant\">Constant endpoints (QC, not plotted)</h2><table><tr><th>Endpoint</th><th>Context</th><th>Range</th></tr>");
    for (index, entry) in entries.iter().filter(|(_, e)| class(e, "constant")) {
        let range = entry["range"]
            .as_array()
            .map(|r| r.iter().map(number).collect::<Vec<_>>().join(" – "))
            .unwrap_or_default();
        let _ = write!(
            body,
            "<tr><td>{}</td><td>{}</td><td class=\"num\">{range}</td></tr>",
            escape(entry["endpoint"].as_str().unwrap_or("")),
            escape(index["context"].as_str().unwrap_or(""))
        );
    }
    body.push_str("</table>");
    body
}

fn recording_link(recording: &Recording, up: &str) -> String {
    let animal = recording.animal.as_deref().unwrap_or("unassigned");
    format!(
        "<a href=\"{up}animals/{}.html#{1}\">{1}</a>",
        index::slug(animal),
        escape(&recording.id)
    )
}

fn session(recording: &Recording) -> String {
    recording.session_label()
}

/// The protocol in words, from the presentation's sections and steps.
fn protocol_prose(presentation: &ResolvedPresentation) -> String {
    // A proper noun or acronym keeps its capital (`T1 training`, `Y-maze`).
    let lower = |text: &str| {
        let mut chars = text.chars();
        match (chars.next(), chars.next()) {
            (Some(first), Some(second)) if second.is_lowercase() => {
                first.to_lowercase().chain(text.chars().skip(1)).collect()
            }
            _ => text.to_string(),
        }
    };
    let parts: Vec<String> = presentation
        .sections
        .iter()
        .map(|section| {
            if !section.by_timepoint {
                return format!("then the {}", lower(&section.title));
            }
            let mut steps: Vec<_> = presentation
                .experiments
                .iter()
                .flat_map(|e| &e.steps)
                .filter(|s| s.section == section.id)
                .map(|s| (s.order, lower(&s.label)))
                .collect();
            steps.sort();
            let labels: Vec<_> = steps.into_iter().map(|(_, label)| label).collect();
            format!(
                "pre-treatment {}, the same post-treatment",
                labels.join(" → ")
            )
        })
        .collect();
    parts.join(", ")
}

/// Every animal with its summary figure, in numeric order.
fn animal_gallery(animals: &BTreeMap<(u64, String), Vec<&Recording>>, site: &Site<'_>) -> String {
    // ponytail: the first experiment's archetype stands for a mixed protocol.
    let archetype = site
        .presentation
        .experiments
        .first()
        .map_or("generic", |e| e.archetype.as_str());
    let mut body = format!(
        "<p class=\"muted {}\">One page per animal: its summary figure (each recorded endpoint against \
its group, pre and post), then every session in protocol order — {} — each with its composite inspection figure and scored video. \
Experiment-wide analytics are on the assay pages.</p><div class=\"gallery {}\">",
        blk("protocol_timeline", &[("archetype", archetype)]),
        escape(&protocol_prose(site.presentation)),
        blk("figure_gallery", &[]),
    );
    for ((_, animal), recs) in animals {
        let group = index::group_label(recs.iter().find_map(|r| r.row()), site.groups);
        let thumb = site.animal_figures.get(animal).map_or(String::new(), |fig| {
            let dims = size_of(fig).map_or(String::new(), |(w, h)| format!(" width=\"{w}\" height=\"{h}\""));
            format!(
                "<img loading=\"lazy\" decoding=\"async\" src=\"../{}\"{dims} alt=\"Animal {} summary\">",
                fig["path"].as_str().unwrap_or(""),
                escape(animal)
            )
        });
        let _ = write!(
            body,
            "<a class=\"card\" href=\"{}.html\"><b>Animal {}</b><br><span class=\"muted\">{} · {} recordings</span>{thumb}</a>",
            index::slug(animal),
            escape(animal),
            escape(&group),
            recs.len()
        );
    }
    body.push_str("</div>");
    body
}

fn key_value(recording: &Recording, site: &Site<'_>) -> String {
    let keys = index::criteria(site.presentation, &recording.assay);
    let key = Some(&keys).and_then(|keys| {
        keys.iter()
            .find(|k| {
                recording
                    .primary_row()
                    .is_some_and(|r| !r[k.as_str()].is_null())
            })
            .or(keys.first())
    });
    key.map(|key| {
        format!(
            "{} = {}",
            escape(key),
            recording
                .primary_row()
                .map(|r| number(&r[key.as_str()]))
                .unwrap_or_else(|| "no row".into())
        )
    })
    .unwrap_or_default()
}

fn animal_page(
    animal: &str,
    recs: &[&Recording],
    site: &Site<'_>,
    previous: Option<&String>,
    next: Option<&String>,
) -> String {
    let mut body = String::new();
    let previous = previous.map_or("<span></span>".into(), |id| {
        format!(
            "<a href=\"{}.html\">← Animal {}</a>",
            index::slug(id),
            escape(id)
        )
    });
    let next = next.map_or("<span></span>".into(), |id| {
        format!(
            "<a href=\"{}.html\">Animal {} →</a>",
            index::slug(id),
            escape(id)
        )
    });
    let _ = write!(body, "<div class=\"pager\">{previous}{next}</div>");
    let group = index::group_label(recs.iter().find_map(|r| r.row()), site.groups);
    let _ = write!(body, "<p>{}</p>", escape(&group));
    if let Some(fig) = site.animal_figures.get(animal) {
        let _ = write!(
            body,
            "<div class=\"figure\"><h3>Animal summary</h3><p class=\"muted\">Every recorded criteria endpoint of this \
animal (highlighted) against its group's recorded values, pre and post.</p>{}</div>",
            img(
                &format!("../{}", fig["path"].as_str().unwrap_or("")),
                "animal summary",
                size_of(fig)
            )
        );
    }
    let mut recs = recs.to_vec();
    recs.sort_by_key(|r| r.protocol_key());
    // Sections in protocol order (Imen: pre NORT protocol, post NORT protocol, Y-maze, other).
    let mut sections: Vec<(String, Vec<&Recording>)> = Vec::new();
    for recording in recs {
        let heading = recording.heading();
        match sections.last_mut() {
            Some((h, members)) if *h == heading => members.push(recording),
            _ => sections.push((heading, vec![recording])),
        }
    }
    for (heading, members) in &sections {
        let experiment = index::experiment(site.presentation, &members[0].assay);
        let archetype = experiment.map_or("generic", |e| e.archetype.as_str());
        let with_row = members.iter().filter(|r| r.primary_row().is_some()).count();
        let grid = blk(
            "animal_step_grid",
            &[
                ("archetype", archetype),
                ("state", coverage(with_row, members.len())),
            ],
        );
        let _ = write!(
            body,
            "<h2>{}</h2><div class=\"flow {grid}\">",
            escape(heading)
        );
        for (n, recording) in members.iter().enumerate() {
            if n > 0 {
                body.push_str("<span class=\"arrow\">→</span>");
            }
            let flags = index::flags(recording.primary_row());
            let raised: BTreeMap<String, bool> =
                flags.into_iter().filter(|(_, good)| !good).collect();
            let objects_class = index::experiment(site.presentation, &recording.assay)
                .filter(|e| e.blocks.iter().any(|b| b == "object_role_legend"))
                .map_or(String::new(), |e| {
                    let class = blk("object_role_legend", &[("archetype", e.archetype.as_str())]);
                    format!(" class=\"{class}\"")
                });
            let _ = write!(
                body,
                "<a class=\"step\" href=\"#{id}\"><b>{label}</b><span>{key}</span><span{objects_class}>{objects}</span>{badges}</a>",
                id = escape(&recording.id),
                label = escape(&recording.session_label()),
                key = key_value(recording, site),
                objects = escape(&recording.objects().unwrap_or_default()),
                badges = badges(&raised)
            );
        }
        body.push_str("</div>");
        for recording in members {
            body.push_str(&recording_block(recording, site));
        }
    }
    body
}

fn recording_block(recording: &Recording, site: &Site<'_>) -> String {
    let mut html = String::new();
    let row = recording.row();
    let _ = write!(
        html,
        "<div class=\"figure\" id=\"{0}\"><h3>{1} <span class=\"muted\">{0}</span></h3>",
        escape(&recording.id),
        escape(&session(recording))
    );
    for reason in &recording.reasons {
        let _ = write!(html, "<span class=\"badge pick\">{}</span>", escape(reason));
    }
    html.push_str(&badges(&index::flags(row)));
    match &recording.failed {
        Some(error) => {
            let _ = write!(
                html,
                "<p><b>Composite not rendered:</b> {}</p>",
                escape(error)
            );
        }
        None => {
            let size = recording.sidecar.as_ref().and_then(size_of);
            html.push_str(&img(
                &format!("../{}.webp", recording.figure),
                "composite",
                size,
            ));
        }
    }
    if let Some(error) = recording.video.as_ref().and_then(|v| v["error"].as_str()) {
        let _ = write!(
            html,
            "<p class=\"muted\"><b>Video not rendered:</b> {}</p>",
            escape(error)
        );
    } else if let Some(video) = &recording.video {
        let dims = size_of(video).map_or(String::new(), |(w, h)| {
            format!(" width=\"{w}\" height=\"{h}\"")
        });
        let poster = video["poster"]
            .as_str()
            .map_or(String::new(), |p| format!(" poster=\"../{}\"", escape(p)));
        let source = escape(video["path"].as_str().unwrap_or(""));
        let _ = write!(
            html,
            "<details open class=\"{}\"><summary>Scored video{}</summary><video controls preload=\"none\"{poster}{dims}>\
<source src=\"../{source}\" type=\"video/mp4\"></video>",
            blk(
                "video_embed",
                &[
                    (
                        "archetype",
                        recording.trials[0].descriptor().archetype.as_str()
                    ),
                    ("state", "ok")
                ]
            ),
            video["variant"].as_str().map_or(String::new(), |v| format!(
                " <span class=\"muted\">({})</span>",
                escape(v)
            ))
        );
        let rows: Vec<String> = video["rows"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| r["heuristic"].as_str().map(escape))
            .collect();
        if !rows.is_empty() {
            let _ = write!(
                html,
                "<p class=\"rows\">Outlines turn blue while their measure is true and red while false. Timeline strip \
under the video, top to bottom: {}; the white line is the current time.</p>",
                rows.join(" · ")
            );
        }
        let _ = write!(
            html,
            "<p class=\"rows\"><a href=\"../{source}\">Open the video file</a></p></details>"
        );
    }
    let keys = index::criteria(site.presentation, &recording.assay);
    if !keys.is_empty() {
        let state = coverage(
            recording.trials.iter().filter(|t| t.row.is_some()).count(),
            recording.trials.len(),
        );
        let _ = write!(
            html,
            "<table class=\"{}\"><tr><th>Endpoint</th>",
            blk("endpoint_table", &[("state", state)])
        );
        for trial in &recording.trials {
            let _ = write!(html, "<th>{}</th>", escape(&trial.variant_label()));
        }
        html.push_str("</tr>");
        for key in &keys {
            let _ = write!(html, "<tr><td>{}</td>", escape(key));
            for trial in &recording.trials {
                let value = trial
                    .row
                    .as_ref()
                    .map(|r| number(&r[key.as_str()]))
                    .unwrap_or_else(|| "no row".into());
                let _ = write!(html, "<td class=\"num\">{value}</td>");
            }
            html.push_str("</tr>");
        }
        html.push_str("</table>");
    }
    html.push_str("</div>");
    html
}

fn issues_page(recordings: &[Recording], presentation: &ResolvedPresentation) -> String {
    let mut body = String::from(
        "<p class=\"muted\">Everything a reviewer should know is missing, excluded or flagged. \
QC flags are not exclusions.</p>",
    );
    let mut list = |title: &str, items: Vec<String>| {
        let _ = write!(
            body,
            "<h2>{} <span class=\"muted\">({})</span></h2>",
            escape(title),
            items.len()
        );
        if !items.is_empty() {
            body.push_str("<ul>");
            for item in items {
                let _ = write!(body, "<li>{item}</li>");
            }
            body.push_str("</ul>");
        }
    };
    let link = |r: &Recording| recording_link(r, "");
    list(
        "Render failures",
        recordings
            .iter()
            .filter_map(|r| {
                r.failed
                    .as_ref()
                    .map(|e| format!("{} — {}", link(r), escape(e)))
            })
            .collect(),
    );
    list(
        "Video render failures",
        recordings
            .iter()
            .filter_map(|r| {
                let error = r.video.as_ref()?["error"].as_str()?;
                Some(format!("{} — {}", link(r), escape(error)))
            })
            .collect(),
    );
    list(
        "Trials without a post-analysis metrics row",
        recordings
            .iter()
            .flat_map(|r| {
                r.trials
                    .iter()
                    .filter(|t| t.row.is_none())
                    .map(move |t| format!("{} ({})", link(r), escape(&t.id)))
            })
            .collect(),
    );
    list(
        "Recordings without a recorded group",
        recordings
            .iter()
            .filter(|r| r.row().is_some_and(|row| row["group"].is_null()))
            .map(link)
            .collect(),
    );
    // A flag no experiment of the package can raise is not listed.
    let has = |archetype: &str| {
        presentation
            .experiments
            .iter()
            .any(|e| e.archetype.as_str() == archetype)
    };
    for (field, label, good) in index::FLAGS {
        let needs = match *field {
            "passes_15s_exploration_threshold" => "object_exploration",
            "low_usable_triads_flag" => "choice_sequence",
            _ => "",
        };
        if !needs.is_empty() && !has(needs) {
            continue;
        }
        list(
            &format!("Flag: {label}{}", if *good { " — failed" } else { "" }),
            recordings
                .iter()
                .flat_map(|r| {
                    r.trials
                        .iter()
                        .filter(|t| {
                            t.row.as_ref().and_then(|row| row[*field].as_bool()) == Some(!*good)
                        })
                        .map(move |t| format!("{} ({})", link(r), escape(&t.variant_label())))
                })
                .collect(),
        );
    }
    let mut missing: BTreeMap<String, usize> = BTreeMap::new();
    for row in recordings
        .iter()
        .flat_map(|r| r.trials.iter().filter_map(|t| t.row.as_ref()))
    {
        for (key, reason) in row["missing_reasons"].as_object().into_iter().flatten() {
            *missing
                .entry(format!(
                    "{key}: {}",
                    reason.as_str().unwrap_or("unavailable")
                ))
                .or_default() += 1;
        }
    }
    list(
        "Unavailable measurements (endpoint: reason × count)",
        missing
            .into_iter()
            .map(|(k, n)| format!("{} × {n}", escape(&k)))
            .collect(),
    );
    list(
        "Not in this package (requires a registered plan with hash-bound metadata)",
        vec![
            "Registered models, Holm-corrected tests and genotype/treatment contrasts".into(),
            "Paired pre/post changes (e.g. ΔSA) and physical side bias".into(),
        ],
    );
    body
}

fn dictionary(cohort: &[Value]) -> String {
    let mut text = String::from(
        "# Data dictionary\n\nEvery recorded cohort endpoint, its class on the assay pages, \
and the sessions it was recorded for. Units are in the endpoint name (`_s` seconds, `_m` metres, `_pct` percent).\n\n\
| Assay | Endpoint | Class | Context | Sessions |\n|---|---|---|---|---|\n",
    );
    for index in cohort {
        for entry in index["endpoints"].as_array().into_iter().flatten() {
            let sessions: Vec<&str> = entry["partitions"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|p| p["label"].as_str())
                .collect();
            let _ = writeln!(
                text,
                "| {} | `{}` | {} | {} | {} |",
                entry["assay"].as_str().unwrap_or(""),
                entry["endpoint"].as_str().unwrap_or(""),
                entry["class"].as_str().unwrap_or(""),
                index["context"].as_str().unwrap_or(""),
                sessions.join("; ")
            );
        }
    }
    text
}

/// The reading guide's small Markdown subset (headings, lists, bold, code) as HTML.
pub(super) fn markdown(text: &str) -> String {
    let inline = |line: &str| {
        let mut html = String::new();
        let (mut bold, mut code) = (false, false);
        let mut rest = escape(line);
        while !rest.is_empty() {
            if let Some(tail) = rest.strip_prefix("**").filter(|_| !code) {
                html.push_str(if bold { "</b>" } else { "<b>" });
                bold = !bold;
                rest = tail.to_string();
            } else if let Some(tail) = rest.strip_prefix('`') {
                html.push_str(if code { "</code>" } else { "<code>" });
                code = !code;
                rest = tail.to_string();
            } else {
                let c = rest.chars().next().unwrap_or_default();
                html.push(c);
                rest = rest[c.len_utf8()..].to_string();
            }
        }
        html
    };
    let mut html = String::new();
    let mut list: Option<&str> = None;
    let close = |html: &mut String, list: &mut Option<&str>| {
        if let Some(tag) = list.take() {
            let _ = write!(html, "</{tag}>");
        }
    };
    for line in text.lines() {
        let numbered = line
            .split_once(". ")
            .filter(|(n, _)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        if let Some(item) = line.strip_prefix("- ") {
            if list != Some("ul") {
                close(&mut html, &mut list);
                html.push_str("<ul>");
                list = Some("ul");
            }
            let _ = write!(html, "<li>{}</li>", inline(item));
        } else if let Some((n, item)) = numbered {
            if list != Some("ol") {
                close(&mut html, &mut list);
                let _ = write!(html, "<ol start=\"{n}\">");
                list = Some("ol");
            }
            let _ = write!(html, "<li>{}</li>", inline(item));
        } else {
            close(&mut html, &mut list);
            if let Some(title) = line.strip_prefix("## ") {
                let _ = write!(html, "<h2>{}</h2>", inline(title));
            } else if line.starts_with("# ") {
                // The page header already carries the title.
            } else if !line.trim().is_empty() {
                let _ = write!(html, "<p>{}</p>", inline(line));
            }
        }
    }
    close(&mut html, &mut list);
    html
}

/// The reading guide; the sentences about contexts come from the descriptors' prose.
fn readme(presentation: &ResolvedPresentation) -> String {
    let prose = |pick: fn(&experiment::Prose) -> Option<&String>| {
        presentation
            .experiments
            .iter()
            .filter_map(|e| experiment::descriptor(&e.id))
            .find_map(|d| pick(&d.prose).cloned())
    };
    let assay_page = prose(|p| p.assay_page.as_ref()).map(|text| {
        let (verb, names) = no_context_names(presentation);
        if names.is_empty() {
            format!(" {text}.")
        } else {
            format!(" {text}; {names} {verb} not depend on it and are shown once.")
        }
    });
    let composite = prose(|p| p.composite.as_ref());
    format!(
        "{}{}{}{}{}",
        README_HEAD,
        assay_page.unwrap_or_default(),
        README_PAGES,
        composite.map_or(String::new(), |text| format!("{text} ")),
        README_TAIL
    )
}

const README_HEAD: &str = "# Reading guide\n\n\
Open `index.html` in a browser (no network needed). Pages:\n\n\
- **Overview** (`index.html`): counts, the primary criteria endpoint per assay at a glance, groups, animals and data files.\n\
- **Assay pages** (`assays/*.html`): a contents list, then criteria endpoints (group figure, recorded distributions and \
a group mean ± SEM table per session), an appendix of every other recorded endpoint grouped by family, and constant QC \
endpoints.";

const README_PAGES: &str = "\n\
- **Animal pages** (`animals/*.html`): a summary table, then every recording of the animal with its composite figure, \
endpoint values and QC flags; arrows step to the previous and next animal.\n\
- **Issues** (`issues.html`): render failures, missing metric rows, missing groups, QC flags and unavailable measurements.\n\
- **Parameters** (`parameters.html`): the effective settings per assay context, preview figures, and the configuration and plan used.\n\n\
## One composite per trial\n\n\
Each trial has a single figure that reads top to bottom:\n\n\
0. **Header**: animal, group, session, analysed duration, the trial's criteria endpoints and QC badges \
(green = pass, red = fail or warning).\n\
1. **Where**: the arena/maze geometry and the full trajectory (colour = recording time, see the bar under the map), \
the assay result panels (centre/periphery, object discrimination, arm sections per threshold) with their recorded \
summaries underneath, and pose tracking quality.\n\
2. **How each score was decided**: one row per scored measure; the first cell (outlined) is the final mask, the next \
cells are its component conditions and filter stages. Blue = true, red = false, gray = unavailable. Diagnostic channels \
(body proximity, olfaction, whiskers) are shown as final masks only; they are not scoring inputs.\n\
3. **When**: every scored mask (bold) and component on one recording-time axis, with key time series (head angle to \
each object, speeds).\n\
4. **Distributions**: whole-session target composition and sensory agreement, the legend and the trial ID.\n\n";

const README_TAIL: &str = "Every manifest feature is accounted for in the composite's `.json` sidecar \
(`coverage`: drawn, folded or omitted).\n\n\
## Figures and originals\n\n\
Package figures are lossy WebP (visually lossless) in night mode. The recorded run originals under \
`data/cohort/<context>/` (`report.pdf` and the per-figure PDFs) are shipped unchanged as vector records of the run.\n\n\
## Integrity\n\n\
`SHA256SUMS` lists every file (`sha256sum -c SHA256SUMS`); `bekiper review verify <dir|zip>` checks a package or its zip.\n";
