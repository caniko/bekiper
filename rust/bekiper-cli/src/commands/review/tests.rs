use super::*;
use std::os::unix::fs::PermissionsExt;

/// A stand-in plot backend: writes placeholder WebPs, sidecars and a cohort index,
/// and appends one line per composite to `renders.log`.
const FAKE_RENDERER: &str = r#"#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "composite":
    for job in json.loads(Path(args[2]).read_text()):
        out = Path(job["out"]); out.parent.mkdir(parents=True, exist_ok=True)
        webp = out.with_suffix(".webp"); webp.write_bytes(b"RIFF" + job["key"].encode())
        out.with_suffix(".json").write_text(json.dumps({"key": job["key"], "color_mode": job["theme"],
            "width": 2420, "height": 4000,
            "webp_sha256": hashlib.sha256(webp.read_bytes()).hexdigest(), "coverage": {}}))
        with open(Path(__file__).parent / "renders.log", "a") as log: log.write(job["out"] + "\n")
elif args[0] == "animal-summary":
    for job in json.loads(Path(args[2]).read_text()):
        out = Path(job["out"]); out.parent.mkdir(parents=True, exist_ok=True)
        webp = out.with_suffix(".webp"); webp.write_bytes(b"RIFF" + job["key"].encode())
        out.with_suffix(".json").write_text(json.dumps({"key": job["key"], "width": 1680, "height": 900,
            "panels": len(job["panels"]), "webp_sha256": hashlib.sha256(webp.read_bytes()).hexdigest()}))
        (out.parent / (out.name + ".job.json")).write_text(json.dumps(job))
elif args[0] == "extension-figures":
    out = Path(args[args.index("--output") + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / "a.png").write_bytes(b"PNG")
    with open(Path(__file__).parent / "renders.log", "a") as log: log.write("extensions\n")
    (out / "index.json").write_text(json.dumps({"color_mode": args[args.index("--theme") + 1], "figures": [
        {"id": "a", "title": "Fig A", "status": "ok", "reason": "", "file": "a.png", "width": 10, "height": 20},
        {"id": "b", "title": "Fig B", "status": "needs cohort", "reason": "One animal only.", "file": None}]}))
else:
    out = Path(args[args.index("--output") + 1]); (out / "criteria/oft").mkdir(parents=True, exist_ok=True)
    (out / "criteria/oft/center_time_pct.webp").write_bytes(b"RIFF")
    context = Path(args[args.index("--context") + 1]).name
    theme = args[args.index("--theme") + 1]
    with open(Path(__file__).parent / "renders.log", "a") as log: log.write(f"cohort {context} {theme}\n")
    (out / "cohort_index.json").write_text(json.dumps({"context": context, "color_mode": theme, "endpoints": [
        {"assay": "oft", "endpoint": "center_time_pct", "class": "criteria",
         "webp": [{"path": "criteria/oft/center_time_pct.webp", "width": 780, "height": 500}],
         "partitions": [{"label": "oft · post · A", "groups": [{"group": "G2", "n_animals": 1, "mean": 10.0, "sem": None}]}]},
        {"assay": "oft", "endpoint": "analysis_duration_s", "class": "constant", "range": [300, 300], "partitions": [],
         "webp": []}]}))
"#;

fn trial(data: &Path, id: &str, task: &str, video: &str) {
    trial_with(data, id, task, video, &Recorded::default());
}

/// What a synthetic trial records beyond its task type.
#[derive(Default)]
struct Recorded<'a> {
    perimeters: &'a [&'a str],
    kinds: &'a [&'a str],
    recipe: Option<&'a str>,
    /// `session_context` fields (`animal`, `phase_part`, ...).
    session: Value,
}

fn trial_with(data: &Path, id: &str, task: &str, video: &str, recorded: &Recorded<'_>) {
    let dir = data.join(id);
    fs::create_dir_all(dir.join("frames")).unwrap();
    let mut manifest = format!("label = \"{id}\"\n");
    for label in recorded.perimeters {
        manifest += &format!("[[perimeters]]\nlabel = \"{label}\"\n");
    }
    for kind in recorded.kinds {
        manifest += &format!("[[feature_inspections]]\nkind = \"{kind}\"\n");
    }
    fs::write(dir.join(format!("{id}_inspection.toml")), manifest).unwrap();
    fs::write(dir.join(format!("{id}_evaluated.parquet")), id).unwrap();
    let mut experiment = json!({"task_type": task, "video": video});
    if let Some(recipe) = recorded.recipe {
        experiment["detection"] = json!({"recipe": recipe});
    }
    fs::write(
        dir.join("trial_context.json"),
        json!({"experiment": experiment, "session_context": recorded.session}).to_string(),
    )
    .unwrap();
}

fn fixture(root: &Path) -> PackageArgs {
    let data = root.join("run/data");
    trial(&data, "oft_a", "open_field", "/v/oft1.mp4");
    trial(&data, "ym_x__proximal45", "y_maze", "/v/ym1.mp4");
    trial(&data, "ym_x__compare_0pct_dwell0s", "y_maze", "/v/ym1.mp4");
    let context = data.join("_post_analysis/cohort-1/primary45");
    fs::create_dir_all(context.join("_internal")).unwrap();
    fs::write(context.parent().unwrap().join("run_manifest.json"), "{}").unwrap();
    fs::write(context.join("report.pdf"), "%PDF").unwrap();
    fs::write(
        context.join("_internal/metrics.json"),
        json!([
            {"analysis_id": "oft_a", "animal_id": "7", "group": "2", "sex": "female", "assay": "oft",
             "presentation_label": "oft · post · A · recorded", "center_time_pct": 10.0,
             "window_complete_flag": false},
            {"analysis_id": "ym_x__proximal45", "animal_id": "7", "group": "2", "assay": "ymaze",
             "spontaneous_alternation_pct": 55.0, "low_usable_triads_flag": true},
        ])
        .to_string(),
    )
    .unwrap();
    fs::write(root.join("run/imen_results.ods"), "ods").unwrap();
    let plan = root.join("plan.toml");
    fs::write(
        &plan,
        "[inputs.group_mapping]\nG2 = { five_xfad = \"AD\", hcar1 = \"KO\", treatment = \"vehicle\" }\n\
         [inputs.session_key_mapping.group]\n\"2\" = \"G2\"\n",
    )
    .unwrap();
    let config = root.join("config.toml");
    fs::write(
        &config,
        "[review]\nworkers = 2\n[review.primary_endpoints]\noft = [\"center_time_pct\"]\n\
         ymaze = [\"spontaneous_alternation_pct\"]\n",
    )
    .unwrap();
    let renderer = root.join("fake-renderer");
    fs::write(&renderer, FAKE_RENDERER).unwrap();
    fs::set_permissions(&renderer, fs::Permissions::from_mode(0o755)).unwrap();
    PackageArgs {
        run: root.join("run"),
        output: root.join("pkg"),
        config: Some(config),
        plan: Some(plan),
        workers: None,
        trial: Vec::new(),
        zip: true,
        preview: None,
        per_stratum: 3,
        theme: None,
        videos: false,
        video_source: None,
        video_renderer: None,
        set: Vec::new(),
        renderer,
    }
}

fn links(html: &str) -> Vec<String> {
    let mut found = Vec::new();
    for attribute in ["href=\"", "src=\""] {
        for part in html.split(attribute).skip(1) {
            let target = part.split('"').next().unwrap_or("");
            found.push(target.split('#').next().unwrap_or("").to_string());
        }
    }
    found
}

/// Every relative link of every page resolves inside the package.
fn assert_links_resolve(out: &Path) {
    let mut pages = vec![
        out.join("index.html"),
        out.join("issues.html"),
        out.join("parameters.html"),
    ];
    pages.extend(
        fs::read_dir(out.join("assays"))
            .unwrap()
            .map(|e| e.unwrap().path()),
    );
    pages.extend(
        fs::read_dir(out.join("animals"))
            .unwrap()
            .map(|e| e.unwrap().path()),
    );
    for page in &pages {
        let html = fs::read_to_string(page).unwrap();
        for link in links(&html) {
            let target = page.parent().unwrap().join(&link);
            assert!(
                target.exists(),
                "{} links to missing {link}",
                page.display()
            );
        }
    }
}

#[test]
fn package_groups_variants_links_resolve_seals_and_resumes() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path());
    let summary = package(&args).unwrap();
    // OFT + one Y-maze recording carrying both variants.
    assert_eq!(summary["recordings"], 2);
    assert_eq!(summary["rendered_now"], 2);
    let out = &args.output;
    assert!(out.join("figures/trials/7/oft_a.webp").is_file());
    assert!(out.join("figures/trials/7/ym_x.webp").is_file());
    let jobs = fs::read_to_string(out.join("logs/composite_00.jobs.json")).unwrap_or_default()
        + &fs::read_to_string(out.join("logs/composite_01.jobs.json")).unwrap_or_default();
    assert!(jobs.contains("0% (no dwell)") && jobs.contains("45%"));
    assert!(jobs.contains("G2 · AD · HCAR1-KO · vehicle"));

    assert_links_resolve(out);
    // The variant without a metrics row and both QC flags reach the issues page.
    let issues = fs::read_to_string(out.join("issues.html")).unwrap();
    assert!(issues.contains("ym_x__compare_0pct_dwell0s"));
    assert!(issues.contains("300 s window complete") && issues.contains("low usable triads"));
    let oft = fs::read_to_string(out.join("assays/oft.html")).unwrap();
    assert!(oft.contains("center_time_pct") && oft.contains("analysis_duration_s"));

    // Sealed: directory and zip verify; a one-byte change fails.
    let files = summary["files"].as_u64().unwrap() as usize;
    assert_eq!(seal::verify_dir(out).unwrap(), files);
    assert_eq!(seal::verify_zip(&out.with_extension("zip")).unwrap(), files);
    let report = out.join("data/cohort/primary45/report.pdf");
    let original = fs::read(&report).unwrap();
    fs::remove_file(&report).unwrap();
    fs::write(&report, b"%PDX").unwrap();
    assert!(seal::verify_dir(out).is_err());
    fs::remove_file(&report).unwrap();
    fs::write(&report, &original).unwrap();

    // Night mode by default: themekit CSS, mode in every job, sized images.
    let css = fs::read_to_string(out.join("style.css")).unwrap();
    assert!(css.starts_with(&themekit::DARK.css()));
    assert!(jobs.contains("\"theme\":\"dark\""));
    let animal = fs::read_to_string(out.join("animals/7.html")).unwrap();
    assert!(
        animal.contains("width=\"2420\" height=\"4000\"") && animal.contains("decoding=\"async\"")
    );
    assert!(out.join("guide.html").is_file());

    // Resume: nothing is re-rendered.
    let again = package(&args).unwrap();
    assert_eq!(again["rendered_now"], 0);
    let renders = fs::read_to_string(root.path().join("renders.log")).unwrap();
    let composites = |log: &str| log.lines().filter(|l| !l.starts_with("cohort ")).count();
    assert_eq!(composites(&renders), 2);
    assert_eq!(
        renders.lines().filter(|l| l.starts_with("cohort ")).count(),
        1
    );

    // Switching to light re-renders every composite and the cohort figures.
    let mut light = args.clone();
    light.theme = Some("light".into());
    assert_eq!(package(&light).unwrap()["rendered_now"], 2);
    let renders = fs::read_to_string(root.path().join("renders.log")).unwrap();
    assert_eq!(composites(&renders), 4);
    assert!(
        renders
            .lines()
            .any(|l| l.starts_with("cohort ") && l.ends_with(" light"))
    );
    assert!(
        fs::read_to_string(out.join("style.css"))
            .unwrap()
            .starts_with(&themekit::LIGHT.css())
    );
    // A package directory from another run is refused.
    let mut other = args.clone();
    other.run = root.path().join("other");
    fs::create_dir_all(other.run.join("data")).unwrap();
    assert!(package(&other).is_err());
}

#[test]
fn extensions_page_is_linked_rendered_once_and_sealed_only_when_configured() {
    let root = tempfile::tempdir().unwrap();
    let mut args = fixture(root.path());
    package(&args).unwrap();
    assert!(!args.output.join("extensions.html").exists());
    assert!(
        !fs::read_to_string(args.output.join("index.html"))
            .unwrap()
            .contains("extensions.html")
    );

    let config = args.config.clone().unwrap();
    let text = fs::read_to_string(&config).unwrap();
    fs::write(
        &config,
        text.replacen("[review]\n", "[review]\nextensions = \"ext.md\"\n", 1),
    )
    .unwrap();
    fs::write(
        root.path().join("ext.md"),
        "## Idea\n\nFigure: a\n\n## Other\n\nFigure: b\n",
    )
    .unwrap();
    args.output = root.path().join("pkg2");
    let summary = package(&args).unwrap();
    let out = &args.output;
    let page = fs::read_to_string(out.join("extensions.html")).unwrap();
    assert!(page.contains("Exploratory — not part of the registered analysis"));
    assert!(
        page.contains("figures/extensions/a.png")
            && page.contains("needs cohort: One animal only.")
    );
    for html in [
        "index.html",
        "issues.html",
        "parameters.html",
        "guide.html",
        "animals/7.html",
    ] {
        let text = fs::read_to_string(out.join(html)).unwrap();
        assert!(
            text.contains("extensions.html\">Extensions</a>")
                || text.contains("extensions.html\" aria-current"),
            "{html}"
        );
    }
    assert!(page.contains("aria-current=\"page\">Extensions"));
    for link in links(&page) {
        assert!(
            out.join(&link).exists(),
            "extensions.html links to missing {link}"
        );
    }
    // The figure and index are sealed like every other file.
    let sums = fs::read_to_string(out.join("SHA256SUMS")).unwrap();
    assert!(sums.contains("figures/extensions/a.png") && sums.contains("extensions.html"));
    assert_eq!(
        seal::verify_dir(out).unwrap(),
        summary["files"].as_u64().unwrap() as usize
    );
    // Resume does not redraw; a changed trial input does.
    package(&args).unwrap();
    let renders = fs::read_to_string(root.path().join("renders.log")).unwrap();
    assert_eq!(renders.lines().filter(|l| *l == "extensions").count(), 1);
    fs::write(
        root.path().join("run/data/oft_a/oft_a_evaluated.parquet"),
        "changed",
    )
    .unwrap();
    package(&args).unwrap();
    let renders = fs::read_to_string(root.path().join("renders.log")).unwrap();
    assert_eq!(renders.lines().filter(|l| *l == "extensions").count(), 2);
}

#[test]
fn preview_samples_strata_and_edge_cases_of_one_assay_and_rerenders() {
    let root = tempfile::tempdir().unwrap();
    let mut args = fixture(root.path());
    let data = root.path().join("run/data");
    // Stratum A: five recordings ranked 10..50; stratum B: two. a2 carries a QC failure.
    let mut rows = vec![
        json!({"analysis_id": "ym_x__proximal45", "animal_id": "7", "group": "2", "assay": "ymaze",
        "presentation_label": "ymaze · post · unrecorded · proximal45", "variant": "proximal45",
        "spontaneous_alternation_pct": 55.0}),
    ];
    for (id, session, value) in [
        ("a1", "A", 10.0),
        ("a2", "A", 20.0),
        ("a3", "A", 30.0),
        ("a4", "A", 40.0),
        ("a5", "A", 50.0),
        ("b1", "B", 5.0),
        ("b2", "B", 6.0),
    ] {
        trial(&data, id, "open_field", &format!("/v/{id}.mp4"));
        rows.push(json!({"analysis_id": id, "animal_id": id, "group": "2", "assay": "oft",
            "presentation_label": format!("oft · post · {session} · recorded"), "center_time_pct": value,
            "window_complete_flag": id != "a2"}));
    }
    let context = data.join("_post_analysis/cohort-1/primary45/_internal/metrics.json");
    fs::write(&context, json!(rows).to_string()).unwrap();
    args.preview = Some("oft".into());
    args.zip = false;
    args.output = root.path().join("preview");

    let summary = package(&args).unwrap();
    // min/median/max per stratum plus the flagged one and oft_a without a row; a4 is not sampled.
    let figures = |out: &Path| {
        let mut ids: Vec<String> = walk(&out.join("figures/trials"))
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "webp"))
            .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(
        figures(&args.output),
        ["a1", "a2", "a3", "a5", "b1", "b2", "oft_a"]
    );
    assert_eq!(summary["recordings"], 7);
    assert!(
        !args.output.join("assays/ymaze.html").exists()
            && args.output.join("assays/oft.html").is_file()
    );
    assert_links_resolve(&args.output);
    let index = fs::read_to_string(args.output.join("index.html")).unwrap();
    assert!(index.contains("Preview: oft, 7 of 8 recordings"));
    assert!(index.contains("max center_time_pct · oft · post · A") && index.contains("QC flag"));
    assert!(index.contains("no metrics row"));
    let a5 = fs::read_to_string(args.output.join("animals/a5.html")).unwrap();
    assert!(a5.contains("badge pick"));
    assert_eq!(
        seal::verify_dir(&args.output).unwrap(),
        summary["files"].as_u64().unwrap() as usize
    );

    // A second round re-renders the same deterministic sample.
    let again = package(&args).unwrap();
    assert_eq!(again["rendered_now"], 7);
    assert_eq!(
        figures(&args.output),
        ["a1", "a2", "a3", "a5", "b1", "b2", "oft_a"]
    );
    // A full build into a preview directory, or another preview, is refused.
    let mut full = args.clone();
    full.preview = None;
    assert!(
        package(&full)
            .unwrap_err()
            .to_string()
            .contains("oft preview")
    );
    full.preview = Some("nort".into());
    assert!(package(&full).is_err());
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

#[test]
fn mean_sem_keeps_three_significant_digits_and_handles_zero() {
    assert_eq!(
        html::mean_sem(&json!({"mean": 0.0, "sem": 0.0})),
        "0.00 ± 0.00"
    );
    assert_eq!(
        html::mean_sem(&json!({"mean": 54.321, "sem": 2.345})),
        "54.3 ± 2.35"
    );
    assert_eq!(
        html::mean_sem(&json!({"mean": -0.01234, "sem": null})),
        "-0.0123"
    );
    assert_eq!(html::mean_sem(&json!({"mean": null})), "NA");
}

fn result(dir: &Path, experiment: &str, body: &str) {
    fs::write(
        dir.join("result.toml"),
        format!("experiment = \"{experiment}\"\n{body}"),
    )
    .unwrap();
}

#[test]
fn parameters_page_groups_contexts_marks_varying_keys_and_copies_figures() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path());
    let data = root.path().join("run/data");
    trial(&data, "ym_y__proximal45", "y_maze", "/v/ym2.mp4");
    trial(&data, "nort_1", "object_recognition", "/v/nort1.mp4");
    let effective = |age: u32| {
        format!(
            "[detailed.effective_parameters.values]\n\"exploration.whisker.apex_fraction\" = 0.9\n\
             \"exploration.whisker.age_days\" = {age}\n\
             [detailed.effective_parameters.source]\n\"exploration.whisker.apex_fraction\" = \"toml\"\n\
             \"exploration.whisker.age_days\" = \"derived\"\n"
        )
    };
    result(
        &data.join("ym_x__proximal45"),
        "7_seq1_ymaze",
        &effective(60),
    );
    result(
        &data.join("ym_y__proximal45"),
        "8_seq1_ymaze",
        &effective(90),
    );
    result(&data.join("nort_1"), "9_seq1_nort", &effective(60));
    // ym_x__compare_0pct_dwell0s is an old run: provenance scalars only.
    result(
        &data.join("ym_x__compare_0pct_dwell0s"),
        "7_seq1_ymaze",
        "[detailed.scoring_provenance]\ncomparison_policy = \"retained\"\n",
    );
    // oft_a has no result.toml at all and no figures.
    for (id, name) in [
        (
            "ym_y__proximal45",
            "feature_ymaze_arm_sections_ymaze_arm_sections.png",
        ),
        // Arm sections render inside a story panel; only the receipt names them.
        ("ym_x__proximal45", "story_arm_sequence.png"),
        ("nort_1", "feature_detection_fields_detection_fields.png"),
    ] {
        let figures = data.join(id).join("figures");
        fs::create_dir_all(&figures).unwrap();
        fs::write(figures.join(name), format!("png {id}")).unwrap();
    }
    fs::write(
        data.join("ym_x__proximal45/figures/render_receipt.json"),
        r#"{"inventory":[{"stem":"story_arm_sequence","selected":true,"features":["ymaze_arm_sections","arm_sequence"]}]}"#,
    )
    .unwrap();
    let summary = package(&args).unwrap();
    let out = &args.output;
    let html = fs::read_to_string(out.join("parameters.html")).unwrap();
    for context in [
        "Y-maze · proximal45",
        "Y-maze · compare_0pct_dwell0s",
        "NORT",
        "OFT",
    ] {
        assert!(html.contains(&format!("{context} <span")), "{context}");
    }
    // Constant key once with its source; varying key marked with a count list and per-trial table.
    assert!(html.contains("<td>exploration.whisker.apex_fraction</td><td>0.9</td><td>toml</td>"));
    assert!(
        html.contains("<b>varies</b>: 60 ×1, 90 ×1")
            || html.contains("<b>varies</b>: 90 ×1, 60 ×1")
    );
    assert!(html.contains("<td>ym_y__proximal45</td><td>90</td>"));
    // Old-run fallback and the audit section.
    assert!(html.contains("Effective parameters not recorded for this run"));
    assert!(html.contains("scoring_provenance.comparison_policy"));
    assert!(html.contains("Source files (audit)") && html.contains("[review]"));
    // The first trial by ID supplies the figure; it is copied, linked and sealed.
    let copied = walk(&out.join("figures/parameters"));
    assert_eq!(copied.len(), 2, "{copied:?}");
    let arm = copied
        .iter()
        .find(|p| p.to_string_lossy().contains("story_arm_sequence"))
        .unwrap();
    assert_eq!(fs::read_to_string(arm).unwrap(), "png ym_x__proximal45");
    assert!(html.contains("from trial ym_x__proximal45"));
    assert!(html.contains("Preview figures not available"));
    assert_links_resolve(out);
    let sums = fs::read_to_string(out.join("SHA256SUMS")).unwrap();
    assert!(sums.contains("figures/parameters/"));
    assert_eq!(
        seal::verify_dir(out).unwrap(),
        summary["files"].as_u64().unwrap() as usize
    );
}

#[test]
fn decision_log_lists_dated_headings_of_review_requirements() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path());
    assert!(
        parameters::decision_log(&args).unwrap().is_none(),
        "no requirements configured"
    );
    let config = args.config.clone().unwrap();
    let text = fs::read_to_string(&config).unwrap();
    fs::write(
        &config,
        format!(
            "[review]\nrequirements = \"req.md\"\n{}",
            text.replacen("[review]\n", "", 1)
        ),
    )
    .unwrap();
    let error = parameters::decision_log(&args).unwrap_err();
    assert!(
        format!("{error:#}").contains("req.md"),
        "configured but missing: {error:#}"
    );
    fs::write(
        root.path().join("req.md"),
        "# Title\n## 2026-10-28 NORT <bout> limits\nbody\n## Undated\n",
    )
    .unwrap();
    let log = parameters::decision_log(&args).unwrap().unwrap();
    assert!(
        log.contains("<li>2026-10-28 NORT &lt;bout&gt; limits</li>"),
        "{log}"
    );
    assert!(!log.contains("Undated") && !log.contains("Title"));
}

#[test]
fn animal_pages_follow_the_nort_protocol_and_carry_a_summary() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path());
    let data = root.path().join("run/data");
    // Animal 7: pre and post OFT (habituation), T1 and T2, plus the Y-maze recording.
    for (id, task, video) in [
        ("oft_pre", "open_field", "/v/oft0.mp4"),
        ("nort_pre_t2", "object_recognition", "/v/n_pre_t2.mp4"),
        ("nort_post_t1", "object_recognition", "/v/n_post_t1.mp4"),
        ("nort_pre_t1", "object_recognition", "/v/n_pre_t1.mp4"),
        ("nort_post_t2", "object_recognition", "/v/n_post_t2.mp4"),
    ] {
        trial(&data, id, task, video);
    }
    let metrics = data.join("_post_analysis/cohort-1/primary45/_internal/metrics.json");
    let mut rows: Vec<Value> = serde_json::from_slice(&fs::read(&metrics).unwrap()).unwrap();
    rows[0]["timepoint"] = json!("post");
    rows[0]["phase"] = json!("A");
    rows[1]["timepoint"] = json!("post");
    rows[1]["variant"] = json!("proximal45");
    for (id, assay, timepoint, phase, value) in [
        ("oft_pre", "oft", "pre", "A", 20.0),
        ("nort_pre_t1", "nort", "pre", "training", 0.1),
        ("nort_pre_t2", "nort", "pre", "test", 0.3),
        ("nort_post_t1", "nort", "post", "training", -0.2),
        ("nort_post_t2", "nort", "post", "test", 0.5),
    ] {
        rows.push(
            json!({"analysis_id": id, "animal_id": "7", "group": "2", "assay": assay,
            "timepoint": timepoint, "phase": phase, "center_time_pct": value, "DI": value,
            "object_A_label": "familiar", "object_A_placement": "U"}),
        );
    }
    // A second animal of the same group gives the group distribution.
    trial(&data, "oft_pre_8", "open_field", "/v/oft8.mp4");
    rows.push(
        json!({"analysis_id": "oft_pre_8", "animal_id": "8", "group": "2", "assay": "oft",
        "timepoint": "pre", "phase": "A", "center_time_pct": 40.0}),
    );
    fs::write(&metrics, json!(rows).to_string()).unwrap();
    let config = args.config.clone().unwrap();
    fs::write(
        &config,
        "[review]\nworkers = 2\n[review.primary_endpoints]\noft = [\"center_time_pct\"]\nnort = [\"DI\"]\n\
         ymaze = [\"spontaneous_alternation_pct\"]\n",
    )
    .unwrap();
    package(&args).unwrap();
    let out = &args.output;
    assert_links_resolve(out);

    // Protocol order on the animal page: pre OFT -> T1 -> T2, then post, then Y-maze.
    let page = fs::read_to_string(out.join("animals/7.html")).unwrap();
    let at = |needle: &str| {
        page.find(needle)
            .unwrap_or_else(|| panic!("missing {needle}"))
    };
    assert!(at("Pre-treatment: NORT protocol") < at("Post-treatment: NORT protocol"));
    assert!(at("Post-treatment: NORT protocol") < at("<h2>Y-maze</h2>"));
    let pre = &page[at("Pre-treatment: NORT protocol")..at("Post-treatment: NORT protocol")];
    let (oft, t1, t2) = (
        pre.find("id=\"oft_pre\"").unwrap(),
        pre.find("id=\"nort_pre_t1\"").unwrap(),
        pre.find("id=\"nort_pre_t2\"").unwrap(),
    );
    assert!(
        oft < t1 && t1 < t2,
        "pre-treatment order is habituation, T1, T2"
    );
    assert!(page.contains("Pre · Habituation (OFT) · batch A"));
    assert!(page.contains("A = familiar (U)"));
    assert!(page.contains("Animal summary"));

    // Animals gallery and the experiment-analytics nav split.
    let gallery = fs::read_to_string(out.join("animals/index.html")).unwrap();
    assert!(gallery.contains("href=\"7.html\"") && gallery.contains("figures/animals/7.webp"));
    assert!(page.contains("Experiment analytics:"));

    // The summary job carries the animal's recorded values and its group's.
    let job: Value =
        serde_json::from_slice(&fs::read(out.join("figures/animals/7.job.json")).unwrap()).unwrap();
    let habituation = job["panels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["step"] == "Habituation (OFT)")
        .unwrap();
    assert_eq!(habituation["animal"]["pre"], json!(20.0));
    assert_eq!(habituation["group"]["pre"], json!([20.0, 40.0]));
    let test = job["panels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["step"] == "T2 test")
        .unwrap();
    assert_eq!(test["animal"], json!({"pre": 0.3, "post": 0.5}));
    assert!(seal::verify_dir(out).unwrap() > 0);
}

/// Stand-in for `bekiper inspect video --scored`: writes the video, poster and receipt.
const FAKE_VIDEO: &str = r#"#!/usr/bin/env python3
import json, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ["inspect", "video"] and "--scored" in args
out = Path(args[args.index("--output") + 1]); source = args[args.index("--source-video") + 1]
out.write_bytes(b"\x00\x00\x00\x18ftypmp42" + source.encode())
out.with_name(out.stem + ".poster.png").write_bytes(b"\x89PNG")
out.with_name(out.stem + ".video.json").write_text(json.dumps({"resolution": [232, 258], "frames_written": 10,
    "encoded_fps": 15.0, "source_video": source,
    "scored_rows": [{"heuristic": "in_center", "result_column": "in_center", "perimeter": "in_center"}]}))
with open(Path(__file__).parent / "videos.log", "a") as log: log.write(source + "\n")
"#;

#[test]
fn videos_are_rebased_rendered_embedded_and_resumed() {
    let root = tempfile::tempdir().unwrap();
    let mut args = fixture(root.path());
    // Source videos moved: only the OFT one exists under the new root.
    let moved = root.path().join("moved");
    fs::create_dir_all(&moved).unwrap();
    fs::write(moved.join("oft1.mp4"), "video").unwrap();
    let renderer = root.path().join("fake-video");
    fs::write(&renderer, FAKE_VIDEO).unwrap();
    fs::set_permissions(&renderer, fs::Permissions::from_mode(0o755)).unwrap();
    args.videos = true;
    args.video_source = Some(format!("/v={}", moved.display()));
    args.video_renderer = Some(renderer);
    let summary = package(&args).unwrap();
    assert_eq!(summary["videos_rendered_now"], 1);
    assert_eq!(summary["video_failures"], 1, "the Y-maze source is missing");
    let out = &args.output;
    assert!(
        out.join("videos/7/oft_a.mp4").is_file() && out.join("videos/7/oft_a.poster.png").is_file()
    );
    assert!(
        !out.join("logs/videos/videos__7__oft_a").exists(),
        "staging is cleaned up"
    );
    let page = fs::read_to_string(out.join("animals/7.html")).unwrap();
    assert!(page.contains("<video controls preload=\"none\" poster=\"../videos/7/oft_a.poster.png\" width=\"232\" height=\"258\">"));
    assert!(page.contains("in_center") && page.contains("Video not rendered:"));
    assert!(
        fs::read_to_string(out.join("issues.html"))
            .unwrap()
            .contains("Video render failures")
    );
    assert_links_resolve(out);
    assert_eq!(
        seal::verify_dir(out).unwrap(),
        summary["files"].as_u64().unwrap() as usize
    );
    // Resume renders nothing and prunes videos of other animals and recordings.
    fs::create_dir_all(out.join("videos/99")).unwrap();
    fs::write(out.join("videos/99/old.mp4"), "x").unwrap();
    fs::write(out.join("videos/7/gone.mp4"), "x").unwrap();
    fs::create_dir_all(out.join("figures/animals")).unwrap();
    fs::write(out.join("figures/animals/99.webp"), "x").unwrap();
    assert_eq!(package(&args).unwrap()["videos_rendered_now"], 0);
    assert!(!out.join("videos/99").exists() && !out.join("videos/7/gone.mp4").exists());
    assert!(!out.join("figures/animals/99.webp").exists());
    assert!(
        out.join("videos/7/oft_a.mp4").is_file() && out.join("figures/animals/7.webp").is_file()
    );
    args.theme = Some("light".into());
    assert_eq!(package(&args).unwrap()["videos_rendered_now"], 1);
    let log = fs::read_to_string(root.path().join("videos.log")).unwrap();
    assert_eq!(log.lines().count(), 2);
    assert!(log.lines().all(|l| l.ends_with("moved/oft1.mp4")));
}

/// Presentation of Y-maze trials `<id>__<variant>` of one recording.
fn ymaze_presentation(variants: &[&str]) -> bekiper_core::experiment::ResolvedPresentation {
    let trials = variants
        .iter()
        .map(|v| bekiper_core::experiment::TrialFacts {
            id: format!("ym__{v}"),
            task_type: "y_maze".into(),
            variant: Some((*v).into()),
            ..Default::default()
        })
        .collect();
    bekiper_core::experiment::resolve(
        &bekiper_core::experiment::CorpusFacts { trials },
        &Default::default(),
    )
    .unwrap()
}

#[test]
fn rv_r1_current_imen_ymaze_variants_select_principal_row() {
    // Imen contexts are arm0_dwell1s/arm25/arm45/arm65 (examples/neo/imen/bekiper_config.toml).
    let presentation = ymaze_presentation(&["arm0_dwell1s", "arm25", "arm45", "arm65"]);
    let row = |v: &str| json!({"analysis_id": format!("ym__{v}")});
    assert_eq!(index::row_step(&presentation, &row("arm45")), Some("ymaze"));
    assert_eq!(index::row_step(&presentation, &row("arm25")), None);
}

#[test]
fn rv_r1_variant_labels_and_order_follow_current_imen_contexts() {
    let presentation =
        ymaze_presentation(&["arm0_dwell1s", "arm25", "arm45", "arm65", "proximal45"]);
    let t = |v: &str| index::Trial {
        id: format!("ym__{v}"),
        dir: PathBuf::new(),
        assay: "ymaze".into(),
        task_type: "y_maze".into(),
        recipe: None,
        video: "v".into(),
        variant: Some(v.into()),
        row: None,
        session: json!({}),
        kinds: Default::default(),
        perimeters: Vec::new(),
        placement: presentation.trials.get(&format!("ym__{v}")).cloned(),
    };
    assert_eq!(t("arm65").variant_label(), "65%");
    assert_eq!(t("arm0_dwell1s").variant_label(), "0% (1 s dwell)");
    assert_eq!(t("proximal45").variant_label(), "45%");
    let principal = |v: &str| t(v).placement.unwrap().principal;
    assert!(principal("arm45") && !principal("arm25"));
}

#[test]
fn rv_r2_r5_r9_escape_attributes_and_numbers_and_start_ordered_lists() {
    assert!(html::img("a\"b.png", "x", None).contains("src=\"a&quot;b.png\""));
    assert!(html::markdown("3. three\n4. four").contains("<ol start=\"3\">"));
}

#[test]
fn rv_r6_slug_collisions_are_rejected() {
    assert!(index::ensure_unique_slugs("animal", ["1-2", "1_2"]).is_err());
    assert!(index::ensure_unique_slugs("animal", ["7", "7", "8"]).is_ok());
}

#[test]
fn design_contract_matches_blocks() {
    let source = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../design/review-package/contract.toml"
    ))
    .unwrap();
    let contract = bekiper_core::experiment::Contract::parse(&source).unwrap();
    let ids: Vec<&str> = contract.blocks.iter().map(|b| b.id.as_str()).collect();
    assert_eq!(html::BLOCKS, ids.as_slice());
    for descriptor in bekiper_core::experiment::descriptors() {
        for rule in &descriptor.blocks {
            assert!(
                html::BLOCKS.contains(&rule.id.as_str()),
                "{}: block {} is not rendered",
                descriptor.id,
                rule.id
            );
        }
    }
    contract
        .check_descriptors(bekiper_core::experiment::descriptors())
        .unwrap();
    // The design system exported from review-package.fig must carry the same
    // component sets, axes, variants, property kinds and tokens. The map is TOML
    // so the Nix test builds ship it; a missing map is a failure.
    let map = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../design/review-package/component-map.toml"
    ))
    .expect("design/review-package/component-map.toml");
    check_component_map(&contract, &map).unwrap();
}

/// Compares an exported component map (TOML text) with the contract.
fn check_component_map(
    contract: &bekiper_core::experiment::Contract,
    map: &str,
) -> Result<(), String> {
    use bekiper_core::experiment::PropertyKind;
    use std::collections::{BTreeMap, BTreeSet};
    let strs = |v: Option<&toml::Value>| -> BTreeSet<String> {
        v.and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|s| s.as_str().map(String::from))
            .collect()
    };
    let owned = |v: &[String]| v.iter().cloned().collect::<BTreeSet<_>>();
    let map: toml::Value = toml::from_str(map).map_err(|e| e.to_string())?;
    let designed: BTreeMap<&str, &toml::Value> = map["blocks"]
        .as_array()
        .ok_or("blocks")?
        .iter()
        .map(|b| (b["name"].as_str().unwrap_or_default(), b))
        .collect();
    let names: BTreeSet<&str> = contract.blocks.iter().map(|b| b.name.as_str()).collect();
    if designed.keys().copied().collect::<BTreeSet<_>>() != names {
        return Err(format!(
            "component sets {:?} vs contract {names:?}",
            designed.keys()
        ));
    }
    for block in &contract.blocks {
        let design = designed[block.name.as_str()];
        let axes = design["axes"].as_table().ok_or("axes")?;
        if axes.keys().cloned().collect::<BTreeSet<_>>() != owned(&block.axes) {
            return Err(format!(
                "{} axes {:?}",
                block.name,
                axes.keys().collect::<Vec<_>>()
            ));
        }
        let mut product = vec![String::new()];
        for axis in &block.axes {
            let values = if axis == "archetype" && !block.archetypes.is_empty() {
                &block.archetypes
            } else {
                &contract.axes[axis]
            };
            if strs(axes.get(axis)) != owned(values) {
                return Err(format!(
                    "{} axis {axis} values {:?}",
                    block.name, axes[axis]
                ));
            }
            product = product
                .iter()
                .flat_map(|head| {
                    let sep = if head.is_empty() { "" } else { ", " };
                    values.iter().map(move |v| format!("{head}{sep}{axis}={v}"))
                })
                .collect();
        }
        let variants = strs(design.get("variants"));
        if variants != product.into_iter().collect() {
            return Err(format!("{} variants {variants:?}", block.name));
        }
        // An absent properties table means no properties, as in the normaliser.
        let props: BTreeMap<String, String> = design
            .get("properties")
            .and_then(toml::Value::as_table)
            .into_iter()
            .flatten()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
            .collect();
        let want: BTreeMap<String, String> = block
            .properties
            .iter()
            .map(|(k, v)| {
                let kind = match v {
                    PropertyKind::Text => "TEXT",
                    PropertyKind::Visible => "VISIBLE",
                    PropertyKind::InstanceSwap => "INSTANCE_SWAP",
                };
                (k.clone(), kind.to_string())
            })
            .collect();
        if props != want {
            return Err(format!("{} properties {props:?} vs {want:?}", block.name));
        }
    }
    let tokens = &contract.tokens;
    let designed = map.get("tokens").ok_or("tokens missing")?;
    if designed["collection"].as_str() != Some(tokens.collection.as_str())
        || strs(designed.get("modes")) != owned(&tokens.modes)
        || strs(designed.get("colors")) != owned(&tokens.colors)
    {
        return Err(format!("tokens {designed:?} vs contract {tokens:?}"));
    }
    for palette in [&themekit::DARK, &themekit::LIGHT] {
        let have: BTreeSet<&str> = palette.colors().iter().map(|(n, _)| *n).collect();
        if let Some(name) = tokens.colors.iter().find(|c| !have.contains(c.as_str())) {
            return Err(format!("token {name} is not a themekit colour"));
        }
    }
    Ok(())
}

#[test]
fn design_component_map_check_is_strict() {
    let contract = bekiper_core::experiment::Contract::parse(
        &fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../design/review-package/contract.toml"
        ))
        .unwrap(),
    )
    .unwrap();
    let map = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../design/review-package/component-map.toml"
    ))
    .unwrap();
    check_component_map(&contract, &map).unwrap();
    let mutations = [
        ("drop a variant", "  \"state=ok, viewport=desktop\",\n", ""),
        ("drop an axis value", "  \"mobile\",\n", ""),
        ("change a property kind", "\"TEXT\"", "\"VISIBLE\""),
        ("rename a token", "\"surface_alt\"", "\"surface_altx\""),
        ("drop a token", "  \"link\",\n", ""),
    ];
    for (what, from, to) in mutations {
        assert!(map.contains(from), "{what}: pattern absent");
        assert!(
            check_component_map(&contract, &map.replacen(from, to, 1)).is_err(),
            "{what} passed"
        );
    }
}

/// A package of synthetic trials `(id, task type, recorded facts)`, one video each,
/// without a metrics row; builds it and returns its directory.
fn build(root: &Path, trials: &[(&str, &str, Recorded<'_>)]) -> PathBuf {
    let args = fixture(root);
    // Only the given trials: drop the fixture's.
    let data = root.join("run/data");
    fs::remove_dir_all(&data).unwrap();
    for (id, task, recorded) in trials {
        trial_with(&data, id, task, &format!("/v/{id}.mp4"), recorded);
    }
    fs::write(args.config.as_ref().unwrap(), "[review]\nworkers = 2\n").unwrap();
    let summary = package(&args).unwrap();
    let out = args.output;
    assert_links_resolve(&out);
    assert_eq!(
        seal::verify_dir(&out).unwrap(),
        summary["files"].as_u64().unwrap() as usize
    );
    out
}

fn page(out: &Path, name: &str) -> String {
    fs::read_to_string(out.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn presentation_json(out: &Path) -> Value {
    serde_json::from_slice(&fs::read(out.join("provenance/presentation.json")).unwrap()).unwrap()
}

/// No page or nav falls back to an "Other" experiment.
fn assert_no_other(out: &Path) {
    assert!(!out.join("assays/other.html").exists());
    for name in ["index.html", "animals/index.html"] {
        let html = page(out, name);
        assert!(!html.contains(">Other<"), "{name}");
    }
}

#[test]
fn oul_corpus_orders_training_update_test_and_shows_the_object_legend() {
    let root = tempfile::tempdir().unwrap();
    let session = json!({"animal": "7"});
    let recorded = |recipe| Recorded {
        perimeters: &["object_1", "object_2", "object_3", "object_4"],
        recipe: Some(recipe),
        session: session.clone(),
        ..Default::default()
    };
    let out = build(
        root.path(),
        &[
            ("o_test", "object_recognition", recorded("oul_test")),
            ("o_update", "object_recognition", recorded("oul_update")),
            ("o_train1", "object_recognition", recorded("oul_training")),
            ("o_train2", "object_recognition", recorded("oul_training")),
            ("o_train3", "object_recognition", recorded("oul_training")),
        ],
    );
    let animal = page(&out, "animals/7.html");
    let at = |needle: &str| {
        animal
            .find(needle)
            .unwrap_or_else(|| panic!("missing {needle}"))
    };
    assert!(animal.contains("OUL protocol (timepoint unrecorded)"));
    assert!(!animal.contains("NORT protocol"));
    assert!(at("<b>Training</b>") < at("<b>Update</b>") && at("<b>Update</b>") < at("<b>Test</b>"));
    assert!(
        at("id=\"o_train3\"") < at("id=\"o_update\"")
            && at("id=\"o_update\"") < at("id=\"o_test\"")
    );
    // The protocol sentence follows the sections and steps.
    assert!(
        page(&out, "animals/index.html")
            .contains("pre-treatment training → update → test, the same post-treatment —")
    );
    assert!(animal.contains("blk-object_role_legend"));
    let presentation = presentation_json(&out);
    let nort = &presentation["experiments"][0];
    assert_eq!(nort["key"], "nort");
    assert!(
        nort["blocks"]
            .as_array()
            .unwrap()
            .contains(&json!("object_role_legend"))
    );
    assert_no_other(&out);
}

#[test]
fn oul_phase_part_splits_pre_and_post_sections() {
    let root = tempfile::tempdir().unwrap();
    let recorded = |recipe, part| Recorded {
        perimeters: &["object_1"],
        recipe: Some(recipe),
        session: json!({"animal": "7", "phase_part": part}),
        ..Default::default()
    };
    let out = build(
        root.path(),
        &[
            (
                "o_post",
                "object_recognition",
                recorded("oul_training", "2"),
            ),
            ("o_pre", "object_recognition", recorded("oul_training", "1")),
        ],
    );
    let animal = page(&out, "animals/7.html");
    assert!(
        animal.find("Pre-treatment: OUL protocol").unwrap()
            < animal.find("Post-treatment: OUL protocol").unwrap()
    );
}

#[test]
fn cheeseboard_corpus_gets_its_own_page_and_nav_entry() {
    let root = tempfile::tempdir().unwrap();
    let out = build(
        root.path(),
        &[(
            "cb_1",
            "cheeseboard",
            Recorded {
                perimeters: &["reward", "start"],
                session: json!({"animal": "7"}),
                ..Default::default()
            },
        )],
    );
    assert!(out.join("assays/cheeseboard.html").is_file());
    assert!(page(&out, "index.html").contains(">Cheeseboard</a>"));
    let presentation = presentation_json(&out);
    assert_eq!(presentation["experiments"][0]["archetype"], "goal_search");
    assert!(
        presentation["experiments"][0]["blocks"]
            .as_array()
            .unwrap()
            .contains(&json!("object_role_legend"))
    );
    assert_no_other(&out);
}

#[test]
fn open_field_alone_is_labelled_open_field() {
    let root = tempfile::tempdir().unwrap();
    let out = build(
        root.path(),
        &[(
            "oft_1",
            "open_field",
            Recorded {
                session: json!({"animal": "7"}),
                ..Default::default()
            },
        )],
    );
    let animal = page(&out, "animals/7.html");
    assert!(animal.contains("<b>Open field</b>") && !animal.contains("Habituation (OFT)"));
    // The protocol sentence names no NORT/Y-maze step.
    let gallery = page(&out, "animals/index.html");
    assert!(gallery.contains("pre-treatment open field, the same post-treatment —"));
    assert!(!gallery.contains("Y-maze"));
    assert_no_other(&out);
}

#[test]
fn ymaze_alone_picks_the_45_percent_context_as_principal() {
    let root = tempfile::tempdir().unwrap();
    let session = json!({"animal": "7"});
    let variants = ["arm25", "arm45", "arm65"];
    let trials: Vec<(String, Recorded<'_>)> = variants
        .iter()
        .map(|v| {
            (
                format!("ym__{v}"),
                Recorded {
                    session: session.clone(),
                    ..Default::default()
                },
            )
        })
        .collect();
    let given: Vec<(&str, &str, Recorded<'_>)> = trials
        .into_iter()
        .map(|(id, r)| (&*Box::leak(id.into_boxed_str()), "y_maze", r))
        .collect();
    let out = build(root.path(), &given);
    let presentation = presentation_json(&out);
    let contexts = presentation["experiments"][0]["contexts"]
        .as_array()
        .unwrap();
    let principal: Vec<&str> = contexts
        .iter()
        .filter(|c| c["principal"] == true)
        .map(|c| c["variant"].as_str().unwrap())
        .collect();
    assert_eq!(principal, ["arm45"]);
    // The contexts note and the cohort sentence come from the Y-maze prose; OFT/NORT are absent.
    assert!(page(&out, "assays/ymaze.html").contains("blk-threshold_comparison"));
    assert!(!page(&out, "guide.html").contains("do not depend on it"));
    assert_no_other(&out);
}

#[test]
fn unknown_task_type_keeps_its_name_and_the_generic_archetype() {
    let root = tempfile::tempdir().unwrap();
    let out = build(
        root.path(),
        &[(
            "rr_1",
            "rotarod",
            Recorded {
                session: json!({"animal": "7"}),
                ..Default::default()
            },
        )],
    );
    let assay = page(&out, "assays/rotarod.html");
    assert!(assay.contains("<title>rotarod</title>"));
    assert!(page(&out, "index.html").contains(">rotarod</a>"));
    let presentation = presentation_json(&out);
    assert_eq!(presentation["experiments"][0]["key"], "rotarod");
    assert_eq!(presentation["experiments"][0]["archetype"], "generic");
    let animal = page(&out, "animals/7.html");
    assert!(animal.contains("<h2>rotarod</h2>") && !animal.contains("Other recordings"));
    assert!(page(&out, "animals/index.html").contains("then the rotarod"));
    assert_no_other(&out);
}

#[test]
fn imen_prose_is_generated_from_the_descriptors() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path());
    let data = root.path().join("run/data");
    for (id, recipe) in [("nort_1", "nort_training"), ("nort_2", "nort_novelty")] {
        let recorded = Recorded {
            recipe: Some(recipe),
            ..Default::default()
        };
        trial_with(
            &data,
            id,
            "object_recognition",
            &format!("/v/{id}.mp4"),
            &recorded,
        );
    }
    package(&args).unwrap();
    let out = &args.output;
    assert!(page(out, "assays/oft.html").contains(
        "OFT and NORT do not depend on the Y-maze arm-entry threshold; shown from the arm45 context."
    ));
    assert!(
        page(out, "assays/ymaze.html")
            .contains("One figure per arm-entry threshold context (25/45/65 %).")
    );
    assert!(page(out, "animals/index.html").contains(
        "every session in protocol order — pre-treatment habituation (OFT) → T1 training → T2 test, \
the same post-treatment, then the Y-maze — each with"
    ));
    let guide = page(out, "README.md");
    assert!(guide.contains(
        "constant QC endpoints. Y-maze shows each arm-entry threshold context (25/45/65 %); OFT and NORT do not depend on it and are shown once.\n"
    ));
    assert!(guide.contains("\nY-maze composites carry all four scoring variants (0 % comparison, 25/45/65 %) of the same recording; rows that are identical across variants are drawn once. Every manifest feature"));
    assert!(page(out, "index.html").contains("class=\"card blk-context_card blk-context_card--archetype-zone_occupancy blk-context_card--state-ok\""));
    assert!(page(out, "index.html").contains("<header class=\"blk-header\">"));
}

#[test]
fn blk_lists_the_block_and_its_variants() {
    assert_eq!(
        html::blk("callout", &[("state", "warn")]),
        "blk-callout blk-callout--state-warn"
    );
}

#[test]
fn review_overrides_reach_the_generic_set_layer() {
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("config.toml");
    fs::write(&config, "[review]\nworkers = 2\n").unwrap();
    let env = |k: &str| (k == "BEKIPER_SET").then(|| "review.workers=3".to_string());
    let (review, _) = load_review_config_from(
        Some(&config),
        None,
        &[
            "review.presentation.nort.title=\"NOR\"".into(),
            "review.title=\"Project X\"".into(),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(review.workers, Some(3));
    assert_eq!(review.title.as_deref(), Some("Project X"));
    assert_eq!(
        review.presentation.experiments["nort"].title.as_deref(),
        Some("NOR")
    );
    for unknown in [
        "review.bogus=1",
        "review.presentation.nort.colour=1",
        "nort.exploration.x=1",
    ] {
        assert!(
            load_review_config_from(Some(&config), None, &[unknown.into()], &|_| None).is_err(),
            "{unknown}"
        );
    }
}

#[test]
fn run_scoped_env_overrides_are_ignored_by_review() {
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("config.toml");
    fs::write(&config, "[review]\nworkers = 2\n").unwrap();
    let env = |k: &str| {
        (k == "BEKIPER_SET")
            .then(|| "nort.exploration.whisker.length_m=0.02;review.workers=3".to_string())
    };
    let (review, _) = load_review_config_from(Some(&config), None, &[], &env).unwrap();
    assert_eq!(review.workers, Some(3));
    // An explicit --set with a run key still fails.
    assert!(
        load_review_config_from(
            Some(&config),
            None,
            &["nort.exploration.whisker.length_m=0.02".into()],
            &env
        )
        .is_err()
    );
}

#[test]
fn overrides_review_and_seal_cover_provenance_and_endpoint_resume() {
    let root = tempfile::tempdir().unwrap();
    let mut args = fixture(root.path());
    args.set = vec!["review.title=\"Project X\"".into()];
    package(&args).unwrap();
    let out = &args.output;
    let provenance: Value =
        serde_json::from_slice(&fs::read(out.join("provenance/provenance.json")).unwrap()).unwrap();
    assert_eq!(
        provenance["overrides"]["cli"],
        json!(["review.title=\"Project X\""])
    );
    assert_eq!(provenance["review"]["title"], "Project X");

    // Tampering the sealed presentation record fails verification.
    let presentation = out.join("provenance/presentation.json");
    let original = fs::read(&presentation).unwrap();
    fs::write(&presentation, b"{}").unwrap();
    assert!(seal::verify_dir(out).is_err());
    fs::write(&presentation, original).unwrap();
    assert!(seal::verify_dir(out).is_ok());

    // Changing the primary endpoints redraws the cohort figures on resume.
    let cohort = |log: &str| log.lines().filter(|l| l.starts_with("cohort ")).count();
    let log = root.path().join("renders.log");
    assert_eq!(cohort(&fs::read_to_string(&log).unwrap()), 1);
    package(&args).unwrap();
    assert_eq!(cohort(&fs::read_to_string(&log).unwrap()), 1);
    fs::write(
        args.config.as_ref().unwrap(),
        "[review]\nworkers = 2\n[review.primary_endpoints]\noft = [\"analysis_duration_s\"]\n",
    )
    .unwrap();
    package(&args).unwrap();
    assert_eq!(cohort(&fs::read_to_string(&log).unwrap()), 2);
}

/// Like [`build`], with the given `config.toml` (without the `[review] workers`).
fn build_configured(root: &Path, config: &str, trials: &[(&str, &str, Recorded<'_>)]) -> PathBuf {
    let args = fixture(root);
    let data = root.join("run/data");
    fs::remove_dir_all(&data).unwrap();
    for (id, task, recorded) in trials {
        trial_with(&data, id, task, &format!("/v/{id}.mp4"), recorded);
    }
    fs::write(
        args.config.as_ref().unwrap(),
        format!("[review]\nworkers = 2\n{config}"),
    )
    .unwrap();
    package(&args).unwrap();
    args.output
}

fn session7() -> Recorded<'static> {
    Recorded {
        session: json!({"animal": "7"}),
        ..Default::default()
    }
}

#[test]
fn rv_override_text_is_escaped_in_nav_cards_and_protocol_prose() {
    let root = tempfile::tempdir().unwrap();
    let out = build_configured(
        root.path(),
        "[review.presentation.nort]\ntitle = \"A <b>&</b>\"\nshort_name = \"N<i>\"\n",
        &[("n_1", "object_recognition", session7())],
    );
    for name in ["index.html", "assays/nort.html", "animals/index.html"] {
        let html = page(&out, name);
        assert!(
            !html.contains("<b>&</b>") && !html.contains("<i>"),
            "{name}"
        );
        assert!(html.contains("N&lt;i&gt;"), "{name}");
    }
    assert!(page(&out, "index.html").contains("A &lt;b&gt;&amp;&lt;/b&gt;"));
}

#[test]
fn rv_nav_marks_the_page_by_target_not_by_title() {
    let root = tempfile::tempdir().unwrap();
    let out = build_configured(
        root.path(),
        "[review.presentation.nort]\ntitle = \"Animals\"\n",
        &[("n_1", "object_recognition", session7())],
    );
    for (name, href) in [
        ("assays/nort.html", "assays/nort.html"),
        ("animals/index.html", "animals/index.html"),
    ] {
        let html = page(&out, name);
        assert_eq!(html.matches("aria-current").count(), 1, "{name}");
        let up = if name.contains('/') { "../" } else { "" };
        assert!(
            html.contains(&format!("<a href=\"{up}{href}\" aria-current=\"page\">")),
            "{name}"
        );
    }
}

#[test]
fn rv_unknown_task_type_cannot_escape_the_assays_directory() {
    let root = tempfile::tempdir().unwrap();
    let out = build(root.path(), &[("x_1", "../x", session7())]);
    assert!(!out.join("x.html").exists());
    assert!(out.join("assays/___x.html").is_file());
    assert!(page(&out, "index.html").contains("href=\"assays/___x.html\""));
}

#[test]
fn rv_cheeseboard_only_package_lists_no_exploration_or_triad_flags() {
    let root = tempfile::tempdir().unwrap();
    let out = build(root.path(), &[("cb_1", "cheeseboard", session7())]);
    let issues = page(&out, "issues.html");
    assert!(!issues.contains("15 s exploration") && !issues.contains("usable triads"));
    assert!(issues.contains("Flag: 360 ± 10 s"));
}
