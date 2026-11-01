//! Independent primary contexts over one retained native trial population.
//!
//! A context varies one assay's variant (`execution.contexts`, or the legacy
//! `ymaze_primary_thresholds` expanded to Y-maze contexts); every other assay
//! takes its principal variant. Figure pages that two contexts draw
//! identically are rendered once (`families`).
use super::*;
use rayon::prelude::*;

pub(super) struct Context {
    /// Stable identity (the variant); the directory only locates the output.
    id: String,
    assay: String,
    variant: String,
    threshold_pct: Option<u8>,
    dir: String,
    /// Legacy threshold contexts select Y-maze rows by threshold alone.
    explicit: bool,
}

pub(super) fn requested(config: &Record) -> bool {
    ["ymaze_primary_thresholds", "contexts"].iter().any(|key| {
        config["execution"][*key]
            .as_array()
            .is_some_and(|v| !v.is_empty())
    })
}

/// Variant name of a legacy threshold context (0% is the dwell-gated primary).
fn legacy_variant(threshold: u8) -> String {
    if threshold == 0 {
        "arm0_dwell1s".into()
    } else {
        format!("arm{threshold}")
    }
}

pub(super) fn contexts(config: &Record) -> Result<Vec<Context>> {
    let execution = &config["execution"];
    let specs: Vec<crate::contracts::ContextSpec> = match execution["contexts"].as_array() {
        Some(list) if !list.is_empty() => serde_json::from_value(json!(list))?,
        _ => Vec::new(),
    };
    let contexts: Vec<Context> = if specs.is_empty() {
        let thresholds: Vec<u8> =
            serde_json::from_value(execution["ymaze_primary_thresholds"].clone())?;
        thresholds
            .into_iter()
            .map(|t| Context {
                id: legacy_variant(t),
                assay: "ymaze".into(),
                variant: legacy_variant(t),
                threshold_pct: Some(t),
                dir: format!("primary{t}"),
                explicit: false,
            })
            .collect()
    } else {
        specs
            .into_iter()
            .map(|spec| Context {
                threshold_pct: (spec.assay == "ymaze")
                    .then(|| crate::inputs::context_threshold_pct(&spec.variant))
                    .flatten(),
                id: spec.variant.clone(),
                dir: spec.variant.clone(),
                assay: spec.assay,
                variant: spec.variant,
                explicit: true,
            })
            .collect()
    };
    let mut seen = BTreeSet::new();
    ensure!(
        contexts.iter().all(|c| seen.insert(c.dir.clone())),
        "duplicate primary context"
    );
    Ok(contexts)
}

fn is_comparison(row: &Record) -> bool {
    row["variant"]
        .as_str()
        .is_some_and(|v| v.starts_with("compare_"))
        || row["analysis_role"]
            .as_str()
            .is_some_and(|v| v.starts_with("ymaze__compare_") || v.starts_with("y_maze__compare_"))
}

/// Principal variant of every assay that has several: the descriptor's first
/// listed principal present, else the bare analysis, else the first by order.
/// Assays with one variant are not filtered.
fn principals(rows: &[Record]) -> BTreeMap<String, Option<String>> {
    let mut variants: BTreeMap<&str, BTreeSet<Option<&str>>> = BTreeMap::new();
    for row in rows.iter().filter(|r| !is_comparison(r)) {
        if let Some(assay) = row["assay"].as_str() {
            variants
                .entry(assay)
                .or_default()
                .insert(row["variant"].as_str());
        }
    }
    variants
        .into_iter()
        .filter(|(_, set)| set.len() > 1)
        .map(|(assay, set)| {
            let contexts = bekiper_core::experiment::descriptor(assay)
                .and_then(|d| d.contexts.clone())
                .unwrap_or_default();
            let chosen = contexts
                .principal
                .iter()
                .find(|p| set.contains(&Some(p.as_str())))
                .cloned()
                .or_else(|| {
                    if set.contains(&None) {
                        None
                    } else {
                        set.iter()
                            .flatten()
                            .min_by_key(|v| contexts.order_key(Some(v)))
                            .map(|v| (*v).to_owned())
                    }
                });
            (assay.to_owned(), chosen)
        })
        .collect()
}

/// Rows one context scores. Y-maze rows are left to the D03 primary threshold
/// when `all_ymaze` (registered-plan path); descriptive populations are
/// subset here.
fn select(
    rows: &[Record],
    context: &Context,
    principals: &BTreeMap<String, Option<String>>,
    all_ymaze: bool,
) -> Vec<Record> {
    let threshold = |row: &Record| {
        context.threshold_pct.is_some_and(|t| {
            row["threshold_pct"].as_u64() == Some(u64::from(t))
                || row["threshold_pct"].as_f64() == Some(f64::from(t))
        })
    };
    rows.iter()
        .filter(|row| {
            let assay = row["assay"].as_str().unwrap_or_default();
            if assay == "ymaze" && (all_ymaze || is_comparison(row)) {
                return true;
            }
            if assay == context.assay {
                return if assay == "ymaze" {
                    threshold(row)
                        && (!context.explicit
                            || row["variant"].is_null()
                            || row["variant"] == context.variant.as_str())
                } else {
                    row["variant"] == context.variant.as_str()
                };
            }
            principals
                .get(assay)
                .is_none_or(|principal| row["variant"].as_str() == principal.as_deref())
        })
        .cloned()
        .collect()
}

pub(super) fn run(
    directory: &Path,
    config: &Record,
    rows: &[Record],
    fingerprints: Vec<Record>,
    config_path: &Path,
    omissions: Option<Vec<Record>>,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Record> {
    let contexts = contexts(config)?;
    let ids: Vec<&str> = contexts.iter().map(|c| c.id.as_str()).collect();
    let workers = config["execution"]["primary_workers"]
        .as_u64()
        .unwrap_or(crate::contracts::DEFAULT_PRIMARY_ANALYSIS_WORKERS as u64);
    let workers = usize::try_from(workers)?
        .min(contexts.len())
        .min(std::thread::available_parallelism()?.get());
    let principals = principals(rows);
    let store = directory.join("families");
    // Render processes of the whole publication are split across the contexts
    // that run at once (the warm-up context runs alone and keeps them all).
    // Unset: the renderer default applies to a lone context; concurrent
    // contexts share the renderer's default total.
    let total = config["execution"]["render_workers"].as_u64();
    let share = |workers: usize| -> Result<Option<u64>> {
        Ok(match total {
            Some(total) => Some((total / workers.max(1) as u64).max(1)),
            None if workers > 1 => {
                Some((std::thread::available_parallelism()?.get().min(8) / workers).max(1) as u64)
            }
            None => None,
        })
    };
    let run = |context: &Context, render: Option<u64>| -> Result<Record> {
        check_cancel(cancelled)?;
        let mut plan = config.clone();
        plan["execution"]["ymaze_primary_thresholds"] = json!([]);
        plan["execution"]["contexts"] = json!([]);
        plan["execution"]["render_workers"] = json!(render);
        plan[families::KEY_STORE] = json!(store);
        if !plan["decision_resolutions"].is_object() {
            plan["decision_resolutions"] = json!({});
        }
        let prior = plan["decision_resolutions"]["D03"].clone();
        if let Some(threshold) = context.threshold_pct {
            plan["assays"]["ymaze"]["primary_threshold_arm_length_pct"] = json!(threshold);
            plan["decision_resolutions"]["D03"] = json!({
                "origin":"existing_analysis_plan", "rationale":"Each explicitly configured threshold is primary in its own independent context",
                "approval_or_plan_reference":"effective_config.execution.contexts",
                "scope":["assays.ymaze"], "value": prior.get("value").filter(|v| v.is_object()).cloned().unwrap_or_else(|| json!({})) });
            plan["decision_resolutions"]["D03"]["value"]["assays.ymaze.primary_threshold_arm_length_pct"] =
                json!(threshold);
        }
        plan["primary_context"] = json!({"id":context.id,"assay":context.assay,"variant":context.variant,
            "threshold_pct":context.threshold_pct,"selection":ids,
            "independent_inference":omissions.is_none(),"configured_D03":prior});
        let path = directory.join(&context.dir);
        plan["outputs"]["directory"] = json!(context.dir);
        let mut stage = "initializing";
        let child = if let Some(omissions) = &omissions {
            publish_descriptive_with_cancel(
                select(rows, context, &principals, false),
                omissions.clone(),
                fingerprints.clone(),
                plan,
                &path,
                cancelled,
            )?
        } else {
            std::fs::create_dir(&path)?;
            let subset = select(rows, context, &principals, true);
            execute(
                &mut plan,
                &subset,
                fingerprints.clone(),
                config_path,
                &path,
                &mut stage,
                cancelled,
            )?
        };
        Ok(
            json!({"id":context.id,"assay":context.assay,"variant":context.variant,
                "threshold_pct":context.threshold_pct,"path":context.dir,
                "status":child["status"],"run_mode":child["run_mode"]}),
        )
    };
    // The first context fills the family store alone; the rest, which draw
    // the shared pages from it, run on the worker pool. Every worker is
    // drained before an error, cancellation, or terminal publication.
    // ponytail: warm-up serialises one context; per-page claims would let the
    // first wave overlap if the shared families ever dominate.
    let (first, rest) = contexts.split_at(usize::from(workers > 1));
    let mut attempts: Vec<_> = first.iter().map(|c| run(c, total)).collect();
    let wave = first.first().map_or(workers, |context| {
        memory_admitted(workers, &directory.join(&context.dir), mem_available())
    });
    let pool = rayon::ThreadPoolBuilder::new().num_threads(wave).build()?;
    let share = share(wave)?;
    attempts.extend(pool.install(|| rest.par_iter().map(|c| run(c, share)).collect::<Vec<_>>()));
    let results: Vec<_> = attempts.into_iter().collect::<Result<_>>()?;
    let status = if results.iter().all(|v| v["status"] == "complete") {
        "complete"
    } else {
        "partial"
    };
    let mut public_config = config.clone();
    sanitize_config(&mut public_config, &fingerprints);
    let manifest = json!({"schema_version":SCHEMA_VERSION,"status":status,"run_mode":config["run_mode"],
        "publication_kind":"independent_primary_contexts","primary_contexts":results,
        "primary_workers":workers,"concurrent_contexts":wave,"shared_trial_records":rows.len(),"source_inputs_reused":true,
        "family_store":store.is_dir().then_some("families"),
        "inputs":fingerprints,"code":crate::code_identity(),
        "scientific_readiness":if config["run_mode"] == "synthetic" {"fixture_only"} else {"see_each_context_dependency_report"},
        "multiplicity_scope":"independent_registered_contexts","effective_config":public_config});
    write_json(&directory.join("run_manifest.json"), &manifest)?;
    write_artifact_manifest(directory)?;
    verify_artifacts(directory)?;
    Ok(manifest)
}

/// Parsed bytes held per concurrent context for each byte of the first
/// context's `plotting_bundle.json` (Neo/Imen: 2.7 GB bundle, ~11 GB RSS
/// per context before the pending bundle stopped re-reading it).
const BUNDLE_MEMORY_FACTOR: u64 = 4;

/// Contexts admitted at once after the warm-up: every concurrent context
/// holds its own parsed bundle, sized by the first context's bundle file.
/// Without a measurement (no bundle, no `MemAvailable`) `workers` stands.
fn memory_admitted(workers: usize, first: &Path, available: Option<u64>) -> usize {
    let bundle = std::fs::metadata(first.join("_internal/plotting_bundle.json"))
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    match available {
        Some(available) if bundle > 0 => {
            let fit = available / (BUNDLE_MEMORY_FACTOR * bundle);
            workers.min(usize::try_from(fit).unwrap_or(usize::MAX).max(1))
        }
        _ => workers,
    }
}

fn mem_available() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?
        .trim()
        .strip_suffix("kB")?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kib| kib * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_contexts_fit_the_measured_bundle_into_available_memory() {
        let first = tempfile::tempdir().unwrap();
        assert_eq!(memory_admitted(4, first.path(), Some(1 << 40)), 4);
        std::fs::create_dir(first.path().join("_internal")).unwrap();
        std::fs::write(
            first.path().join("_internal/plotting_bundle.json"),
            [0u8; 1000],
        )
        .unwrap();
        assert_eq!(memory_admitted(4, first.path(), None), 4);
        assert_eq!(memory_admitted(4, first.path(), Some(8_000)), 2);
        assert_eq!(memory_admitted(4, first.path(), Some(10)), 1);
        assert_eq!(memory_admitted(4, first.path(), Some(1 << 40)), 4);
        assert!(mem_available().is_some_and(|bytes| bytes > 0));
    }

    #[test]
    fn principal_falls_back_to_the_bare_analysis_then_the_first_by_order() {
        let rows = |variants: &[Option<&str>]| -> Vec<Record> {
            variants
                .iter()
                .map(|v| json!({"assay": "oft", "variant": v}))
                .collect()
        };
        let bare = principals(&rows(&[None, Some("zz")]));
        assert_eq!(bare["oft"], None);
        let named = principals(&rows(&[Some("zz"), Some("aa")]));
        assert!(named["oft"].is_some());
    }
}
