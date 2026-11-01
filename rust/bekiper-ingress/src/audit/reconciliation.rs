//! Recording-bound researcher metadata handoff. This is an observation, not run admission.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use bekiper_post_analysis::{Record, genotype, inputs};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::population::{PopulationFile, PopulationLedger, PopulationRow};
use crate::aggregate::metadata::read_workbook_source;
use bekiper_core::types::TaskType;

mod geometry_csv;
#[cfg(test)]
mod tests;
mod ymaze;

/// An explicit decision bound to the exact pre-filter population and workbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExclusionAuthorization {
    pub population_fingerprint: String,
    pub workbook_sha256: String,
    pub animal_ids: BTreeSet<u32>,
    pub reason: String,
    pub reference: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationReconciliation {
    pub schema_version: u32,
    pub population_fingerprint: String,
    pub sources: Vec<PopulationFile>,
    pub authorization: Option<ExclusionAuthorization>,
    pub recordings: Vec<Record>,
    pub metadata: Vec<Record>,
    pub analysis_session_mapping: BTreeMap<String, String>,
    pub policy_candidates: Vec<Record>,
    pub missing_animal_ids: BTreeSet<u32>,
    pub authorized_excluded_recordings: usize,
    pub remaining_recordings: usize,
    pub scientific_readiness: String,
    pub analysis_admitted: bool,
    pub workbook_sheet: String,
    pub task_counts: BTreeMap<String, Record>,
    pub animal_coding_audit: Vec<Record>,
    pub code: Record,
}

#[derive(Debug)]
pub struct ReconciliationInputs<'a> {
    pub population: &'a Path,
    pub workbook: &'a Path,
    pub workbook_sheet: &'a str,
    pub plan: &'a Path,
    pub ymaze_ods: Option<&'a Path>,
    pub metadata_csv: Option<&'a Path>,
    pub authorization: Option<&'a Path>,
}

/// Decode the native ledger and original sources directly; no extracted intermediate is trusted.
pub fn reconcile_population(
    inputs: &ReconciliationInputs<'_>,
) -> bekiper_core::error::Result<PopulationReconciliation> {
    reconcile(inputs).map_err(|e| {
        bekiper_core::error::BekiperError::Data(format!("population reconciliation: {e:#}"))
    })
}

fn capture(path: &Path) -> anyhow::Result<PopulationFile> {
    let path = path.canonicalize()?;
    Ok(PopulationFile {
        sha256: inputs::fingerprint(&path)?["sha256"]
            .as_str()
            .context("source SHA-256")?
            .into(),
        path,
    })
}

fn reconcile(source: &ReconciliationInputs<'_>) -> anyhow::Result<PopulationReconciliation> {
    let original_paths: Vec<_> = [source.population, source.workbook, source.plan]
        .into_iter()
        .chain(source.ymaze_ods)
        .chain(source.metadata_csv)
        .chain(source.authorization)
        .collect();
    let sources = original_paths
        .iter()
        .map(|path| capture(path))
        .collect::<anyhow::Result<Vec<_>>>()?;
    // Decode the captured canonical path; a replacement symlink cannot select
    // different bytes while the final recheck keeps hashing the original target.
    let mut paths = sources.iter().map(|s| s.path.as_path());
    let ledger_path = paths.next().context("population source")?;
    let workbook = paths.next().context("workbook source")?;
    let plan_path = paths.next().context("plan source")?;
    let ymaze_ods = source
        .ymaze_ods
        .map(|_| paths.next().context("ODS source"))
        .transpose()?;
    let metadata_csv = source
        .metadata_csv
        .map(|_| paths.next().context("CSV source"))
        .transpose()?;
    let authorization = source
        .authorization
        .map(|_| paths.next().context("authorization source"))
        .transpose()?;
    let authorization = authorization
        .map(|path| {
            bekiper_core::handoff::from_str::<ExclusionAuthorization>(&std::fs::read_to_string(
                path,
            )?)
        })
        .transpose()?;
    let ledger: PopulationLedger =
        bekiper_core::handoff::from_str(&std::fs::read_to_string(ledger_path)?)?;
    ledger.validate()?;
    if let Some(expected) = &ledger.metadata_workbook {
        ensure!(
            expected.sha256 == sources[1].sha256,
            "workbook differs from population ledger"
        );
    }
    let plan = inputs::resolved_config(plan_path, &Default::default(), &|_| None)?;
    ensure!(
        genotype::policy_plan(&plan)?.is_some(),
        "register the HCAR1 policy plan before reconciliation"
    );
    let biology = workbook_animals(read_workbook_source(workbook, source.workbook_sheet)?)?;
    let maze = ymaze_ods.map(ymaze::load).transpose()?;
    let geometry = metadata_csv.map(geometry_csv::load).transpose()?;
    let mut result = reconcile_rows(
        &ledger,
        &biology,
        &plan,
        maze.as_ref(),
        authorization.as_ref(),
        &sources[1].sha256,
    )?;
    if let Some(geometry) = geometry {
        for (record, meta) in result.recordings.iter_mut().zip(&mut result.metadata) {
            if meta["assay"] == "ymaze" {
                continue;
            }
            let animal = record["animal_id"]
                .as_u64()
                .map(u32::try_from)
                .transpose()?;
            let video = record["video"]
                .as_str()
                .map(Path::new)
                .and_then(bekiper_annotation::extract_video_context);
            let mapping = geometry.resolve(animal, video.as_ref())?;
            meta["source_metadata"]["geometry_csv_mapping"] = mapping.clone();
            meta["cohort_phase"] =
                mapping["source_fields"]["binding_identity"]["cohort_phase"].clone();
            if meta["assay"] == "nort" {
                for (target, column) in [
                    ("source_object_A_up_down", "Object A (D=down, U=up)"),
                    ("source_object_B_up_down", "Object B (D=down, U=up)"),
                ] {
                    record["nort_role_side_review"][target] =
                        mapping["source_fields"][column].clone();
                }
                record["nort_role_side_review"]["cohort_phase"] = meta["cohort_phase"].clone();
                record["nort_role_side_review"]["source_row"] =
                    mapping["source_fields"]["source_row"].clone();
                let video = record["video"].as_str().unwrap_or("").to_ascii_lowercase();
                let placement = if video.contains("configa") || video.contains("config a") {
                    Some("ConfigA")
                } else if video.contains("configb") || video.contains("config b") {
                    Some("ConfigB")
                } else {
                    None
                };
                record["nort_role_side_review"]["video_configuration_name"] = json!(placement);
            }
            record["geometry_csv_mapping"] = mapping;
        }
    }
    for (original, source) in original_paths.iter().zip(&sources) {
        ensure!(
            capture(original)? == *source && capture(&source.path)? == *source,
            "source changed during reconciliation: {}",
            source.path.display()
        );
    }
    result.sources = sources;
    result.workbook_sheet = source.workbook_sheet.into();
    Ok(result)
}

fn workbook_animals(rows: Vec<Record>) -> anyhow::Result<BTreeMap<u32, Record>> {
    let mut animals = BTreeMap::new();
    for row in rows {
        if row["Animal ID"].is_null() {
            continue;
        }
        let id = number(&row["Animal ID"], "workbook Animal ID")?;
        ensure!(
            animals.insert(id, row).is_none(),
            "duplicate workbook animal {id}"
        );
    }
    Ok(animals)
}

fn number(value: &Value, field: &str) -> anyhow::Result<u32> {
    if let Some(value) = value.as_u64() {
        return u32::try_from(value).with_context(|| field.to_string());
    }
    value
        .as_str()
        .context("explicit numeric identity required")?
        .trim()
        .parse()
        .with_context(|| format!("invalid {field}: {value}"))
}

fn text<'a>(value: &'a Value, field: &str) -> anyhow::Result<&'a str> {
    value
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .with_context(|| format!("missing {field}"))
}

fn recordings(ledger: &PopulationLedger) -> anyhow::Result<BTreeMap<&str, Vec<&PopulationRow>>> {
    let mut grouped: BTreeMap<&str, Vec<&PopulationRow>> = BTreeMap::new();
    let mut native_keys = BTreeMap::new();
    for row in &ledger.rows {
        let base = row
            .decision
            .analysis_id
            .split("__")
            .next()
            .context("native recording ID")?;
        if let Some(previous) = native_keys.insert(base, row.recording_key.as_str()) {
            ensure!(
                previous == row.recording_key,
                "native recording ID refers to multiple recordings: {base}"
            );
        }
        let rows = grouped.entry(&row.recording_key).or_default();
        if let Some(first) = rows.first() {
            ensure!(
                first.decision.animal_id == row.decision.animal_id
                    && first.task_type == row.task_type
                    && first.video == row.video
                    && first.pose_sources == row.pose_sources
                    && first.session_context == row.session_context
                    && first.decision.analysis_id.split("__").next() == Some(base),
                "analysis variants disagree on recording identity: {}",
                row.recording_key
            );
        }
        rows.push(row);
    }
    Ok(grouped)
}

fn reconcile_rows(
    ledger: &PopulationLedger,
    biology: &BTreeMap<u32, Record>,
    plan: &Record,
    maze: Option<&ymaze::Mappings>,
    authorization: Option<&ExclusionAuthorization>,
    workbook_hash: &str,
) -> anyhow::Result<PopulationReconciliation> {
    let grouped = recordings(ledger)?;
    let missing_animal_ids: BTreeSet<_> = grouped
        .values()
        .filter_map(|rows| rows[0].decision.animal_id)
        .filter(|id| !biology.contains_key(id))
        .collect();
    if let Some(auth) = authorization {
        ensure!(
            auth.population_fingerprint == ledger.fingerprint
                && auth.workbook_sha256 == workbook_hash,
            "exclusion authorization is bound to different population/workbook bytes"
        );
        ensure!(
            !auth.reason.trim().is_empty()
                && !auth.reference.trim().is_empty()
                && !auth.animal_ids.is_empty(),
            "exclusion authorization requires animals, reason and reference"
        );
        ensure!(
            auth.animal_ids.is_subset(&missing_animal_ids),
            "exclusion authorization includes an absent or metadata-present animal"
        );
    }
    let mut metadata = Vec::new();
    let mut records = Vec::new();
    let mut session_map = BTreeMap::new();
    for (key, variants) in grouped {
        let row = variants[0];
        let id = row.decision.animal_id;
        let source_biology = id.and_then(|id| biology.get(&id));
        let excluded =
            id.is_some_and(|id| authorization.is_some_and(|auth| auth.animal_ids.contains(&id)));
        let recording = row
            .decision
            .analysis_id
            .split("__")
            .next()
            .context("native recording ID")?;
        let assay = match row.task_type {
            TaskType::OpenField => "oft",
            TaskType::YMaze => "ymaze",
            TaskType::ObjectRecognition => "nort",
            TaskType::Cheeseboard => anyhow::bail!("unsupported reconciliation assay: cheeseboard"),
        };
        let video_context = row
            .video
            .as_deref()
            .and_then(bekiper_annotation::extract_video_context);
        let timepoint = video_context
            .as_ref()
            .map(|c| if c.phase_part == 1 { "pre" } else { "post" });
        let phase = if assay == "nort" {
            video_context.as_ref().and_then(|c| match c.stage {
                1 => Some("training"),
                2 => Some("test"),
                _ => None,
            })
        } else {
            Some("session")
        };
        let mut conflicts = Vec::new();
        for (field, expected) in [
            ("animal", id),
            ("video_animal", video_context.as_ref().map(|c| c.animal)),
            ("phase_part", video_context.as_ref().map(|c| c.phase_part)),
            ("stage", video_context.as_ref().map(|c| c.stage)),
        ] {
            if field == "video_animal" {
                if expected.is_some() && id != expected {
                    conflicts.push(field);
                }
                continue;
            }
            if !row.session_context[field].is_null()
                && expected.is_some()
                && number(&row.session_context[field], field)?
                    != expected.context("source identity")?
            {
                conflicts.push(field);
            }
        }
        let maze_mapping = if assay == "ymaze" {
            maze.map(|maze| maze.resolve(id, video_context.as_ref().map(|c| c.phase_part)))
                .transpose()?
        } else {
            None
        };
        let cohort = maze_mapping
            .as_ref()
            .filter(|m| m["phase"].is_string())
            .map(|m| m["phase"].clone())
            .unwrap_or_else(|| row.session_context["phase"].clone());
        let mut meta = json!({"animal_id":id.map(|id|id.to_string()),"assay":assay,"session_id":recording,
            "timepoint":timepoint,"phase":phase,"pair_id":id.map(|id|format!("animal-{id}/{assay}/{}",phase.unwrap_or("unresolved"))),
            "group":null,"five_xfad":null,"hcar1":null,"treatment":null,"sex":null,"age_treatment_window":null,
            "excluded":excluded,"exclusion_reason":if excluded { authorization.map(|a|a.reason.as_str()) } else { None },
            "exclusion_authorization":if excluded { authorization.map(|a|a.reference.as_str()) } else { None },
            "recording_id":recording,"retained_recording_key":key,"canonical_registry_recording_id":null,
            "cohort_phase":cohort,"historical_context_conflicts":conflicts,
            "phase_evidence":if assay == "nort" {"recording_T1_T2_recipe_partition"} else {"single_session_assay"},
            "source_metadata":{"workbook":source_biology,"session_context":row.session_context,"ymaze_ods_mapping":maze_mapping},
            "input_verification":{"status":"unresolved","reason":"source_observation_requires_current_registered_seal"}});
        if let Some(biology) = source_biology {
            for (source, target) in [
                ("Group", "group"),
                ("HCAR1", "hcar1"),
                ("5XFAD", "five_xfad"),
                ("Treatment", "treatment"),
                ("Sex", "sex"),
            ] {
                meta[target] = biology[source].clone();
            }
            if !biology["Age group"].is_null() && !biology["Treatment onset"].is_null() {
                meta["age_treatment_window"] = format!(
                    "{}/{}",
                    text(&biology["Age group"], "age group")?,
                    text(&biology["Treatment onset"], "treatment onset")?
                )
                .into();
            }
            inputs::apply_session_value_mappings(
                &mut meta,
                &plan["inputs"]["session_key_mapping"],
            )?;
        }
        let analyses: Vec<_> = variants
            .iter()
            .map(|r| r.decision.analysis_id.clone())
            .collect();
        for analysis in &analyses {
            session_map.insert(analysis.clone(), recording.into());
        }
        records.push(json!({"retained_recording_key":key,"native_recording_id":recording,"canonical_registry_recording_id":null,
            "animal_id":id,"task_type":row.task_type,"video":row.video,"pose_sources":row.pose_sources,
            "historical_analysis_ids":analyses,"excluded":excluded,"exclusion_reason":meta["exclusion_reason"],
            "historical_population_decisions":variants.iter().map(|r|&r.decision).collect::<Vec<_>>(),
            "source_context_conflicts":conflicts,"ymaze_ods_mapping":maze_mapping,
            "nort_role_side_review":if assay == "nort" { json!({"source_object_A_up_down":row.session_context["object_a"],
                "source_object_B_up_down":row.session_context["object_b"],"roles":null,"physical_sides":null,
                "source_column_definitions":{"object_A":"Object A (D=down, U=up)","object_B":"Object B (D=down, U=up)"},
                "cohort_phase":cohort,"object_configuration":null,
                "status":"requires_recording_specific_role_geometry_and_camera_side_review"}) } else { Value::Null },
            "binding_status":"requires_current_registered_binding_and_execution_seal"}));
        metadata.push(meta);
    }
    // Validate the canonical biology contract without claiming that unresolved session keys are usable.
    let complete: Vec<_> = metadata
        .iter()
        .filter(|r| inputs::session_key(r).is_ok())
        .cloned()
        .collect();
    inputs::validate_metadata(&complete, plan)?;
    let mut policies = Vec::new();
    for policy in ["A", "B", "C"] {
        let (coded, _) = genotype::project(&metadata, plan, policy)?;
        let eligible: Vec<_> = coded
            .iter()
            .filter(|r| r["excluded"] != true && r["genotype_eligibility"]["eligible"] == true)
            .collect();
        let animals: BTreeSet<_> = eligible
            .iter()
            .filter_map(|r| r["animal_id"].as_str())
            .collect();
        policies.push(json!({"policy":policy,"primary":policy=="A","candidate_animals":animals.len(),"candidate_recordings":eligible.len(),
            "scope":"source_genotype_candidates_not_endpoint_or_model_eligibility"}));
    }
    let excluded = records.iter().filter(|r| r["excluded"] == true).count();
    let mut task_counts = BTreeMap::new();
    for task in ["oft", "nort", "ymaze"] {
        let rows: Vec<_> = metadata.iter().filter(|r| r["assay"] == task).collect();
        task_counts.insert(task.into(),json!({"recordings":rows.len(),"authorized_excluded":rows.iter().filter(|r|r["excluded"]==true).count(),
            "remaining":rows.iter().filter(|r|r["excluded"]!=true).count()}));
    }
    let mut animal_coding_audit = Vec::new();
    for (&id, biology) in biology {
        let animal_id = id.to_string();
        let meta = metadata.iter().find(|r| r["animal_id"] == animal_id);
        let mut audit = json!({"animal_id":id,"workbook":biology,"policies":[]});
        if let Some(meta) = meta {
            for policy in ["A", "B", "C"] {
                let (rows, _) = genotype::project(std::slice::from_ref(meta), plan, policy)?;
                audit["policies"].as_array_mut().context("coding audit")?.push(json!({"policy":policy,
                    "eligibility":rows[0]["genotype_eligibility"],"coding":rows[0]["analysis_factor_coding"]}));
            }
        }
        animal_coding_audit.push(audit);
    }
    Ok(PopulationReconciliation {
        schema_version: 1,
        population_fingerprint: ledger.fingerprint.clone(),
        sources: Vec::new(),
        authorization: authorization.cloned(),
        remaining_recordings: records.len() - excluded,
        authorized_excluded_recordings: excluded,
        recordings: records,
        metadata,
        analysis_session_mapping: session_map,
        policy_candidates: policies,
        missing_animal_ids,
        scientific_readiness: "unverified".into(),
        analysis_admitted: false,
        workbook_sheet: String::new(),
        task_counts,
        animal_coding_audit,
        code: json!({"package":env!("CARGO_PKG_NAME"),"version":env!("CARGO_PKG_VERSION"),
            "reconciliation_source_sha256":env!("BEKIPER_RECONCILIATION_VERSION"),"policy_engine":bekiper_post_analysis::code_identity()}),
    })
}

/// Publish a fresh handoff atomically, outside the source inputs.
pub fn export_reconciliation(
    report: &PopulationReconciliation,
    output: &Path,
) -> anyhow::Result<PathBuf> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let destination = parent.join(output.file_name().context("handoff output directory")?);
    for source in &report.sources {
        ensure!(
            !source.path.starts_with(&destination),
            "handoff output overlaps a source input"
        );
        ensure!(
            capture(&source.path)? == *source,
            "reconciliation source changed before publication"
        );
    }
    let mut publication = bekiper_post_analysis::pipeline::Publication::claim(&destination)?;
    let directory = publication.directory();
    for (name, value) in [
        ("reconciliation.json", serde_json::to_value(report)?),
        ("metadata.json", json!(report.metadata)),
        (
            "analysis-session-mapping.json",
            json!(report.analysis_session_mapping),
        ),
    ] {
        bekiper_post_analysis::pipeline::write_json(&directory.join(name), &value)?;
    }
    for source in &report.sources {
        ensure!(
            capture(&source.path)? == *source,
            "reconciliation source changed during publication"
        );
    }
    bekiper_post_analysis::pipeline::write_artifact_manifest(directory)?;
    publication.publish()?;
    Ok(destination)
}
