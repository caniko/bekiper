use super::*;
use crate::config::{PopulationDecision, PopulationDisposition};

fn plan() -> Record {
    let groups = json!({"G5":{"five_xfad":"AD","hcar1":"WT","treatment":"lactate"}});
    let policies = json!({"primary":"A","policies":["A","B","C"]});
    json!({"inputs":{"group_mapping":groups,"factor_levels":{"five_xfad":["WT","AD"],"hcar1":["WT","KO","Het"],"treatment":["lactate","vehicle"]},
        "session_key_mapping":{"group":{"5":"G5"},"five_xfad":{"tg":"AD"}}},"statistics":{"hcar1_policy_plan":policies},
        "decision_resolutions":{"D10":{"origin":"researcher_decision","rationale":"approved policies","approval_or_plan_reference":"researcher",
            "scope":["inputs","statistics"],"value":{"inputs.group_mapping":groups,"statistics.hcar1_policy_plan":policies}}}})
}

fn ledger() -> PopulationLedger {
    let identity = PopulationFile {
        path: "source".into(),
        sha256: "workbook-hash".into(),
    };
    let mut rows = Vec::new();
    for animal in [1, 2, 3] {
        for threshold in [25, 45, 65] {
            rows.push(PopulationRow {
                decision: PopulationDecision {
                    analysis_id: format!("video_{animal}__proximal{threshold}"),
                    animal_id: Some(animal),
                    disposition: PopulationDisposition::Admitted,
                    reason: None,
                },
                recording_key: format!("recording-{animal}"),
                task_type: TaskType::YMaze,
                video: Some(format!("Y-maze (post)/post.video-{animal}.mp4").into()),
                pose_sources: vec![],
                session_context: json!({"animal":animal.to_string(),"phase_part":"2"}),
                task_params: json!({}),
            });
        }
    }
    PopulationLedger {
        schema_version: 2,
        config: identity.clone(),
        metadata_workbook: Some(identity),
        metadata_policy: None,
        data_root: ".".into(),
        pre_filter_analyses: 9,
        admitted_analyses: 9,
        excluded_analyses: 0,
        blocked_analyses: 0,
        pre_filter_recordings: 3,
        admitted_recordings: 3,
        rows,
        fingerprint: "population-hash".into(),
    }
}

fn biology() -> BTreeMap<u32, Record> {
    [1,2].into_iter().map(|id|(id,json!({"Animal ID":id.to_string(),"HCAR1":if id==1 {"WT"} else {" Het"},
        "Group":"5","5XFAD":"tg","Sex":"female","Treatment":"lactate","Age group":"mid","Treatment onset":"early",
        "All comments":"literal\ncomment","source_row":id+1}))).collect()
}

fn authorization() -> ExclusionAuthorization {
    ExclusionAuthorization {
        population_fingerprint: "population-hash".into(),
        workbook_sha256: "workbook-hash".into(),
        animal_ids: BTreeSet::from([3]),
        reason: "missing_source_biology".into(),
        reference: "researcher-decision".into(),
    }
}

#[test]
fn variants_share_metadata_and_authorized_exclusion_is_separate_from_genotype_policy() {
    let ledger = ledger();
    let original = serde_json::to_value(&ledger).unwrap();
    let result = reconcile_rows(
        &ledger,
        &biology(),
        &plan(),
        None,
        Some(&authorization()),
        "workbook-hash",
    )
    .unwrap();
    assert_eq!(result.recordings.len(), 3);
    assert_eq!(result.analysis_session_mapping.len(), 9);
    assert_eq!(result.authorized_excluded_recordings, 1);
    assert_eq!(result.remaining_recordings, 2);
    assert_eq!(
        result
            .policy_candidates
            .iter()
            .map(|r| r["candidate_animals"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [1, 2, 2]
    );
    let het = &result.metadata[1];
    assert_eq!(het["hcar1"], " Het");
    assert_eq!(het["excluded"], false);
    assert_eq!(
        het["source_metadata"]["workbook"]["All comments"],
        "literal\ncomment"
    );
    let missing = &result.metadata[2];
    assert_eq!(missing["excluded"], true);
    assert_eq!(missing["exclusion_authorization"], "researcher-decision");
    assert!(missing["hcar1"].is_null());
    assert_eq!(result.scientific_readiness, "unverified");
    assert!(!result.analysis_admitted);
    assert!(
        result
            .recordings
            .iter()
            .all(|r| r["canonical_registry_recording_id"].is_null())
    );
    assert_eq!(serde_json::to_value(&ledger).unwrap(), original);
}

#[test]
fn stale_or_overbroad_authorization_cannot_exclude_an_animal_with_recorded_metadata() {
    let mut auth = authorization();
    auth.animal_ids.insert(2);
    assert!(
        reconcile_rows(
            &ledger(),
            &biology(),
            &plan(),
            None,
            Some(&auth),
            "workbook-hash"
        )
        .is_err()
    );
    auth = authorization();
    assert!(
        reconcile_rows(
            &ledger(),
            &biology(),
            &plan(),
            None,
            Some(&auth),
            "changed-workbook"
        )
        .is_err()
    );
    auth.reference.clear();
    assert!(
        reconcile_rows(
            &ledger(),
            &biology(),
            &plan(),
            None,
            Some(&auth),
            "workbook-hash"
        )
        .is_err()
    );
    let result =
        reconcile_rows(&ledger(), &biology(), &plan(), None, None, "workbook-hash").unwrap();
    assert_eq!(result.authorized_excluded_recordings, 0);
    assert_eq!(result.missing_animal_ids, BTreeSet::from([3]));
}

#[test]
fn recording_variants_cannot_hide_conflicting_session_or_animal_bindings() {
    let mut ledger = ledger();
    ledger.rows[1].session_context["animal"] = "different".into();
    assert!(reconcile_rows(&ledger, &biology(), &plan(), None, None, "workbook-hash").is_err());
    let mut ledger = super::tests::ledger();
    ledger.rows[1].recording_key = "other-recording".into();
    assert!(reconcile_rows(&ledger, &biology(), &plan(), None, None, "workbook-hash").is_err());
}

#[test]
fn historical_context_conflicts_are_retained_without_reassigning_biology() {
    let mut ledger = ledger();
    for row in &mut ledger.rows {
        if row.decision.animal_id == Some(3) {
            row.session_context["animal"] = "99".into();
        }
    }
    let result = reconcile_rows(
        &ledger,
        &biology(),
        &plan(),
        None,
        Some(&authorization()),
        "workbook-hash",
    )
    .unwrap();
    assert_eq!(result.metadata[2]["animal_id"], "3");
    assert_eq!(
        result.recordings[2]["source_context_conflicts"],
        json!(["animal"])
    );
    assert_eq!(
        result.metadata[2]["source_metadata"]["session_context"]["animal"],
        "99"
    );
}

#[test]
fn original_geometry_csv_retains_forward_fill_provenance_and_cannot_supply_biology() {
    let dir =
        std::env::temp_dir().join(format!("bekiper-reconciliation-csv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("metadata.csv");
    std::fs::write(
        &path,
        "Phase,PhasePart,Animal,Stage,HCAR1,Perimeter\nA,1,3,0, -/-,arena\n,,1,0, +/+,other\n",
    )
    .unwrap();
    let rows = geometry_csv::load(&path).unwrap();
    let video = bekiper_annotation::VideoContext {
        animal: 1,
        stage: 0,
        phase_part: 1,
    };
    let mapping = rows.resolve(Some(1), Some(&video)).unwrap();
    assert_eq!(mapping["source_fields"]["HCAR1"], " +/+");
    assert_eq!(
        mapping["source_fields"]["binding_identity"]["phase_part_source_row"],
        2
    );
    assert_eq!(
        mapping["source_fields"]["binding_identity"]["cohort_phase"],
        "A"
    );
    assert_eq!(mapping["source_fields"]["source_row"], 3);
    std::fs::write(&path, "Phase,PhasePart,Animal,Stage\nA,1,1,0\n,,1,0\n").unwrap();
    assert!(geometry_csv::load(&path).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
