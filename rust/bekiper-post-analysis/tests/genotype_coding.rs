use bekiper_post_analysis::{genotype, statistics};
use serde_json::{Value, json};

fn config() -> Value {
    let plan = json!({"primary":"A", "policies":["A", "B", "C"]});
    let groups = json!({"G5":{"five_xfad":"AD","hcar1":"WT","treatment":"lactate"},
        "G2":{"five_xfad":"AD","hcar1":"KO","treatment":"vehicle"}});
    json!({"statistics":{"hcar1_policy_plan":plan},"inputs":{"group_mapping":groups},
        "decision_resolutions":{"D10":{"origin":"researcher_decision","rationale":"Three authorized policies",
            "approval_or_plan_reference":"test researcher plan","scope":["statistics","inputs"],
            "value":{"statistics.hcar1_policy_plan":plan,"inputs.group_mapping":groups}}}})
}

fn row(id: &str, hcar1: &str, group: &str) -> Value {
    json!({"animal_id":id,"session_id":format!("post-{id}"),"assay":"nort","timepoint":"post","phase":"test",
        "group":group,"five_xfad":"AD","hcar1":hcar1,"treatment":if group=="G5" {"lactate"}else{"vehicle"},
        "excluded":false,"qc_flags":{"low_quality":true},"DI":0.25})
}

#[test]
fn all_policies_preserve_source_endpoints_and_separate_eligibility() {
    let rows = vec![
        row("1", "WT", "G5"),
        row("2", " Het", "G5"),
        row("37", "Het/KO", "G2"),
    ];
    let original = rows.clone();
    for policy in ["A", "B", "C"] {
        let (coded, _) = genotype::project(&rows, &config(), policy).unwrap();
        assert_eq!(coded.len(), rows.len());
        for (before, after) in rows.iter().zip(&coded) {
            assert_eq!(after["literal_hcar1"], before["hcar1"]);
            assert_eq!(after["assigned_group"], before["group"]);
            assert_eq!(after["DI"], before["DI"]);
            assert_eq!(after["qc_flags"], before["qc_flags"]);
            assert_eq!(after["excluded"], false);
        }
        let (used, audit) =
            statistics::analysis_subset(&coded, &["DI"], Some("allow_unverified")).unwrap();
        assert_eq!(
            used.len(),
            match policy {
                "A" => 1,
                "B" => 2,
                _ => 3,
            }
        );
        assert_eq!(audit.len(), 3);
        if policy == "B" {
            assert_eq!(coded[1]["hcar1"], "Het");
        }
        if policy == "C" {
            assert_eq!(coded[1]["hcar1"], "WT");
            assert_eq!(coded[2]["hcar1"], "KO");
            assert_eq!(
                coded[2]["analysis_factor_coding"]["assumption"],
                "assigned_group_not_genotype_verification"
            );
        }
    }
    assert_eq!(rows, original);
}

#[test]
fn missing_endpoint_is_independent_of_genotype_and_qc_does_not_exclude() {
    let mut rows = vec![row("1", "WT", "G5"), row("2", "Het", "G5")];
    rows[1]["DI"] = Value::Null;
    let (coded, _) = genotype::project(&rows, &config(), "A").unwrap();
    let (_, audit) =
        statistics::analysis_subset(&coded, &["DI"], Some("allow_unverified")).unwrap();
    assert_eq!(audit[1]["genotype_eligibility"]["eligible"], false);
    assert_eq!(audit[1]["missing_fields"], json!(["DI"]));
    assert_eq!(audit[0]["reason"], Value::Null);
}

#[test]
fn unregistered_policy_and_changed_primary_are_rejected() {
    let mut config = config();
    config["statistics"]["hcar1_policy_plan"]["primary"] = "C".into();
    assert!(genotype::policy_plan(&config).is_err());
    config["decision_resolutions"]["D10"] = Value::Null;
    assert!(genotype::policy_plan(&config).is_err());
}
