use super::*;

#[test]
fn legacy_excel_alias_writes_the_spreadsheet_and_handles_empty_heuristics() {
    let root = std::env::temp_dir().join(format!("bekiper_export_alias_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = ExperimentOutputDir::new(&root, "trial");
    dir.ensure_dir().unwrap();
    let summaries = [HeuristicSummary {
        name: "never_evaluated".into(),
        true_frames: 0,
        total_frames: 0,
        seconds: 0.0,
    }];
    export_excel(&dir, "trial", &summaries, None, None).unwrap();
    assert!(dir.root().join("trial.ods").is_file());
    let sheet = format!("{:?}", summary_sheet(&summaries));
    assert!(sheet.contains("Float(0.0)"), "{sheet}");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn detailed_sheet_keeps_counts_beyond_i64_as_text() {
    let detail = serde_json::json!({"small": u64::MAX >> 1, "large": u64::MAX});
    let sheet = format!("{:?}", detailed_sheet(detail.as_object().unwrap()));
    assert!(
        sheet.contains(&format!("String(\"{}\")", u64::MAX)),
        "{sheet}"
    );
    assert!(
        sheet.contains(&format!("Int({})", u64::MAX >> 1)),
        "{sheet}"
    );
}

#[test]
fn borrowed_frames_export_writes_one_sheet_per_frame() {
    let root = std::env::temp_dir().join(format!("bekiper_export_frames_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = ExperimentOutputDir::new(&root, "trial");
    dir.ensure_dir().unwrap();
    let frame = df!("a" => [1i64, 2]).unwrap();
    export_spreadsheet_frames(&dir, "trial", &[], None, None, &[("frame", frame)]).unwrap();
    assert!(dir.root().join("trial.ods").is_file());
    std::fs::remove_dir_all(root).unwrap();
}
