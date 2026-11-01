use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;

use super::IngressConfig;
use crate::config::aggregate::MissingMetadataPolicy;

/// Write a minimal Excel workbook with one `Animal ID` column populated
/// from `ids`. Sheet name is `Updated_Mice-overview` (matches the
/// AggregateMetadataConfig default).
fn write_xlsx_with_animals(path: &Path, ids: &[u32]) {
    let file = std::fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts = SimpleFileOptions::default();

    zip.start_file("xl/workbook.xml", opts).unwrap();
    zip.write_all(
            br#"<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Updated_Mice-overview" sheetId="1" r:id="rId1"/></sheets>
</workbook>"#,
        )
        .unwrap();

    zip.start_file("xl/_rels/workbook.xml.rels", opts).unwrap();
    zip.write_all(
            br#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#,
        )
        .unwrap();

    zip.start_file("xl/sharedStrings.xml", opts).unwrap();
    zip.write_all(
        br#"<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="1" uniqueCount="1">
  <si><t>Animal ID</t></si>
</sst>"#,
    )
    .unwrap();

    zip.start_file("xl/worksheets/sheet1.xml", opts).unwrap();
    let mut sheet = String::from(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c></row>
"#,
    );
    for (idx, id) in ids.iter().enumerate() {
        let row = idx + 2;
        sheet.push_str(&format!(
            "    <row r=\"{row}\"><c r=\"A{row}\"><v>{id}</v></c></row>\n"
        ));
    }
    sheet.push_str("  </sheetData>\n</worksheet>");
    zip.write_all(sheet.as_bytes()).unwrap();

    zip.finish().unwrap();
}

/// Build a project directory containing a minimal config TOML with the
/// supplied experiment names and an aggregate metadata workbook listing
/// `metadata_ids`. Returns `(tempdir, config_path)`.
fn setup_project(
    tag: &str,
    experiment_names: &[&str],
    metadata_ids: Option<&[u32]>,
    policy: MissingMetadataPolicy,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("bekiper_prefilter_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let policy_str = match policy {
        MissingMetadataPolicy::Fail => "fail",
        MissingMetadataPolicy::ExcludeMissing => "exclude_missing",
        MissingMetadataPolicy::AllowBlanks => "allow_blanks",
    };

    let mut toml = String::new();
    toml.push_str(
        r#"[project]
project_name = "prefilter_test"
data_directory = "."
"#,
    );
    if let Some(ids) = metadata_ids {
        write_xlsx_with_animals(&dir.join("animals.xlsx"), ids);
        toml.push_str(&format!(
            "\n[aggregate.metadata]\npath = \"animals.xlsx\"\nmissing_policy = \"{policy_str}\"\n",
        ));
    }
    for name in experiment_names {
        toml.push_str(&format!(
                "\n[[experiments]]\nname = \"{name}\"\ndata_files = [\"trial.csv\"]\ntask_type = \"open_field\"\nfps = 30.0\nmeters_per_pixel = 0.001\n",
            ));
    }

    let config_path = dir.join("bekiper_config.toml");
    std::fs::write(&config_path, toml).unwrap();
    std::fs::write(dir.join("trial.csv"), b"frame\n").unwrap();

    (dir, config_path)
}

#[test]
fn population_retains_exclusions_variants_and_unparseable_identities() {
    let (dir, path) = setup_project(
        "population",
        &[
            "ymaze_video_1__principal_25",
            "ymaze_video_99__principal_25",
            "ymaze_video_99__compare_0pct_dwell0s",
            "unknown",
        ],
        Some(&[1]),
        MissingMetadataPolicy::ExcludeMissing,
    );
    let before = std::fs::read(&path).unwrap();
    let ledger = crate::audit::population_ledger_at(&path, &dir).unwrap();
    assert_eq!(ledger.pre_filter_analyses, 4);
    assert_eq!(ledger.excluded_analyses, 3);
    assert_eq!(ledger.admitted_analyses, 1);
    assert_eq!(ledger.pre_filter_recordings, 1);
    let text = bekiper_core::handoff::to_string(&ledger).unwrap();
    let restored: crate::audit::PopulationLedger = bekiper_core::handoff::from_str(&text).unwrap();
    restored.validate().unwrap();
    assert_eq!(restored.fingerprint, ledger.fingerprint);
    let mut corrupt = ledger.clone();
    corrupt.rows[1].decision.reason = Some("different".into());
    assert!(corrupt.validate().is_err());
    let mut config = IngressConfig::from_file(&path).unwrap();
    let decisions = config.expand_for_population(&dir).unwrap();
    assert_eq!(config.experiments.len(), 4);
    assert_eq!(decisions.len(), 4);
    assert_eq!(
        decisions[0].disposition,
        super::super::PopulationDisposition::Admitted
    );
    assert_eq!(decisions[1].animal_id, Some(99));
    assert_eq!(
        decisions[1].reason.as_deref(),
        Some("missing_animal_metadata")
    );
    assert_eq!(
        decisions[2].disposition,
        super::super::PopulationDisposition::Excluded
    );
    assert_eq!(
        decisions[3].reason.as_deref(),
        Some("unparseable_animal_identity")
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    config.prefilter_experiments_by_metadata(&dir).unwrap();
    assert_eq!(config.experiments.len(), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn population_fail_policy_records_blocked_rows_and_allow_blanks_keeps_rows() {
    for (tag, policy, expected) in [
        (
            "population_fail",
            MissingMetadataPolicy::Fail,
            super::super::PopulationDisposition::Blocked,
        ),
        (
            "population_blanks",
            MissingMetadataPolicy::AllowBlanks,
            super::super::PopulationDisposition::Admitted,
        ),
    ] {
        let (dir, path) = setup_project(tag, &["oft_video_1", "oft_video_99"], Some(&[1]), policy);
        let mut config = IngressConfig::from_file(&path).unwrap();
        let decisions = config.expand_for_population(&dir).unwrap();
        assert_eq!(decisions[1].disposition, expected);
        assert_eq!(config.experiments.len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn prefilter_no_metadata_config_is_noop() {
    let (dir, config_path) = setup_project(
        "no_meta",
        &["oft_pre_video_1", "oft_pre_video_2"],
        None,
        MissingMetadataPolicy::ExcludeMissing,
    );

    let mut config = IngressConfig::from_file(&config_path).unwrap();
    config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap();

    let names: Vec<&str> = config.experiments.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["oft_pre_video_1", "oft_pre_video_2"]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn prefilter_allow_blanks_keeps_unknown_animals() {
    let (dir, config_path) = setup_project(
        "allow_blanks",
        &["oft_pre_video_1", "oft_pre_video_2", "oft_pre_video_99"],
        Some(&[1, 2]),
        MissingMetadataPolicy::AllowBlanks,
    );

    let mut config = IngressConfig::from_file(&config_path).unwrap();
    config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap();

    // AllowBlanks → policy is a no-op at the prefilter; everything
    // survives the pre-flight stage. (Aggregate-time merge will leave
    // metadata blank for animal 99.)
    assert_eq!(config.experiments.len(), 3);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn prefilter_exclude_missing_drops_unknown_animals() {
    let (dir, config_path) = setup_project(
        "exclude_missing",
        &["oft_pre_video_1", "oft_pre_video_99", "oft_pre_video_2"],
        Some(&[1, 2]),
        MissingMetadataPolicy::ExcludeMissing,
    );

    let mut config = IngressConfig::from_file(&config_path).unwrap();
    config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap();

    let names: Vec<&str> = config.experiments.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["oft_pre_video_1", "oft_pre_video_2"]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn prefilter_fail_policy_returns_error_listing_missing_ids() {
    let (dir, config_path) = setup_project(
        "fail",
        &["oft_pre_video_1", "oft_pre_video_99", "oft_pre_video_42"],
        Some(&[1]),
        MissingMetadataPolicy::Fail,
    );

    let mut config = IngressConfig::from_file(&config_path).unwrap();
    let err = config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap_err()
        .to_string();
    // Both unknown IDs surfaced.
    assert!(err.contains("42"), "missing 42 in error: {err}");
    assert!(err.contains("99"), "missing 99 in error: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn prefilter_unparseable_experiment_name_dropped_when_exclude() {
    let (dir, config_path) = setup_project(
        "unparseable",
        &["oft_pre_video_1", "no_animal_id_anywhere"],
        Some(&[1]),
        MissingMetadataPolicy::ExcludeMissing,
    );

    let mut config = IngressConfig::from_file(&config_path).unwrap();
    config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap();

    let names: Vec<&str> = config.experiments.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["oft_pre_video_1"]);

    std::fs::remove_dir_all(&dir).ok();
}
