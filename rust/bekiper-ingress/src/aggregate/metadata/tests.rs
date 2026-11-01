use super::*;
use std::io::Write;
use zip::write::SimpleFileOptions;

#[test]
fn workspace_preflight_accounts_for_zip_expansion_sparse_columns_and_shared_strings() {
    fn workbook(cell: &str, text: &str) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        zip.start_file("xl/sharedStrings.xml", SimpleFileOptions::default())
            .unwrap();
        write!(zip, "<sst><si><t>{text}</t></si></sst>").unwrap();
        zip.start_file("xl/worksheets/sheet1.xml", SimpleFileOptions::default())
            .unwrap();
        write!(zip, "<worksheet><sheetData><row><c r=\"{cell}\" t=\"s\"><v>0</v></c></row></sheetData></worksheet>").unwrap();
        zip.finish().unwrap().into_inner()
    }
    let compact = workbook("A1", "short");
    let sparse = workbook("XFD1", "short");
    let long = workbook("A1", &"a".repeat(10000));
    assert!(metadata_xml_bytes(&compact).unwrap() > 0);
    assert!(
        metadata_workspace_bytes(&sparse).unwrap()
            > metadata_workspace_bytes(&compact).unwrap() + 1024 * 1024
    );
    assert!(metadata_workspace_bytes(&long).unwrap() > metadata_workspace_bytes(&compact).unwrap());
    assert!(metadata_workspace_bytes(&workbook("XXXXX1", "short")).is_err());
}

#[test]
fn animal_id_from_experiment_strips_variant_suffix() {
    // Legacy {animal}_seq{N}_{recipe} name with a variant suffix.
    assert_eq!(
        animal_id_from_experiment("1035_seq3_ymaze__proximal65"),
        Some(1035)
    );
    // Descriptive videomap-style name with a variant suffix.
    assert_eq!(
        animal_id_from_experiment("y_maze_videos_y_maze_pre_pre_video_14__proximal25"),
        Some(14)
    );
    // No-variant names still parse.
    assert_eq!(animal_id_from_experiment("1035_seq3_ymaze"), Some(1035));
    assert_eq!(
        animal_id_from_experiment("y_maze_videos_y_maze_pre_pre_video_14"),
        Some(14)
    );
}

#[test]
fn parse_metadata_table_normalizes_headers() {
    let table = vec![
        vec![
            "Animal ID".into(),
            "Group".into(),
            "Group name".into(),
            "Sex".into(),
            "HCAR1".into(),
            "Age group".into(),
            "Treatment onset".into(),
            "5XFAD".into(),
            "Age group nr".into(),
            "Start age (days)".into(),
            "Start age (months)".into(),
            "End age (days)".into(),
            "End age (months)".into(),
            "Treatment".into(),
            "Round".into(),
            "All comments".into(),
        ],
        vec![
            "1".into(),
            "3".into(),
            "Group 3\nWT".into(),
            "male".into(),
            "KO".into(),
            "old".into(),
            "late".into(),
            "wt".into(),
            "1".into(),
            "99".into(),
            "3.3".into(),
            "182".into(),
            "6.06".into(),
            "lactate".into(),
            "1".into(),
            "Kicks".into(),
        ],
    ];

    let parsed = parse_metadata_table("metadata.xlsx".into(), table).unwrap();
    let row = parsed.rows.get(&1).unwrap();
    assert_eq!(row.get("animal"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("group"), Some(&CellValue::Int(3)));
    assert_eq!(
        row.get("group_name"),
        Some(&CellValue::String("Group 3 WT".into()))
    );
    assert_eq!(row.get("five_xfad"), Some(&CellValue::String("wt".into())));
    assert_eq!(row.get("start_age_months"), Some(&CellValue::Float(3.3)));
}

#[test]
fn parse_metadata_table_rejects_duplicate_animals() {
    let table = vec![
        vec!["Animal ID".into(), "Group".into()],
        vec!["1".into(), "3".into()],
        vec!["1".into(), "4".into()],
    ];
    let err = parse_metadata_table("metadata.xlsx".into(), table).unwrap_err();
    assert!(err.to_string().contains("duplicate Animal ID `1`"));
}

#[test]
fn read_xlsx_sheet_loads_minimal_workbook() {
    let dir = std::env::temp_dir().join("bekiper_metadata_xlsx_minimal");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("metadata.xlsx");
    write_minimal_xlsx(&path);

    let table = read_xlsx_sheet(&path, "Updated_Mice-overview").unwrap();
    assert_eq!(table[0][0], "Animal ID");
    assert_eq!(table[0][1], "Group");
    assert_eq!(table[1][0], "1");
    assert_eq!(table[1][1], "3");
    let source = read_workbook_source(&path, "Updated_Mice-overview").unwrap();
    assert_eq!(source[0]["source_row"], 2);
    assert_eq!(source[0]["Animal ID"], "1");
}

#[test]
fn source_workbook_does_not_silently_replace_corrupt_or_duplicate_cells() {
    for xml in [
        "<worksheet><row r=\"1\"><c r=\"A1\" t=\"s\"><v>999</v></c></row></worksheet>",
        "<worksheet><row r=\"1\"><c r=\"A1\"/><c r=\"A1\"/></row></worksheet>",
        "<worksheet><row r=\"1\"/><row r=\"1\"/></worksheet>",
    ] {
        assert!(worksheet_source_rows(xml, &[]).is_err());
    }
    assert!(cell_column_index(&"A".repeat(100)).is_none());
}

#[test]
fn enrich_trials_excludes_missing_metadata() {
    let mut rows = vec![
        trial("oft_videos_oft_pre_pre_video_1"),
        trial("oft_videos_oft_pre_pre_video_2"),
    ];
    let parsed = parse_metadata_table(
        "metadata.xlsx".into(),
        vec![
            vec!["Animal ID".into(), "Group".into(), "Sex".into()],
            vec!["1".into(), "3".into(), "male".into()],
        ],
    )
    .unwrap();

    let exclusions =
        enrich_trials_with_metadata(&mut rows, &parsed, MissingMetadataPolicy::ExcludeMissing)
            .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(trial_animal_id(&rows[0]), Some(1));
    assert_eq!(exclusions[0].animal, 2);
    assert_eq!(exclusions[0].reason, "missing_animal_metadata");
    assert_eq!(rows[0].animal_metadata["group"], serde_json::json!(3));
    assert_eq!(rows[0].animal_metadata["sex"], serde_json::json!("male"));
}

fn trial(experiment: &str) -> TrialResult {
    TrialResult {
        experiment: experiment.into(),
        task_type: bekiper_core::types::TaskType::OpenField,
        analysis_role: None,
        fps: 30.0,
        meters_per_pixel: 0.001,
        animal_metadata: Value::Null,
        session_context: Value::Null,
        summaries: Vec::new(),
        detailed: Value::Null,
    }
}

fn write_minimal_xlsx(path: &Path) {
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
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
  <si><t>Animal ID</t></si><si><t>Group</t></si>
</sst>"#,
    )
    .unwrap();
    zip.start_file("xl/worksheets/sheet1.xml", opts).unwrap();
    zip.write_all(
        br#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
    <row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>3</v></c></row>
  </sheetData>
</worksheet>"#,
    )
    .unwrap();
    zip.finish().unwrap();
}

fn zip_bytes(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, bytes) in files {
        zip.start_file(*name, SimpleFileOptions::default()).unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

const WORKBOOK: &str =
    r#"<workbook xmlns:r="r"><sheets><sheet name="Sheet" r:id="rId1"/></sheets></workbook>"#;

fn rels(target: &str) -> String {
    format!(r#"<Relationships><Relationship Id="rId1" Target="{target}"/></Relationships>"#)
}

/// One-sheet workbook with inline worksheet XML.
fn workbook_file(dir: &Path, name: &str, sheet_xml: &str) -> PathBuf {
    let path = dir.join(name);
    let rels = rels("worksheets/sheet1.xml");
    std::fs::write(
        &path,
        zip_bytes(&[
            ("xl/workbook.xml", WORKBOOK.as_bytes()),
            ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
            ("xl/worksheets/sheet1.xml", sheet_xml.as_bytes()),
        ]),
    )
    .unwrap();
    path
}

fn metadata_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper_metadata_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn metadata_loading_resolves_paths_sheets_and_relationships() {
    let dir = metadata_dir("loading");
    let cfg = |path: PathBuf, sheet: &str| AggregateMetadataConfig {
        path,
        sheet: sheet.into(),
        missing_policy: MissingMetadataPolicy::Fail,
    };
    fn error<T: std::fmt::Debug>(result: Result<T>) -> String {
        result.unwrap_err().to_string()
    }
    assert!(
        error(load_aggregate_metadata(
            &dir,
            &cfg("missing.xlsx".into(), "Sheet")
        ))
        .contains("workbook is missing")
    );
    let inline = "<worksheet><row><c t=\"inlineStr\"><is><t>Animal ID</t></is></c></row><row><c><v>4</v></c></row></worksheet>";
    let absolute = workbook_file(&dir, "inline.xlsx", inline);
    let loaded =
        load_aggregate_metadata(Path::new("/elsewhere"), &cfg(absolute.clone(), "Sheet")).unwrap();
    assert_eq!(loaded.rows[&4]["animal"], CellValue::Int(4));
    assert!(
        error(load_aggregate_metadata(
            &dir,
            &cfg(absolute.clone(), "Other")
        ))
        .contains("does not contain sheet `Other`")
    );
    assert!(error(read_xlsx_sheet(&dir.join("absent.xlsx"), "Sheet")).contains("failed to open"));
    std::fs::write(dir.join("text.xlsx"), "not a zip").unwrap();
    assert!(
        error(read_xlsx_sheet(&dir.join("text.xlsx"), "Sheet"))
            .contains("failed to read aggregate metadata XLSX")
    );
    // Relationship targets may be package-absolute or missing.
    let write = |name: &str, rels: &str| {
        let path = dir.join(name);
        std::fs::write(
            &path,
            zip_bytes(&[
                ("xl/workbook.xml", WORKBOOK.as_bytes()),
                ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
                ("xl/worksheets/sheet1.xml", inline.as_bytes()),
            ]),
        )
        .unwrap();
        path
    };
    let rooted = write("rooted.xlsx", &rels("xl/worksheets/sheet1.xml"));
    assert_eq!(read_xlsx_sheet(&rooted, "Sheet").unwrap()[1], ["4"]);
    let unrelated = write("unrelated.xlsx", "<Relationships/>");
    assert!(error(read_xlsx_sheet(&unrelated, "Sheet")).contains("has no worksheet relationship"));
    let no_workbook = dir.join("no_workbook.xlsx");
    std::fs::write(&no_workbook, zip_bytes(&[("xl/other.xml", b"<x/>")])).unwrap();
    assert!(error(read_xlsx_sheet(&no_workbook, "Sheet")).contains("missing `xl/workbook.xml`"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn worksheet_rows_reject_bad_references_formula_errors_and_oversized_expansion() {
    let error = |xml: &str, strings: &[String]| {
        worksheet_source_rows(xml, strings).unwrap_err().to_string()
    };
    assert!(
        error("<worksheet><row><c r=\"1A\"/></row></worksheet>", &[])
            .contains("invalid metadata cell reference 1A")
    );
    assert!(
        error("<worksheet><row><c r=\"XFE1\"/></row></worksheet>", &[])
            .contains("exceeds XLSX limits")
    );
    assert!(
        error(
            "<worksheet><row><c t=\"e\"><v>#DIV/0!</v></c></row></worksheet>",
            &[]
        )
        .contains("formula error #DIV/0!")
    );
    let big = vec!["x".repeat(1024 * 1024)];
    let cells = "<c t=\"s\"><v>0</v></c>".repeat(65);
    assert!(
        error(&format!("<worksheet><row>{cells}</row></worksheet>"), &big)
            .contains("oversized expanded metadata text")
    );
    let wide = "<row><c r=\"XFD1\"/></row>".repeat(245);
    assert!(
        error(&format!("<worksheet>{wide}</worksheet>"), &[])
            .contains("oversized expanded metadata cells")
    );
}

#[test]
fn workspace_preflight_skips_non_xml_parts_and_counts_unreferenced_cells() {
    let bytes = zip_bytes(&[
        ("docProps/thumbnail.jpeg", b"jpeg"),
        (
            "xl/worksheets/sheet1.xml",
            b"<worksheet><row><c><v>1</v></c><c><v>2</v></c></row></worksheet>",
        ),
    ]);
    let xml = metadata_xml_bytes(&bytes).unwrap();
    assert_eq!(
        xml,
        "<worksheet><row><c><v>1</v></c><c><v>2</v></c></row></worksheet>".len() as u64
    );
    // Two implicit columns: XML expansion plus two sparse cells.
    assert_eq!(
        metadata_workspace_bytes(&bytes).unwrap(),
        xml * 128 + 2 * 128
    );
}

#[test]
fn parse_metadata_table_requires_a_header_and_skips_unidentified_rows() {
    let error = |table: Vec<Vec<String>>| {
        parse_metadata_table("metadata.xlsx".into(), table)
            .unwrap_err()
            .to_string()
    };
    assert!(error(Vec::new()).contains("metadata workbook sheet is empty"));
    assert!(error(vec![vec!["Group".into()]]).contains("must contain an `Animal ID` column"));
    let parsed = parse_metadata_table(
        "metadata.xlsx".into(),
        vec![
            vec!["Group".into(), "Animal ID".into()],
            vec!["1".into()],
            vec!["2".into(), "  ".into()],
            vec!["3".into(), "mouse".into()],
            vec!["4".into(), "9.0".into()],
        ],
    )
    .unwrap();
    assert_eq!(parsed.rows.keys().copied().collect::<Vec<_>>(), [9]);
    assert_eq!(parse_animal_id(" "), None);
}

#[test]
fn enrich_trials_handles_every_policy_and_identity_source() {
    let metadata = parse_metadata_table(
        "metadata.xlsx".into(),
        vec![
            vec!["Animal ID".into(), "Sex".into()],
            vec!["5".into(), "female".into()],
            vec!["12".into(), "male".into()],
        ],
    )
    .unwrap();
    let mut by_metadata = trial("descriptive");
    by_metadata.animal_metadata = serde_json::json!({"animal": "12"});
    let mut by_context = trial("oft_videos_unnumbered");
    by_context.session_context = serde_json::json!({"animal": 5});
    let anonymous = trial("descriptive");
    let missing = || {
        [
            trial("8_seq0_oft"),
            trial("8_seq1_oft"),
            trial("7_seq0_oft"),
        ]
    };

    let mut rows = vec![by_metadata, by_context, anonymous];
    rows.extend(missing());
    let exclusions =
        enrich_trials_with_metadata(&mut rows, &metadata, MissingMetadataPolicy::AllowBlanks)
            .unwrap();
    assert!(exclusions.is_empty());
    assert_eq!(rows.len(), 6, "blanks keep unmatched and anonymous trials");
    assert_eq!(rows[0].animal_metadata["sex"], "male");
    assert_eq!(rows[1].animal_metadata["sex"], "female");
    assert_eq!(rows[2].animal_metadata, Value::Null);

    let mut rows = missing().to_vec();
    let exclusions =
        enrich_trials_with_metadata(&mut rows, &metadata, MissingMetadataPolicy::ExcludeMissing)
            .unwrap();
    assert!(rows.is_empty());
    assert_eq!(
        exclusions.iter().map(|e| e.animal).collect::<Vec<_>>(),
        [7, 8],
        "each animal is reported once, sorted"
    );

    let error = enrich_trials_with_metadata(
        &mut missing().to_vec(),
        &metadata,
        MissingMetadataPolicy::Fail,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("missing required Animal ID row(s): 7, 8"),
        "{error}"
    );
}

#[test]
fn cell_values_convert_to_json() {
    assert_eq!(cell_to_json(&CellValue::Bool(true)), Value::Bool(true));
    assert_eq!(cell_to_json(&CellValue::Float(0.5)), serde_json::json!(0.5));
    assert_eq!(cell_to_json(&CellValue::Float(f64::NAN)), Value::Null);
    assert_eq!(
        cell_to_json(&CellValue::FloatList(vec![1.5, f64::INFINITY])),
        serde_json::json!([1.5])
    );
}

#[test]
fn source_workbook_rejects_oversized_or_inconsistent_tables_and_nulls_blanks() {
    let dir = metadata_dir("source");
    let error = |xml: &str| {
        read_workbook_source(&workbook_file(&dir, "source.xlsx", xml), "Sheet")
            .unwrap_err()
            .to_string()
    };
    let row = |cells: &[&str]| {
        format!(
            "<row>{}</row>",
            cells
                .iter()
                .map(|c| format!("<c t=\"inlineStr\"><is><t>{c}</t></is></c>"))
                .collect::<String>()
        )
    };
    for (rows, expected) in [
        (vec![row(&["a", "a"])], "duplicate source workbook header"),
        (
            vec![row(&["source_row"])],
            "duplicate source workbook header",
        ),
        (vec![row(&["a"]), row(&["1", "2"])], "row 2 exceeds header"),
        (
            vec![row(&["a", ""]), row(&["1", "2"])],
            "populated unnamed workbook column at row 2",
        ),
    ] {
        let message = error(&format!("<worksheet>{}</worksheet>", rows.concat()));
        assert!(message.contains(expected), "{expected}: {message}");
    }
    let path = workbook_file(
        &dir,
        "blank.xlsx",
        &format!(
            "<worksheet>{}{}</worksheet>",
            row(&["a", "b", ""]),
            row(&["1", "", ""])
        ),
    );
    let source = read_workbook_source(&path, "Sheet").unwrap();
    assert_eq!(source[0]["b"], Value::Null);
    let oversized = dir.join("oversized.xlsx");
    std::fs::write(
        &oversized,
        zip_bytes(&[("xl/padding.bin", &vec![0u8; 64 * 1024 * 1024 + 1])]),
    )
    .unwrap();
    assert!(
        read_workbook_source(&oversized, "Sheet")
            .unwrap_err()
            .to_string()
            .contains("oversized source workbook")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn read_animal_ages_reads_pairs_and_rejects_bad_sheets() {
    let dir = metadata_dir("ages");
    let head = "<row>\
        <c t=\"inlineStr\"><is><t>Animal</t></is></c>\
        <c t=\"inlineStr\"><is><t>Pre</t></is></c>\
        <c t=\"inlineStr\"><is><t>Post</t></is></c></row>";
    let animal = |id: &str, pre: &str, post: &str| {
        format!(
            "<row><c><v>{id}</v></c><c><v>{pre}</v></c>\
             <c t=\"inlineStr\"><is><t>{post}</t></is></c></row>"
        )
    };
    let sheet = |rows: &str| format!("<worksheet><sheetData>{rows}</sheetData></worksheet>");
    let ages = |name: &str, xml: String, post: &str| {
        let path = workbook_file(&dir, name, &xml);
        read_animal_ages(&path, "Sheet", "Animal", "Pre", post).map_err(|e| e.to_string())
    };
    let ok = ages(
        "ok.xlsx",
        sheet(&format!(
            "{head}{}{}",
            animal("1", "30", "x"),
            animal("2", "31", "40")
        )),
        "Post",
    )
    .unwrap();
    assert_eq!(ok[&1], (Some(30.0), None));
    assert_eq!(ok[&2], (Some(31.0), Some(40.0)));
    assert!(
        ages("empty.xlsx", sheet(""), "Post")
            .unwrap_err()
            .contains("is empty")
    );
    assert!(
        ages("nocol.xlsx", sheet(head), "Missing")
            .unwrap_err()
            .contains("has no `Missing` column")
    );
    let duplicate = sheet(&format!(
        "{head}{}{}",
        animal("1", "30", "40"),
        animal("1", "31", "41")
    ));
    assert!(
        ages("dup.xlsx", duplicate, "Post")
            .unwrap_err()
            .contains("duplicate `Animal` 1")
    );
    std::fs::remove_dir_all(dir).unwrap();
}
