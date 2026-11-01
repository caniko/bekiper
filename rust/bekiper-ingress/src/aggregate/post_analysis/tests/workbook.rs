#[test]
fn cohort_export_enriches_all_discrepant_animals_from_configured_workbook() {
    use std::io::Write;
    let root = std::env::temp_dir().join(format!(
        "bekiper_native_biology_export_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let updated = [
        (16, 7, "Het", "wt"),
        (19, 2, "KO", "tg"),
        (21, 1, "KO", "tg"),
        (37, 4, "Het/KO", "wt"),
        (46, 5, "Het", "tg"),
        (67, 8, "Het", "wt"),
        (70, 6, "Het", "tg"),
    ];
    let mut zip = zip::ZipWriter::new(std::fs::File::create(root.join("animals.xlsx")).unwrap());
    for (path, content) in [
        (
            "xl/workbook.xml",
            r#"<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Updated_Mice-overview" r:id="rId1"/></sheets></workbook>"#,
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#,
        ),
    ] {
        zip.start_file(path, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(content.as_bytes()).unwrap();
    }
    zip.start_file(
        "xl/worksheets/sheet1.xml",
        zip::write::SimpleFileOptions::default(),
    )
    .unwrap();
    let mut worksheet = String::from("<worksheet><sheetData>");
    let table = std::iter::once(vec![
        "Animal ID".into(),
        "Group".into(),
        "HCAR1".into(),
        "5XFAD".into(),
        "Sex".into(),
        "Treatment".into(),
    ])
    .chain(updated.iter().map(|(animal, group, hcar1, xfad)| {
        vec![
            animal.to_string(),
            group.to_string(),
            (*hcar1).into(),
            (*xfad).into(),
            "F".into(),
            "vehicle".into(),
        ]
    }));
    for (index, cells) in table.enumerate() {
        worksheet.push_str(&format!("<row r=\"{}\">", index + 1));
        for (column, cell) in cells.iter().enumerate() {
            worksheet.push_str(&format!(
                "<c r=\"{}{}\" t=\"inlineStr\"><is><t>{cell}</t></is></c>",
                char::from(b'A' + column as u8),
                index + 1
            ));
        }
        worksheet.push_str("</row>");
    }
    worksheet.push_str("</sheetData></worksheet>");
    zip.write_all(worksheet.as_bytes()).unwrap();
    zip.finish().unwrap();
    let mut sessions = Vec::new();
    let mut trials = Vec::new();
    for (animal, group, hcar1, xfad) in updated {
        let mut meta = metadata(&animal.to_string(), "session");
        meta["session_id"] = format!("registered-{animal}").into();
        meta["group"] = format!("G{group}").into();
        meta["hcar1"] = hcar1.into();
        meta["five_xfad"] = xfad.into();
        meta["treatment"] = "vehicle".into();
        let mut trial = oft_trial();
        trial.experiment = format!("oft_videos_oft_post_post_video_{animal}");
        trial.animal_metadata =
            json!({"animal":animal,"group":99,"hcar1":"legacy","five_xfad":"legacy"});
        trial.session_context = json!({"animal":animal.to_string(),"session_id":meta["session_id"],"timepoint":"post","phase":"session", "group":"legacy"});
        sessions.push(meta);
        trials.push(trial);
    }
    sessions.push(metadata("99", "session"));
    std::fs::write(
        root.join("sessions.json"),
        serde_json::to_vec(&sessions).unwrap(),
    )
    .unwrap();
    let mut plan = json!({"run_mode":"synthetic","synthetic_fixture":true,
        "inputs":{"metadata_path":"sessions.json","units_and_calibration":{"oft":{"time":"s","distance":"m","speed":"m/s"}}},
        "outputs":{"requested_outputs":["metrics"]}});
    plan["inputs"]["session_key_mapping"] = json!({"group":
        (1..=8).map(|group| (group.to_string(),json!(format!("G{group}")))).collect::<serde_json::Map<_,_>>()});
    std::fs::write(root.join("plan.json"), serde_json::to_vec(&plan).unwrap()).unwrap();
    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "synthetic"
        data_directory = "."
        [aggregate.metadata]
        path = "animals.xlsx"
        missing_policy = "exclude_missing"
    "#,
    )
    .unwrap();
    config.project.data_directory = root.clone();
    config.post_analysis.enabled = true;
    config.post_analysis.plan_path = Some(root.join("plan.json"));
    config.post_analysis.run_id = Some("biology-regression".into());
    config.post_analysis.figures.enabled = false;
    config.post_analysis.report.enabled = false;
    let run_root = root.join("run");
    let manifest = export_post_analysis_trials(&config, &run_root, &trials)
        .unwrap()
        .unwrap();
    let output = post_analysis_output_path(&run_root, "biology-regression").unwrap();
    let rows = inputs::load_json(&bekiper_post_analysis::export::path(
        &output,
        "metrics.json",
    ))
    .unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 8);
    for (animal, group, hcar1, xfad) in updated {
        let animal_key = animal.to_string();
        let row = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["animal_id"].as_str() == Some(animal_key.as_str()))
            .unwrap();
        assert_eq!(row["group"], format!("G{group}"));
        assert_eq!(row["hcar1"], hcar1);
        assert_eq!(row["five_xfad"], xfad);
        assert_eq!(row["native_session_context"]["group"], "legacy");
    }
    assert_eq!(
        rows.as_array().unwrap().last().unwrap()["measurement_available"],
        false
    );
    assert!(
        manifest["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["role"] == "aggregate_animal_metadata"
                && item["sheet"] == "Updated_Mice-overview")
    );
    bekiper_post_analysis::pipeline::verify_artifacts(&output).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
