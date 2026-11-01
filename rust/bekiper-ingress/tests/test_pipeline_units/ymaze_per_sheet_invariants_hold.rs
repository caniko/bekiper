#[test]
fn ymaze_per_sheet_invariants_hold() {
    let root = unique_pipeline_dir_per_sheet("ymaze_per_sheet_invariants_hold");
    let video_dir = root.join("videos");
    std::fs::create_dir_all(&video_dir).unwrap();
    std::fs::write(video_dir.join("mouse-01.mp4"), b"").unwrap();
    std::fs::write(video_dir.join("mouse-02.mp4"), b"").unwrap();

    let config_path = root.join("bekiper_config.toml");
    std::fs::write(
        &config_path,
        r#"
        [project]
        project_name = "variant_e2e"
        data_directory = "."

        [aggregate.recipe_to_sheet]
        ymaze__proximal65 = "ymaze_proximal65"
        ymaze__proximal45 = "ymaze_proximal45"
        ymaze__proximal25 = "ymaze_proximal25"

        [[scan]]
        video_dir = "videos"
        task_type = "y_maze"
        recipe = "ymaze"
        fps = 30.0
        meters_per_pixel = 0.001

        [[scan.variants]]
        name = "proximal65"

        [scan.variants.ymaze]
        proximal_threshold = 0.65
        start_mode = "video_start"

        [[scan.variants]]
        name = "proximal45"

        [scan.variants.ymaze]
        proximal_threshold = 0.45
        start_mode = "first_detection"

        [[scan.variants]]
        name = "proximal25"

        [scan.variants.ymaze]
        proximal_threshold = 0.25
        start_mode = "first_detection"
        "#,
    )
    .unwrap();

    let mut config = bekiper_ingress::config::IngressConfig::from_file(&config_path).unwrap();
    config
        .expand(&root, &[bekiper_ingress::stage::Stage::Aggregate])
        .unwrap();

    assert_eq!(config.experiments.len(), 6);
    for experiment in &config.experiments {
        write_variant_ymaze_result_per_sheet(&root, experiment);
    }

    bekiper_ingress::aggregate::run_aggregate(&config, &root).unwrap();

    let workbook = read_ods_workbook_per_sheet(&root.join("results.ods"));
    let ymaze_sheets = ["ymaze_proximal65", "ymaze_proximal45", "ymaze_proximal25"];
    assert_eq!(
        workbook
            .keys()
            .filter(|name| name.starts_with("ymaze_proximal"))
            .count(),
        ymaze_sheets.len()
    );

    for sheet_name in ymaze_sheets {
        let rows = workbook.get(sheet_name).expect("ymaze sheet");
        assert!(rows.len() > 1, "{sheet_name} should contain data rows");
        assert_ymaze_sheet_invariants_per_sheet(rows, sheet_name);
    }

    std::fs::remove_dir_all(&root).ok();
}

fn unique_pipeline_dir_per_sheet(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("bekiper_test_pipeline_{name}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_variant_ymaze_result_per_sheet(
    root: &std::path::Path,
    experiment: &bekiper_ingress::config::ExperimentConfig,
) {
    let variant = experiment.name.rsplit_once("__").unwrap().1;
    let ymaze = experiment
        .task_params
        .as_ref()
        .and_then(|params| params.ymaze.as_ref())
        .unwrap();
    let start_mode = match ymaze.start_mode.unwrap() {
        bekiper_behaviour::radial_arm::YMazeStartMode::VideoStart => "video_start",
        bekiper_behaviour::radial_arm::YMazeStartMode::FirstDetection => "first_detection",
    };
    write_ymaze_result_per_sheet(
        &root.join("data").join(&experiment.name).join("result.toml"),
        &experiment.name,
        Some(&format!("ymaze__{variant}")),
        ymaze.proximal_threshold,
        Some(start_mode),
    );
}

fn write_ymaze_result_per_sheet(
    path: &std::path::Path,
    experiment: &str,
    analysis_role: Option<&str>,
    proximal_threshold: Option<f64>,
    start_mode: Option<&str>,
) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let analysis_role = analysis_role
        .map(|role| format!("analysis_role = \"{role}\"\n"))
        .unwrap_or_default();
    let proximal_threshold = proximal_threshold
        .map(|value| format!("proximal_threshold = {value}\n"))
        .unwrap_or_default();
    let start_mode = start_mode
        .map(|value| format!("start_mode = \"{value}\"\n"))
        .unwrap_or_default();

    std::fs::write(
        path,
        format!(
            r#"
            experiment = "{experiment}"
            {analysis_role}
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "y_maze"

            [animal_metadata]
            animal = 42

            [detailed]
            {proximal_threshold}
            {start_mode}
            arm_membership_hysteresis = true
            seconds_in_arms = 20.0
            seconds_in_arms_and_center = 24.0
            freezing_seconds = 0.0
            total_distance_meters = 2.0
            total_duration_seconds = 120.0
            analysis_duration_seconds = 120.0
            total_mean_speed = 0.02
            area_motion_summaries = []

            [detailed.analysis]
            arm_entries = [1, 1, 1]
            arm_preference = 0.5
            arm_sequence = ["A", "B", "C"]
            raw_arm_sequence = ["A", "B", "C"]
            center_entries = 1
            correct_alternations = 1
            first_arm_entered = "A"
            incorrect_alternations = 0
            last_arm_entered = "C"
            neutral_triads = 0
            spontaneous_alternation_pct = 100.0
            sum_of_permutation_distribution = 1
            zone_sequence = ["A", "center", "B", "center", "C"]
            incorrect_triad_distribution = {{ abc = 0 }}
            neutral_triad_distribution = {{}}

            [detailed.analysis.permutation_distribution]
            abc = 1
            acb = 0
            bac = 0
            bca = 0
            cab = 0
            cba = 0

            [[detailed.zone_summaries]]
            zone_name = "in_arm_A"
            entries = 1
            duration_seconds = 12.0
            distance_meters = 1.0
            mean_speed = 0.1

            [[detailed.zone_summaries]]
            zone_name = "in_arm_B"
            entries = 1
            duration_seconds = 8.0
            distance_meters = 0.8
            mean_speed = 0.1

            [[detailed.zone_summaries]]
            zone_name = "in_arm_C"
            entries = 1
            duration_seconds = 10.0
            distance_meters = 0.9
            mean_speed = 0.1

            [[detailed.zone_summaries]]
            zone_name = "in_center"
            entries = 1
            duration_seconds = 4.0
            distance_meters = 0.2
            mean_speed = 0.05
            "#,
        ),
    )
    .unwrap();
}

fn read_ods_workbook_per_sheet(
    path: &std::path::Path,
) -> std::collections::BTreeMap<String, Vec<Vec<String>>> {
    let file = std::fs::File::open(path).unwrap();
    let mut zip = ::zip::ZipArchive::new(file).unwrap();
    let mut content = String::new();
    std::io::Read::read_to_string(&mut zip.by_name("content.xml").unwrap(), &mut content).unwrap();
    let doc = roxmltree::Document::parse(&content).unwrap();

    doc.descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "table")
        .map(|sheet| {
            let name = sheet
                .attributes()
                .find(|attr| attr.name() == "name")
                .unwrap()
                .value()
                .to_string();
            let rows = sheet
                .children()
                .filter(|node| node.is_element() && node.tag_name().name() == "table-row")
                .map(|row| {
                    row.children()
                        .filter(|node| node.is_element() && node.tag_name().name() == "table-cell")
                        .map(cell_text_per_sheet)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (name, rows)
        })
        .collect()
}

fn cell_text_per_sheet(cell: roxmltree::Node<'_, '_>) -> String {
    if let Some(value) = cell
        .attributes()
        .find(|attr| attr.name() == "value")
        .map(|attr| attr.value().to_string())
    {
        return value;
    }
    cell.descendants()
        .filter(|node| node.is_text())
        .filter_map(|node| node.text())
        .collect::<String>()
}

fn assert_ymaze_sheet_invariants_per_sheet(rows: &[Vec<String>], sheet_name: &str) {
    let header = &rows[0];
    let valid_idx = column_index_per_sheet(header, "valid_total_arm_entries");
    let triads_idx = column_index_per_sheet(header, "total_triads");
    let correct_idx = column_index_per_sheet(header, "correct_alternations");
    let standard_idx = column_index_per_sheet(header, "spontaneous_alternation_pct_standard");

    for (row_index, row) in rows.iter().enumerate().skip(1) {
        let valid = parse_i64_per_sheet(
            &row[valid_idx],
            sheet_name,
            row_index,
            "valid_total_arm_entries",
        );
        let triads = parse_i64_per_sheet(&row[triads_idx], sheet_name, row_index, "total_triads");
        let correct = parse_i64_per_sheet(
            &row[correct_idx],
            sheet_name,
            row_index,
            "correct_alternations",
        );
        let standard = parse_f64_per_sheet(
            &row[standard_idx],
            sheet_name,
            row_index,
            "spontaneous_alternation_pct_standard",
        );

        assert_eq!(
            triads,
            std::cmp::max(0, valid - 2),
            "{sheet_name} row {row_index}: total_triads mismatch"
        );
        if triads > 0 {
            let expected = correct as f64 / triads as f64 * 100.0;
            assert!(
                (standard - expected).abs() <= 1e-6,
                "{sheet_name} row {row_index}: spontaneous_alternation_pct_standard mismatch"
            );
        }
    }
}

fn parse_i64_per_sheet(value: &str, sheet_name: &str, row_index: usize, column: &str) -> i64 {
    value.parse().unwrap_or_else(|error| {
        panic!(
            "{sheet_name} row {row_index}: could not parse {column} as i64: {error:?} (value={value:?})"
        )
    })
}

fn parse_f64_per_sheet(value: &str, sheet_name: &str, row_index: usize, column: &str) -> f64 {
    value.parse().unwrap_or_else(|error| {
        panic!(
            "{sheet_name} row {row_index}: could not parse {column} as f64: {error:?} (value={value:?})"
        )
    })
}

fn column_index_per_sheet(header: &[String], column: &str) -> usize {
    header
        .iter()
        .position(|name| name == column)
        .unwrap_or_else(|| panic!("missing column {column} in {header:?}"))
}
