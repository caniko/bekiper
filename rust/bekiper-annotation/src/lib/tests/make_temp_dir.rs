fn make_temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("bekiper_ann_test_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assert_has_point(points: &[(f64, f64)], expected: (f64, f64)) {
    assert!(
        points
            .iter()
            .any(|&(x, y)| { (x - expected.0).abs() < 0.01 && (y - expected.1).abs() < 0.01 }),
        "expected point {expected:?} in {points:?}"
    );
}

#[test]
fn parse_rectangle_csv_basic() {
    let dir = make_temp_dir();
    let csv_path = &dir.join("test.csv");
    std::fs::write(
        csv_path,
        "familiar,48,49,25,38,image.png,224,224\nnovel,152,150,28,27,image.png,224,224\n",
    )
    .unwrap();

    let specs = parse_rectangle_csv(csv_path).unwrap();
    assert_eq!(specs.len(), 2);

    match &specs[0] {
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } => {
            assert_eq!(label.as_str(), "familiar");
            assert!((center_x - 60.5).abs() < 0.01); // 48 + 25/2
            assert!((center_y - 68.0).abs() < 0.01); // 49 + 38/2
            assert!((width - 25.0).abs() < 0.01);
            assert!((height - 38.0).abs() < 0.01);
        }
        other => panic!("expected Rectangle, got {:?}", other),
    }
}

#[test]
fn parse_coco_json_basic() {
    let dir = make_temp_dir();
    let json_path = &dir.join("test.json");
    std::fs::write(
        json_path,
        r#"{
                "annotations": [
                    {
                        "category_id": 1,
                        "segmentation": [[30.0, 40.0, 10.0, 40.0, 10.0, 20.0, 30.0, 20.0]],
                        "bbox": [10.0, 20.0, 20.0, 20.0]
                    }
                ],
                "categories": [{"id": 1, "name": "familiar"}]
            }"#,
    )
    .unwrap();

    let specs = parse_coco_json(json_path).unwrap();
    assert_eq!(specs.len(), 1);

    match &specs[0] {
        PerimeterSpec::Polygon { label, vertices } => {
            assert_eq!(label.as_str(), "familiar");
            assert_eq!(vertices.len(), 4);
            assert_eq!(
                vertices,
                &vec![(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
            );
        }
        other => panic!("expected Polygon, got {:?}", other),
    }
}

#[test]
fn extract_video_number_patterns() {
    assert_eq!(
        extract_video_number(Path::new("post.T1.configA.video-10.mp4")),
        Some(10)
    );
    assert_eq!(extract_video_number(Path::new("video-42.mp4")), Some(42));
    assert_eq!(extract_video_number(Path::new("Test 85.szv")), Some(85));
    assert_eq!(
        extract_video_number(Path::new("pre.T2.configB.video-7.mp4")),
        Some(7)
    );
}

#[test]
fn extract_video_context_patterns() {
    let ctx = extract_video_context(Path::new("post.T1.configA.video-10.mp4")).unwrap();
    assert_eq!(ctx.animal, 10);
    assert_eq!(ctx.stage, 1);
    assert_eq!(ctx.phase_part, 2); // post = 2

    let ctx = extract_video_context(Path::new("pre.T2.configB.video-7.mp4")).unwrap();
    assert_eq!(ctx.animal, 7);
    assert_eq!(ctx.stage, 2);
    assert_eq!(ctx.phase_part, 1); // pre = 1

    // Path-based extraction
    let ctx = extract_video_context(Path::new(
        "NORT (post)/T1/Config A/post.T1.configA.video-3.mp4",
    ))
    .unwrap();
    assert_eq!(ctx.animal, 3);
    assert_eq!(ctx.stage, 1);
    assert_eq!(ctx.phase_part, 2);

    // Y-maze: no T1/T2 → stage=0
    let ctx = extract_video_context(Path::new("pre.video-18.mp4")).unwrap();
    assert_eq!(ctx.animal, 18);
    assert_eq!(ctx.stage, 0);
    assert_eq!(ctx.phase_part, 1);

    let ctx = extract_video_context(Path::new("Y-maze (post)/post.video-5.mp4")).unwrap();
    assert_eq!(ctx.animal, 5);
    assert_eq!(ctx.stage, 0);
    assert_eq!(ctx.phase_part, 2);
}

#[test]
fn load_perimeter_by_id_csv() {
    let dir = make_temp_dir();
    let plugin_dir = &dir.join("plugin_files");
    std::fs::create_dir(plugin_dir).unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-rectangle-A1_100.csv"),
        "familiar,48,49,25,38,img.png,224,224\nnovel,152,150,28,27,img.png,224,224\n",
    )
    .unwrap();

    let specs = load_perimeter_by_id(plugin_dir, "A1_100").unwrap();
    assert_eq!(specs.len(), 2);
}

#[test]
fn video_map_csv_roundtrip() {
    let dir = make_temp_dir();
    let map_path = &dir.join("video_map.csv");
    let mut f = std::fs::File::create(map_path).unwrap();
    writeln!(f, "video_number,perimeter_id").unwrap();
    writeln!(f, "10,A1_100").unwrap();
    writeln!(f, "42,novel_2").unwrap();

    let map = read_video_map(map_path).unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map[&10], "A1_100");
    assert_eq!(map[&42], "novel_2");
}

#[test]
fn annotation_resolver_with_video_map() {
    let dir = make_temp_dir();
    let plugin_dir = &dir.join("plugin_files");
    std::fs::create_dir(plugin_dir).unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-rectangle-A1_100.csv"),
        "familiar,48,49,25,38,img.png,224,224\nnovel,152,150,28,27,img.png,224,224\n",
    )
    .unwrap();

    let mut f = std::fs::File::create(dir.join("video_map.csv")).unwrap();
    writeln!(f, "video_number,perimeter_id").unwrap();
    writeln!(f, "10,A1_100").unwrap();

    let resolver = AnnotationResolver::from_dir(&dir).unwrap();
    let ctx = VideoContext {
        animal: 10,
        stage: 1,
        phase_part: 2,
    };
    let specs = resolver.resolve(&ctx).unwrap();
    assert!(specs.is_some());
    assert_eq!(specs.unwrap().len(), 2);

    let missing_ctx = VideoContext {
        animal: 999,
        stage: 1,
        phase_part: 1,
    };
    let missing = resolver.resolve(&missing_ctx).unwrap();
    assert!(missing.is_none());
}

#[test]
fn annotation_resolver_with_metadata_direct() {
    let dir = make_temp_dir();
    let plugin_dir = &dir.join("plugin_files");
    std::fs::create_dir(plugin_dir).unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-rectangle-training_1.csv"),
        "familiar,48,49,25,38,img.png,224,224\nnovel,152,150,28,27,img.png,224,224\n",
    )
    .unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-rectangle-novel_2.csv"),
        "familiar,50,50,20,20,img.png,224,224\nnovel,150,150,30,30,img.png,224,224\n",
    )
    .unwrap();

    let mut f = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(f, "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName").unwrap();
    writeln!(f, "A,1,47,1,1,1,training_1,,").unwrap();
    writeln!(f, ",2,93,2,1,1,novel_2,,").unwrap();

    let resolver = AnnotationResolver::from_dir(&dir).unwrap();

    // Direct perimeter: Animal 1, Stage 1, PhasePart 1
    let ctx = VideoContext {
        animal: 1,
        stage: 1,
        phase_part: 1,
    };
    let specs = resolver.resolve(&ctx).unwrap().unwrap();
    assert_eq!(specs.len(), 2);

    // Direct perimeter: Animal 1, Stage 2, PhasePart 2
    let ctx = VideoContext {
        animal: 1,
        stage: 2,
        phase_part: 2,
    };
    assert!(resolver.resolve(&ctx).unwrap().is_some());

    // Not in metadata
    let ctx = VideoContext {
        animal: 1,
        stage: 1,
        phase_part: 2,
    };
    assert!(resolver.resolve(&ctx).unwrap().is_none());
}

#[test]
fn annotation_resolver_re_referencing() {
    let dir = make_temp_dir();
    let plugin_dir = &dir.join("plugin_files");
    std::fs::create_dir(plugin_dir).unwrap();

    // Perimeter annotated on A2 image: familiar at center (60.5, 68), novel at (166, 163.5)
    std::fs::write(
            plugin_dir.join("perimeter-rectangle-training_1.csv"),
            "familiar,48,49,25,38,A2_training_1.png,232,216\nnovel,152,150,28,27,A2_training_1.png,232,216\n",
        )
        .unwrap();

    // Source reference point (on A2 image)
    std::fs::write(
        plugin_dir.join("reference-training_1.csv"),
        "reference,100,107,A2_training_1.png,232,216\n",
    )
    .unwrap();

    // Target reference points for A1 and B1
    std::fs::write(
            plugin_dir.join("change_reference-training_1.csv"),
            "reference,100,109,A1_training_1.png,232,216\nreference,100,106,B1_training_1.png,232,216\n",
        )
        .unwrap();

    let mut f = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(f, "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName").unwrap();
    // Animal 1 uses ChangeReference → needs re-referencing to A1 image
    writeln!(f, "A,1,47,1,1,1,,training_1,A1_training_1").unwrap();
    // Animal 2 uses direct Perimeter → no translation
    writeln!(f, ",,48,1,2,2,training_1,,").unwrap();
    // Animal 3: Perimeter column has ID + ChangeReferenceImageName → also needs re-referencing
    writeln!(f, ",,49,1,3,3,training_1,,B1_training_1").unwrap();

    let resolver = AnnotationResolver::from_dir(&dir).unwrap();

    // Animal 1: re-referenced from A2 → A1, offset = (100-100, 109-107) = (0, 2)
    let ctx = VideoContext {
        animal: 1,
        stage: 1,
        phase_part: 1,
    };
    let specs = resolver.resolve(&ctx).unwrap().unwrap();
    assert_eq!(specs.len(), 2);
    match &specs[0] {
        PerimeterSpec::Rectangle {
            center_x, center_y, ..
        } => {
            // Original center: (48 + 25/2, 49 + 38/2) = (60.5, 68.0)
            // After translation (+0, +2): (60.5, 70.0)
            assert!((center_x - 60.5).abs() < 0.01);
            assert!((center_y - 70.0).abs() < 0.01);
        }
        other => panic!("expected Rectangle, got {:?}", other),
    }

    // Animal 2: direct, no translation
    let ctx = VideoContext {
        animal: 2,
        stage: 1,
        phase_part: 1,
    };
    let specs = resolver.resolve(&ctx).unwrap().unwrap();
    match &specs[0] {
        PerimeterSpec::Rectangle { center_y, .. } => {
            assert!((center_y - 68.0).abs() < 0.01); // untranslated
        }
        other => panic!("expected Rectangle, got {:?}", other),
    }

    // Animal 3: Perimeter column has ID + ChangeReferenceImageName
    // should re-reference from A2 → B1, offset = (100-100, 106-107) = (0, -1)
    let ctx = VideoContext {
        animal: 3,
        stage: 1,
        phase_part: 1,
    };
    let specs = resolver.resolve(&ctx).unwrap().unwrap();
    assert_eq!(specs.len(), 2);
    match &specs[0] {
        PerimeterSpec::Rectangle {
            center_x, center_y, ..
        } => {
            // Original center: (60.5, 68.0), after translation (+0, -1): (60.5, 67.0)
            assert!((center_x - 60.5).abs() < 0.01);
            assert!((center_y - 67.0).abs() < 0.01);
        }
        other => panic!("expected Rectangle, got {:?}", other),
    }
}

#[test]
fn ymaze_metadata_format() {
    let dir = make_temp_dir();
    let plugin_dir = &dir.join("plugin_files");
    std::fs::create_dir(plugin_dir).unwrap();

    // Radial maze annotation
    let maze_dir = plugin_dir.join("radial-maze_maze_1");
    std::fs::create_dir(&maze_dir).unwrap();
    std::fs::write(
            maze_dir.join("arm-lines.csv"),
            "a,151,161,173,122,img.png,640,480\nb,462,156,438,119,img.png,640,480\nc,294,397,337,397,img.png,640,480\n",
        )
        .unwrap();
    std::fs::write(
            maze_dir.join("center-triangle.json"),
            r#"{"annotations": [{"category_id": 1, "segmentation": [[288,237,309,196,335,236]], "bbox": [288,196,47,41]}], "categories": [{"id": 1, "name": "center"}]}"#,
        )
        .unwrap();

    // Source + change reference points
    std::fs::write(
        plugin_dir.join("reference-maze_1.csv"),
        "reference,467,103,a_p1_1_before_1_phd.png,640,480\n",
    )
    .unwrap();
    std::fs::write(
            plugin_dir.join("change_reference-maze_1.csv"),
            "reference,468,104,a_p1_2_before_32_phd.png,640,480\nreference,465,106,b_p1_1_before_1.png,640,480\n",
        )
        .unwrap();

    // Y-maze metadata (Part/Number format, no Stage)
    let mut f = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(f, "Part,Number,ChangeReference,ChangeReferenceImageName").unwrap();
    writeln!(f, "1,1,maze_1,b_p1_1_before_1").unwrap();
    writeln!(f, ",3,maze_1,b_p1_1_before_1").unwrap();
    writeln!(f, "2,1,maze_1,a_p1_2_before_32_phd").unwrap();

    let resolver = AnnotationResolver::from_dir(&dir).unwrap();

    // Animal 1, Part 1 → re-referenced maze_1 to b_p1_1_before_1
    // offset = (465-467, 106-103) = (-2, 3)
    let ctx = VideoContext {
        animal: 1,
        stage: 0,
        phase_part: 1,
    };
    let specs = resolver.resolve(&ctx).unwrap().unwrap();
    assert_eq!(specs.len(), 1); // RadialMaze is a single PerimeterSpec
    match &specs[0] {
        PerimeterSpec::RadialMaze { arms, .. } => {
            // First arm distal line translated by (-2, 3)
            assert_eq!(arms[0].len(), 4);
            assert_has_point(&arms[0], (149.0, 164.0));
            assert_has_point(&arms[0], (171.0, 125.0));
        }
        other => panic!("expected RadialMaze, got {:?}", other),
    }

    // Animal 3, Part 1 → also re-referenced, same target image
    let ctx = VideoContext {
        animal: 3,
        stage: 0,
        phase_part: 1,
    };
    assert!(resolver.resolve(&ctx).unwrap().is_some());

    // Animal 1, Part 2 → re-referenced to a_p1_2_before_32_phd
    // offset = (468-467, 104-103) = (1, 1)
    let ctx = VideoContext {
        animal: 1,
        stage: 0,
        phase_part: 2,
    };
    let specs = resolver.resolve(&ctx).unwrap().unwrap();
    match &specs[0] {
        PerimeterSpec::RadialMaze { arms, .. } => {
            // First arm distal line translated by (1, 1)
            assert_has_point(&arms[0], (152.0, 162.0));
            assert_has_point(&arms[0], (174.0, 123.0));
        }
        other => panic!("expected RadialMaze, got {:?}", other),
    }
}
