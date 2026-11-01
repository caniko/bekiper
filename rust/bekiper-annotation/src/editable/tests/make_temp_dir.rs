fn make_temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("bekiper_ann_editable_{}_{n}", std::process::id()));
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
fn rectangle_csv_round_trip_preserves_header() {
    let dir = make_temp_dir();
    let path = dir.join("perimeter-rectangle-test.csv");
    std::fs::write(
            &path,
            "label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height\nfamiliar,48,49,25,38,A2_training_1.png,232,216\nnovel,152,150,28,27,A2_training_1.png,232,216\n",
        )
        .unwrap();

    let mut asset = RectangleCsvAsset::load(&path).unwrap();
    asset
        .replace_from_specs(&[PerimeterSpec::Rectangle {
            label: "familiar".into(),
            center_x: 70.0,
            center_y: 80.0,
            width: 20.0,
            height: 30.0,
        }])
        .unwrap();
    asset.write().unwrap();

    let reloaded = RectangleCsvAsset::load(&path).unwrap();
    assert!(reloaded.header.is_some());
    assert_eq!(reloaded.rows.len(), 1);
    assert_eq!(reloaded.rows[0].bbox_x, 60.0);
    assert_eq!(reloaded.rows[0].bbox_y, 65.0);
}

#[test]
fn circle_csv_round_trip_is_editable() {
    let dir = make_temp_dir();
    let path = dir.join("perimeter-circle-test.csv");
    std::fs::write(
            &path,
            "object_index,x1,y1,x2,y2,image_name,image_width,image_height\n1,50,60,70,60,A1.png,100,100\n",
        )
        .unwrap();

    let mut asset = CircleCsvAsset::load(&path, CircleCsvKind::EdgePoint).unwrap();
    asset
        .replace_from_specs(&[PerimeterSpec::Circle {
            label: "object_1".into(),
            center_x: 80.0,
            center_y: 90.0,
            radius: 12.0,
        }])
        .unwrap();
    asset.write().unwrap();

    let reloaded = CircleCsvAsset::load(&path, CircleCsvKind::EdgePoint).unwrap();
    assert!(reloaded.header.is_some());
    assert_eq!(reloaded.rows.len(), 1);
    assert_eq!(reloaded.rows[0].center_x, 80.0);
    assert_eq!(reloaded.rows[0].center_y, 90.0);
    assert_eq!(reloaded.rows[0].radius, 12.0);
    assert!(crate::parse_circle_csv(&path).is_ok());
}

#[test]
fn coco_json_round_trip_updates_annotations() {
    let dir = make_temp_dir();
    let path = dir.join("perimeter-rectangle-training_1.json");
    std::fs::write(
            &path,
            r#"{"info":{"description":"my-project-name"},"images":[{"id":1,"width":232,"height":216,"file_name":"A2_training_1.png"}],"annotations":[{"id":0,"iscrowd":0,"image_id":1,"category_id":2,"segmentation":[[175.6,71.9,175.6,38.5,160.5,38.5,160.5,72.2]],"bbox":[160.5,38.5,15.1,33.7],"area":509.1}],"categories":[{"id":2,"name":"variable"}]}"#,
        )
        .unwrap();

    let mut asset = CocoJsonAsset::load(&path).unwrap();
    asset
        .replace_from_specs(&[PerimeterSpec::Rectangle {
            label: "variable".into(),
            center_x: 100.0,
            center_y: 100.0,
            width: 20.0,
            height: 10.0,
        }])
        .unwrap();
    asset.write().unwrap();

    let specs = CocoJsonAsset::load(&path).unwrap().to_specs().unwrap();
    match &specs[0] {
        PerimeterSpec::Polygon { vertices, .. } => {
            assert_eq!(vertices, &rectangle_vertices(90.0, 95.0, 20.0, 10.0));
        }
        other => panic!("expected polygon, got {other:?}"),
    }
}

#[test]
fn radial_maze_round_trip_updates_arms_and_center() {
    let dir = make_temp_dir();
    let maze_dir = dir.join("radial-maze_maze_1");
    std::fs::create_dir(&maze_dir).unwrap();
    std::fs::write(
            maze_dir.join("arm-lines.csv"),
            "a,151,161,173,122,a_p1_1_before_1_phd.png,640,480\nb,462,156,438,119,a_p1_1_before_1_phd.png,640,480\nc,294,397,337,397,a_p1_1_before_1_phd.png,640,480\n",
        )
        .unwrap();
    std::fs::write(
            maze_dir.join("center-triangle.json"),
            r#"{"images":[{"id":1,"width":640,"height":480,"file_name":"a_p1_1_before_1_phd.png"}],"annotations":[{"id":0,"iscrowd":0,"image_id":1,"category_id":5,"segmentation":[[288.6,237.3,309.0,196.1,334.8,235.8]],"bbox":[288.6,196.1,46.2,41.2],"area":936.8}],"categories":[{"id":5,"name":"center"}]}"#,
        )
        .unwrap();

    let mut asset = RadialMazeAsset::load(&maze_dir).unwrap();
    asset
        .replace_from_spec(&PerimeterSpec::RadialMaze {
            label: "maze".into(),
            center_vertices: vec![(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)],
            arms: vec![
                vec![(1.0, 1.0), (2.0, 2.0)],
                vec![(3.0, 3.0), (4.0, 4.0)],
                vec![(5.0, 5.0), (6.0, 6.0)],
            ],
        })
        .unwrap();
    asset.write().unwrap();

    let specs = parse_radial_maze(&maze_dir).unwrap();
    match &specs[0] {
        PerimeterSpec::RadialMaze {
            center_vertices,
            arms,
            ..
        } => {
            assert_eq!(center_vertices[0], (10.0, 10.0));
            assert_eq!(arms[0].len(), 4);
            assert_has_point(&arms[0], (1.0, 1.0));
            assert_has_point(&arms[0], (2.0, 2.0));
            assert_has_point(&arms[2], (5.0, 5.0));
            assert_has_point(&arms[2], (6.0, 6.0));
        }
        other => panic!("expected radial maze, got {other:?}"),
    }
}

#[test]
fn reference_document_upserts_rows() {
    let dir = make_temp_dir();
    let path = dir.join("change_reference-training_1.csv");
    std::fs::write(&path, "reference,100,109,A1_training_1.png,232,216\n").unwrap();

    let mut doc = ReferenceDocument::load(&path).unwrap();
    let template = doc.rows.first().cloned();
    let index = doc.upsert_point("B1_training_1", 200.0, 201.0, template.as_ref());
    assert_eq!(index, 1);
    doc.write().unwrap();

    let reloaded = ReferenceDocument::load(&path).unwrap();
    let index = reloaded.find_row_by_stem("B1_training_1").unwrap();
    let point = reloaded.as_reference_point(index).unwrap();
    assert_eq!(point.x, 200.0);
    assert_eq!(point.y, 201.0);
}

#[test]
fn metadata_document_resolves_rereferenced_rows() {
    let dir = make_temp_dir();
    let path = dir.join("metadata.csv");
    let mut file = std::fs::File::create(&path).unwrap();
    writeln!(
            file,
            "Phase,PhasePart,TrialNumber,Stage,Animal,Perimeter,ChangeReference,ChangeReferenceImageName"
        )
        .unwrap();
    writeln!(file, "A,1,47,1,1,,training_1,A1_training_1").unwrap();
    writeln!(file, ",,48,1,2,training_1,,").unwrap();

    let doc = MetadataDocument::load(&path).unwrap();
    let (row, source) = doc
        .resolve_row(&VideoContext {
            animal: 1,
            stage: 1,
            phase_part: 1,
        })
        .unwrap();
    assert_eq!(row, 0);
    match source {
        PerimeterSource::ReReferenced {
            perimeter_id,
            target_image,
        } => {
            assert_eq!(perimeter_id, "training_1");
            assert_eq!(target_image, "A1_training_1");
        }
        other => panic!("expected rereferenced row, got {other:?}"),
    }
}

#[test]
fn metadata_document_normalizes_bom_headers_for_lookup_and_write() {
    let dir = make_temp_dir();
    let path = dir.join("metadata.csv");
    let mut file = std::fs::File::create(&path).unwrap();
    writeln!(
        file,
        "\u{feff}Part,Number,ChangeReference,ChangeReferenceImageName"
    )
    .unwrap();
    writeln!(file, "1,1,maze_1,b_p1_1_before_1").unwrap();

    let mut doc = MetadataDocument::load(&path).unwrap();
    assert!(doc.has_column("Part"));
    assert!(doc.has_column("ChangeReference"));
    assert_eq!(
        doc.get(0, "ChangeReferenceImageName"),
        Some("b_p1_1_before_1")
    );

    doc.set(0, "ChangeReferenceImageName", "b_p1_2_before_24".into())
        .unwrap();
    doc.write().unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.starts_with("Part,Number,ChangeReference,ChangeReferenceImageName"));
    assert!(!written.starts_with('\u{feff}'));
    assert!(written.contains("b_p1_2_before_24"));
}

fn write_imen_nort_trial_number_fixture(root: &Path) {
    let plugin_dir = root.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-rectangle-wrong_48.csv"),
        "familiar,10,20,30,40,A1_wrong.png,224,212\n",
    )
    .unwrap();
    std::fs::write(
            plugin_dir.join("perimeter-rectangle-training_2.csv"),
            "familiar,48,49,25,38,A1_training_2.png,224,212\nnovel,152,150,28,27,A1_training_2.png,224,212\n",
        )
        .unwrap();
    std::fs::write(
        plugin_dir.join("reference-training_2.csv"),
        "reference,100,107,A1_training_2.png,224,212\n",
    )
    .unwrap();
    std::fs::write(
        plugin_dir.join("change_reference-training_2.csv"),
        "reference,104,110,A2_training_2.png,224,212\n",
    )
    .unwrap();
    let mut metadata = std::fs::File::create(root.join("metadata.csv")).unwrap();
    writeln!(
            metadata,
            "\u{feff}Phase,PhasePart,TrialNumber,Stage,Animal,Perimeter,ChangeReference,ChangeReferenceImageName"
        )
        .unwrap();
    writeln!(metadata, "A,1,1,1,48,wrong_48,,").unwrap();
    writeln!(metadata, ",,48,1,6,,training_2,A2_training_2").unwrap();
}

fn imen_nort_source_row_context() -> IngressSessionContext {
    IngressSessionContext {
        profile: "videomap".into(),
        source_row: Some(3),
        animal: Some("6".into()),
        trial: Some("48".into()),
        phase_part: Some("1".into()),
        stage: Some("1".into()),
        ..IngressSessionContext::default()
    }
}

#[test]
fn session_prefers_ingress_source_row_over_video_number_as_animal() {
    let dir = make_temp_dir();
    write_imen_nort_trial_number_fixture(&dir);

    let session = AnnotationSession::load_with_context(
        &dir,
        Path::new("pre.T1.configA.video-48.mp4"),
        Some(imen_nort_source_row_context()),
    )
    .unwrap();

    assert_eq!(session.perimeter_id, "training_2");
    assert_eq!(
        session.current_target_image.as_deref(),
        Some("A2_training_2")
    );
    assert_eq!(
        session.mapping,
        SessionMapping::Metadata {
            row_index: 1,
            source: PerimeterSource::ReReferenced {
                perimeter_id: "training_2".into(),
                target_image: "A2_training_2".into(),
            },
        }
    );
}

#[test]
fn session_still_falls_back_to_video_context_without_source_row() {
    let dir = make_temp_dir();
    write_imen_nort_trial_number_fixture(&dir);

    let session = AnnotationSession::load(&dir, Path::new("pre.T1.configA.video-48.mp4")).unwrap();

    assert_eq!(session.perimeter_id, "wrong_48");
    assert_eq!(
        session.mapping,
        SessionMapping::Metadata {
            row_index: 0,
            source: PerimeterSource::Direct {
                perimeter_id: "wrong_48".into(),
            },
        }
    );
}

#[test]
fn session_source_row_preserves_change_reference_target_image() {
    let dir = make_temp_dir();
    write_imen_nort_trial_number_fixture(&dir);

    let session = AnnotationSession::load_with_context(
        &dir,
        Path::new("pre.T1.configA.video-48.mp4"),
        Some(imen_nort_source_row_context()),
    )
    .unwrap();

    assert!(session.is_rereferenced());
    assert_eq!(session.current_translation(), Some((4.0, 3.0)));
    assert_eq!(
        session.target_reference_state(),
        TargetReferenceState::SavedRow
    );
}

#[test]
fn session_refresh_preserves_ingress_source_row_mapping() {
    let dir = make_temp_dir();
    write_imen_nort_trial_number_fixture(&dir);

    let mut session = AnnotationSession::load_with_context(
        &dir,
        Path::new("pre.T1.configA.video-48.mp4"),
        Some(imen_nort_source_row_context()),
    )
    .unwrap();
    session.refresh().unwrap();

    assert_eq!(session.perimeter_id, "training_2");
    assert!(matches!(
        session.mapping,
        SessionMapping::Metadata {
            row_index: 1,
            source: PerimeterSource::ReReferenced { .. },
        }
    ));
}

#[test]
fn session_inverse_translates_shared_geometry_before_save() {
    let dir = make_temp_dir();
    let plugin_dir = dir.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    std::fs::write(
            plugin_dir.join("perimeter-rectangle-training_1.csv"),
            "familiar,48,49,25,38,A2_training_1.png,232,216\nnovel,152,150,28,27,A2_training_1.png,232,216\n",
        )
        .unwrap();
    std::fs::write(
        plugin_dir.join("reference-training_1.csv"),
        "reference,100,107,A2_training_1.png,232,216\n",
    )
    .unwrap();
    std::fs::write(
        plugin_dir.join("change_reference-training_1.csv"),
        "reference,100,109,A1_training_1.png,232,216\n",
    )
    .unwrap();
    let mut file = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(
            file,
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName"
        )
        .unwrap();
    writeln!(file, "A,2,47,1,1,1,,training_1,A1_training_1").unwrap();

    let video_path = PathBuf::from("post.T1.configA.video-1.mp4");
    let mut session = AnnotationSession::load(&dir, &video_path).unwrap();
    assert_eq!(session.current_translation(), Some((0.0, 2.0)));
    let mut edited = session.resolved_specs.clone();
    if let PerimeterSpec::Rectangle { center_y, .. } = &mut edited[0] {
        *center_y = 80.0;
    }
    session.set_shared_geometry_from_resolved(&edited).unwrap();
    let source_specs = session.asset.to_specs().unwrap();
    match &source_specs[0] {
        PerimeterSpec::Rectangle { center_y, .. } => assert_eq!(*center_y, 78.0),
        other => panic!("expected rectangle, got {other:?}"),
    }
    session.save().unwrap();

    let store = CanonicalAnnotationStore::open(&dir).unwrap().unwrap();
    let specs = store.load_perimeter("training_1").unwrap();
    match &specs[0] {
        PerimeterSpec::Rectangle { center_y, .. } => {
            assert_eq!(*center_y, 78.0);
        }
        other => panic!("expected rectangle, got {other:?}"),
    }
}
