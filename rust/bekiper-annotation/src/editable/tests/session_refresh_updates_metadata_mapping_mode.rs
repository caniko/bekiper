#[test]
fn session_refresh_updates_metadata_mapping_mode() {
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
    assert!(session.is_rereferenced());

    session
        .metadata
        .as_mut()
        .unwrap()
        .set(0, "ChangeReferenceImageName", String::new())
        .unwrap();
    session.refresh().unwrap();

    assert!(!session.is_rereferenced());
    assert!(session.current_target_image.is_none());
    assert!(matches!(
        session.mapping,
        SessionMapping::Metadata {
            source: PerimeterSource::Direct { .. },
            ..
        }
    ));
}

#[test]
fn session_updates_current_target_reference_row_only() {
    let dir = make_temp_dir();
    let plugin_dir = dir.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    let maze_dir = plugin_dir.join("radial-maze_maze_1");
    std::fs::create_dir_all(&maze_dir).unwrap();
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
    std::fs::write(
        plugin_dir.join("reference-maze_1.csv"),
        "reference,467,103,a_p1_1_before_1_phd.png,640,480\n",
    )
    .unwrap();
    std::fs::write(
            plugin_dir.join("change_reference-maze_1.csv"),
            "reference,465,106,b_p1_1_before_1.png,640,480\nreference,468,104,a_p1_2_before_32_phd.png,640,480\n",
        )
        .unwrap();
    let mut file = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(
        file,
        "\u{feff}Part,Number,ChangeReference,ChangeReferenceImageName"
    )
    .unwrap();
    writeln!(file, "1,1,maze_1,b_p1_1_before_1").unwrap();

    let video_path = PathBuf::from("pre.video-1.mp4");
    let mut session = AnnotationSession::load(&dir, &video_path).unwrap();
    session
        .set_current_target_reference_point(500.0, 500.0)
        .unwrap();
    session.save().unwrap();

    let doc = ReferenceDocument::load(&plugin_dir.join("change_reference-maze_1.csv")).unwrap();
    let updated = doc
        .as_reference_point(doc.find_row_by_stem("b_p1_1_before_1").unwrap())
        .unwrap();
    let untouched = doc
        .as_reference_point(doc.find_row_by_stem("a_p1_2_before_32_phd").unwrap())
        .unwrap();
    assert_eq!((updated.x, updated.y), (500.0, 500.0));
    assert_eq!((untouched.x, untouched.y), (468.0, 104.0));
}

#[test]
fn session_helpers_surface_mapping_reference_and_arm_label_state() {
    let dir = make_temp_dir();
    let plugin_dir = dir.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    let maze_dir = plugin_dir.join("radial-maze_maze_1");
    std::fs::create_dir_all(&maze_dir).unwrap();
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
    std::fs::write(
        plugin_dir.join("reference-maze_1.csv"),
        "reference,467,103,a_p1_1_before_1_phd.png,640,480\n",
    )
    .unwrap();
    std::fs::write(
        plugin_dir.join("change_reference-maze_1.csv"),
        "reference,465,106,b_p1_1_before_1.png,640,480\n",
    )
    .unwrap();
    let mut file = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(
        file,
        "\u{feff}Part,Number,ChangeReference,ChangeReferenceImageName"
    )
    .unwrap();
    writeln!(file, "1,1,maze_1,b_p1_1_before_1").unwrap();

    let session = AnnotationSession::load(&dir, Path::new("pre.video-1.mp4")).unwrap();
    assert!(session.is_metadata_backed());
    assert!(session.metadata_has_column("ChangeReference"));
    assert!(!session.metadata_has_column("Perimeter"));
    assert_eq!(
        session.perimeter_selection_source(),
        PerimeterSelectionSource::MetadataChangeReference
    );
    assert!(session.has_source_reference_row());
    assert_eq!(
        session.target_reference_state(),
        TargetReferenceState::SavedRow
    );
    assert_eq!(session.ymaze_arm_labels(), vec!["a", "b", "c"]);
}

#[test]
fn session_helpers_report_fallback_target_reference_state() {
    let dir = make_temp_dir();
    let plugin_dir = dir.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    let maze_dir = plugin_dir.join("radial-maze_maze_1");
    std::fs::create_dir_all(&maze_dir).unwrap();
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
    std::fs::write(
        plugin_dir.join("reference-maze_1.csv"),
        "reference,467,103,a_p1_1_before_1_phd.png,640,480\n",
    )
    .unwrap();
    std::fs::write(
        plugin_dir.join("change_reference-maze_1.csv"),
        "reference,468,104,a_p1_2_before_32_phd.png,640,480\n",
    )
    .unwrap();
    let mut file = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(file, "Part,Number,ChangeReference,ChangeReferenceImageName").unwrap();
    writeln!(file, "1,1,maze_1,b_p1_1_before_1").unwrap();

    let session = AnnotationSession::load(&dir, Path::new("pre.video-1.mp4")).unwrap();
    assert_eq!(
        session.target_reference_state(),
        TargetReferenceState::FallbackPreview
    );
}

#[test]
fn resolved_object_labels_preserve_spec_order() {
    let dir = make_temp_dir();
    let plugin_dir = dir.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    std::fs::write(
            plugin_dir.join("perimeter-rectangle-A1_95.csv"),
            "label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height\nfamiliar,48,49,25,38,A1_training_1.png,232,216\nnovel,152,150,28,27,A1_training_1.png,232,216\n",
        )
        .unwrap();
    let mut metadata = std::fs::File::create(dir.join("metadata.csv")).unwrap();
    writeln!(
            metadata,
            "Phase,PhasePart,TrialNumber,Stage,Animal,Perimeter,ChangeReference,ChangeReferenceImageName"
        )
        .unwrap();
    writeln!(metadata, "A,2,95,1,1,A1_95,,").unwrap();

    let session = AnnotationSession::load(&dir, Path::new("post.T1.configA.video-1.mp4")).unwrap();
    assert_eq!(session.resolved_object_labels(), vec!["familiar", "novel"]);
    assert_eq!(
        session.perimeter_selection_source(),
        PerimeterSelectionSource::MetadataPerimeter
    );
    assert_eq!(
        session.target_reference_state(),
        TargetReferenceState::Unavailable
    );
}
