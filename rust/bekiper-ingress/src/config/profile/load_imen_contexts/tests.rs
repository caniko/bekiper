use super::*;

fn temp_annotation_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_imen_context_{label}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_nort_metadata(dir: &Path) {
    std::fs::write(
        dir.join("metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Video_file_name,",
            "Perimeter,ChangeReference,ChangeReferenceImageName\n",
            "A,1,22,0,23,3,Test 22.szv,,,\n",
            ",,48,1,3,3,Test 48.szv,training_3,,A1_training_3\n",
            "A,2,48,1,6,2,Test 48.szv,training_2,,A2_training_2\n",
        ),
    )
    .unwrap();
}

fn write_canonical_session(dir: &Path, name: &str, animal: u32, stage: u32, phase_part: u32) {
    let sessions_dir = dir.join("bekiper/sessions");
    std::fs::create_dir_all(&sessions_dir).unwrap();
    std::fs::write(dir.join("bekiper/manifest.toml"), "").unwrap();
    std::fs::write(
            sessions_dir.join(format!("{name}.toml")),
            format!(
                "key = \"{}\"\nvideo_number = {}\nanimal = {}\nstage = {}\nphase_part = {}\nperimeter_id = \"training_{}\"\ncoordinate_space = \"target\"\n",
                name, animal, animal, stage, phase_part, animal
            ),
        )
        .unwrap();
}

#[test]
fn nort_recipe_lookup_keeps_timepoint_separate_from_training_stage() {
    let dir = temp_annotation_dir("timepoint_and_stage");
    std::fs::write(
        dir.join("metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Video_file_name,Perimeter\n",
            "A,1,101,1,42,Test 101.szv,pre_training\n",
            ",,102,2,42,Test 102.szv,pre_test\n",
            "A,2,201,1,42,Test 201.szv,post_training\n",
            ",,202,2,42,Test 202.szv,post_test\n",
        ),
    )
    .unwrap();
    let contexts = load_imen_contexts(&dir).unwrap();
    for (timepoint, stage, part, row, recipe) in [
        ("pre", 1, "1", 2, DetectionRecipe::NortTraining),
        ("pre", 2, "1", 3, DetectionRecipe::NortNovelty),
        ("post", 1, "2", 4, DetectionRecipe::NortTraining),
        ("post", 2, "2", 5, DetectionRecipe::NortNovelty),
    ] {
        let video = dir.join(format!("{timepoint}.T{stage}.configA.video-42.mp4"));
        let context = contexts.context_for_video_recipe(&video, recipe).unwrap();
        assert_eq!(context.source_row, Some(row), "{}", video.display());
        assert_eq!(context.phase_part.as_deref(), Some(part));
        assert_eq!(context.stage, Some(stage.to_string()));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_animal_stage_timepoint_metadata_is_rejected() {
    let dir = temp_annotation_dir("duplicate_key");
    std::fs::write(
        dir.join("metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal\n",
            "A,1,2,0,1\n",
            "B,1,3,0,1\n",
        ),
    )
    .unwrap();
    assert!(load_imen_contexts(&dir).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nort_video_metadata_lookup_uses_video_number_not_animal_id() {
    let dir = temp_annotation_dir("video_number");
    write_nort_metadata(&dir);
    write_canonical_session(&dir, "animal_3", 3, 1, 1);

    let contexts = load_imen_contexts(&dir).unwrap();
    let video = dir.join("NORT (pre)/T1/Config A/pre.T1.configA.video-48.mp4");
    let context = contexts.context_for_video(&video).unwrap();

    assert_eq!(context.animal.as_deref(), Some("3"));
    assert_eq!(context.stage.as_deref(), Some("1"));
    assert_eq!(context.phase_part.as_deref(), Some("1"));
    assert_eq!(context.source_row, Some(3));

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn nort_post_timepoint_selects_phase_part_two_independently_of_stage() {
    let dir = temp_annotation_dir("t2_phase_part");
    write_nort_metadata(&dir);
    write_canonical_session(&dir, "animal_6", 6, 1, 2);

    let contexts = load_imen_contexts(&dir).unwrap();
    let video = dir.join("NORT (post)/T1/Config A/post.T1.configA.video-48.mp4");
    let context = contexts.context_for_video(&video).unwrap();

    assert_eq!(context.animal.as_deref(), Some("6"));
    assert_eq!(context.stage.as_deref(), Some("1"));
    assert_eq!(context.phase_part.as_deref(), Some("2"));
    assert_eq!(
        metadata_recipe_for_context(Some(&context)),
        Some(Some(DetectionRecipe::NortTraining))
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn nort_recipe_lookup_prefers_direct_renamed_animal_match() {
    let dir = temp_annotation_dir("direct_animal_match");
    std::fs::write(
        dir.join("metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Video_file_name,",
            "Perimeter,ChangeReference,ChangeReferenceImageName\n",
            "A,2,48,1,3,3,Test 48.szv,training_3,,A2_training_3\n",
            ",,90,1,48,4,Test 90.szv,training_4,,A2_training_4\n",
        ),
    )
    .unwrap();
    write_canonical_session(&dir, "animal_48", 48, 1, 2);

    let contexts = load_imen_contexts(&dir).unwrap();
    let video = dir.join("NORT (post)/T1/Config A/post.T1.configA.video-48.mp4");
    let context = contexts
        .context_for_video_recipe(&video, DetectionRecipe::NortTraining)
        .unwrap();

    assert_eq!(context.animal.as_deref(), Some("48"));
    assert_eq!(context.trial.as_deref(), Some("90"));
    assert_eq!(context.stage.as_deref(), Some("1"));
    assert_eq!(context.phase_part.as_deref(), Some("2"));

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn nort_stage_zero_context_is_habituation() {
    let dir = temp_annotation_dir("stage_zero");
    write_nort_metadata(&dir);

    let contexts = load_imen_contexts(&dir).unwrap();
    let video = dir.join("NORT (pre)/T1/Config A/pre.T1.configA.video-22.mp4");
    let context = contexts.context_for_video(&video).unwrap();

    assert_eq!(context.stage.as_deref(), Some("0"));
    assert!(is_imen_habituation_context(Some(&context)));
    assert_eq!(metadata_recipe_for_context(Some(&context)), Some(None));

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn missing_nort_video_metadata_message_reports_exact_key() {
    let dir = temp_annotation_dir("missing_message");
    write_nort_metadata(&dir);

    let video = dir.join("NORT (pre)/T1/Config A/pre.T1.configA.video-999.mp4");
    let message = missing_imen_metadata_context_message(&dir, &video).unwrap();

    assert!(message.contains("video number 999"));
    assert!(message.contains("phase A"));
    assert!(message.contains("phase_part 1"));
    assert!(message.contains("candidate rows for video/phase: none"));

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn recipe_lookup_and_candidate_summaries_cover_every_fallback() {
    let dir = temp_annotation_dir("lookup_fallbacks");
    write_nort_metadata(&dir);
    let contexts = load_imen_contexts(&dir).unwrap();
    let video = dir.join("NORT (pre)/T1/Config A/pre.T1.configA.video-48.mp4");
    assert!(
        contexts
            .context_for_video_recipe(&video, DetectionRecipe::Oft)
            .is_none()
    );
    // Candidate 3 has no novelty row, so the novelty lookup finds nothing.
    assert!(
        contexts
            .context_for_video_recipe(&video, DetectionRecipe::NortNovelty)
            .is_none()
    );
    assert!(
        contexts
            .candidate_summary_for_video(&video)
            .unwrap()
            .contains("row 3 phase_part 1 stage 1 animal 3")
    );
    let unknown = dir.join("NORT (pre)/T1/Config A/pre.T1.configA.video-7.mp4");
    assert_eq!(
        contexts.candidate_summary_for_video(&unknown).as_deref(),
        Some("none")
    );
    assert!(
        contexts
            .candidate_summary_for_video(Path::new("video-48.mp4"))
            .is_none()
    );
    assert_eq!(
        imen_video_phase_part(Path::new("NORT/T1/video-48.mp4")),
        None
    );

    // Exact matches need no message; other timepoints are listed as candidates.
    assert!(missing_imen_metadata_context_message(&dir, &video).is_none());
    let post = dir.join("NORT (post)/T1/Config A/post.T1.configA.video-22.mp4");
    let message = missing_imen_metadata_context_message(&dir, &post).unwrap();
    assert!(
        message.contains("row 2 phase_part 1 stage 0 animal 23"),
        "{message}"
    );
    assert!(missing_imen_metadata_context_message(&dir.join("absent"), &video).is_none());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn metadata_rows_need_animals_and_contexts_need_videomap_stages() {
    let dir = temp_annotation_dir("row_filters");
    std::fs::write(
        dir.join("metadata.csv"),
        "Phase,PhasePart,TrialNumber,Stage,Animal,Video_file_name\nA,1,5,1,unknown,Test 5.szv\n",
    )
    .unwrap();
    assert!(
        load_imen_contexts(&dir)
            .unwrap()
            .by_animal_stage_part
            .is_empty()
    );
    let mut context = IngressSessionContext {
        profile: "sequencetable".into(),
        phase: Some("A".into()),
        stage: Some("1".into()),
        ..Default::default()
    };
    assert_eq!(metadata_recipe_for_context(Some(&context)), None);
    context.profile = "videomap".into();
    context.stage = Some("3".into());
    assert_eq!(metadata_recipe_for_context(Some(&context)), None);

    use std::os::unix::fs::PermissionsExt;
    let metadata = dir.join("metadata.csv");
    std::fs::set_permissions(&metadata, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::read(&metadata).is_err() {
        let error = load_imen_contexts(&dir).err().unwrap().to_string();
        assert!(error.contains("failed to read"), "{error}");
    }
    std::fs::set_permissions(&metadata, std::fs::Permissions::from_mode(0o644)).unwrap();
    std::fs::remove_dir_all(dir).ok();
}
