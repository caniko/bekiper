#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bekiper_canonical_{label}_{}_{}",
            std::process::id(),
            now_unix_s()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_legacy_nort(dir: &Path) {
        let plugin = dir.join("plugin_files");
        fs::create_dir_all(&plugin).unwrap();
        fs::write(
            plugin.join("perimeter-rectangle-training_1.csv"),
            "familiar,48,49,25,38,A2_training_1.png,232,216\nnovel,152,150,28,27,A2_training_1.png,232,216\n",
        )
        .unwrap();
        fs::write(
            plugin.join("reference-training_1.csv"),
            "reference,100,107,A2_training_1.png,232,216\n",
        )
        .unwrap();
        fs::write(
            plugin.join("change_reference-training_1.csv"),
            "reference,100,109,A1_training_1.png,232,216\n",
        )
        .unwrap();
        fs::write(
            dir.join("metadata.csv"),
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName\nA,2,47,1,1,1,,training_1,A1_training_1\n",
        )
        .unwrap();
    }

    fn write_grouped_legacy_nort(dir: &Path) {
        write_legacy_nort(dir);
        fs::write(
            dir.join("metadata.csv"),
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName\nA,2,47,1,1,1,,training_1,A1_training_1\nA,2,48,1,2,1,,training_1,A1_training_1\n",
        )
        .unwrap();
    }

    #[test]
    fn ingestion_eligibility_is_explicit_and_immutable_on_retries() {
        for reviewed in [false, true] {
            let dir = temp_dir(if reviewed {
                "reviewed_import"
            } else {
                "unreviewed_import"
            });
            write_legacy_nort(&dir);
            let store =
                CanonicalAnnotationStore::import_legacy_with_review(&dir, reviewed).unwrap();
            let first = store.ingestion().unwrap();
            assert_eq!(first.len(), 1);
            let provenance = first.values().next().unwrap();
            assert!(provenance.imported);
            assert_eq!(provenance.reviewed_externally, reviewed);
            assert_eq!(provenance.validation_request_id.is_some(), reviewed);
            CanonicalAnnotationStore::import_legacy_with_review(&dir, !reviewed).unwrap();
            assert_eq!(store.ingestion().unwrap(), first);
            store.write_perimeter("training_1", &[]).unwrap();
            assert_eq!(store.ingestion().unwrap(), first);
        }
    }

    #[test]
    fn concurrent_ingestion_reuses_one_immutable_request() {
        let dir = temp_dir("concurrent_ingestion");
        write_legacy_nort(&dir);
        let start = std::sync::Barrier::new(4);
        let results = std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        CanonicalAnnotationStore::import_legacy_with_review(&dir, true)?.ingestion()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().unwrap().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results[0].len(), 1);
        assert!(
            results[0]
                .values()
                .next()
                .unwrap()
                .validation_request_id
                .is_some()
        );
        for result in &results[1..] {
            assert_eq!(result, &results[0]);
        }
        let store = CanonicalAnnotationStore::open(&dir).unwrap().unwrap();
        assert!(!store.load_perimeter("training_1").unwrap().is_empty());
    }

    #[test]
    fn rejected_import_keeps_existing_geometry_and_request() {
        let dir = temp_dir("rejected_import");
        write_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy_with_review(&dir, true).unwrap();
        let sessions = store.load_sessions().unwrap();
        let geometry = store.load_perimeter("training_1").unwrap();
        let request = store.ingestion().unwrap();
        // A bad destination must be detected before any existing files change.
        fs::remove_dir_all(store.root.join("references")).unwrap();
        fs::write(store.root.join("references"), "unwritable destination").unwrap();
        assert!(CanonicalAnnotationStore::import_legacy_with_review(&dir, true).is_err());
        assert_eq!(store.load_sessions().unwrap(), sessions);
        assert_eq!(store.load_perimeter("training_1").unwrap(), geometry);
        assert_eq!(store.ingestion().unwrap(), request);
    }

    #[test]
    fn failed_ingestion_reuses_request_and_legacy_state_does_not_infer_review() {
        let dir = temp_dir("failed_ingestion");
        write_legacy_nort(&dir);
        let perimeter = dir.join("plugin_files/perimeter-rectangle-training_1.csv");
        let original = fs::read(&perimeter).unwrap();
        fs::write(&perimeter, "invalid geometry").unwrap();
        assert!(CanonicalAnnotationStore::import_legacy_with_review(&dir, true).is_err());
        let store = CanonicalAnnotationStore::create_or_open(&dir);
        let pending = store.ingestion().unwrap();
        assert!(
            pending
                .values()
                .next()
                .unwrap()
                .validation_request_id
                .is_some()
        );
        fs::write(&perimeter, original).unwrap();
        CanonicalAnnotationStore::import_legacy_with_review(&dir, true).unwrap();
        assert_eq!(store.ingestion().unwrap(), pending);
        fs::remove_file(store.root.join("ingestion.toml")).unwrap();
        CanonicalAnnotationStore::import_legacy_with_review(&dir, true).unwrap();
        assert!(
            !store
                .ingestion()
                .unwrap()
                .values()
                .next()
                .unwrap()
                .reviewed_externally
        );
    }

    #[test]
    fn import_legacy_creates_canonical_store() {
        let dir = temp_dir("import");
        write_legacy_nort(&dir);

        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();

        assert!(store.root.join("manifest.toml").exists());
        assert!(
            store
                .root
                .join("sessions")
                .join("metadata_row_2_animal_1_stage_1_part_2.toml")
                .exists()
        );
        assert!(
            store
                .root
                .join("perimeters")
                .join("training_1.toml")
                .exists()
        );
        assert!(
            store
                .root
                .join("references")
                .join("training_1.toml")
                .exists()
        );
        let sessions = store.load_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].target_image.as_deref(), Some("A1_training_1"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejected_geometry_batch_does_not_change_existing_files() {
        let dir = temp_dir("rejected_geometry_batch");
        write_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let mut session = store.load_sessions().unwrap().remove(0);
        let path = store.perimeter_path(&session.perimeter_id);
        let baseline = fs::read(&path).unwrap();
        // An unreadable destination is detected before publishing any geometry.
        session.key = "directory-not-file".into();
        fs::create_dir(store.session_path(&session.key)).unwrap();
        assert!(
            store
                .write_geometry_edit(&session.perimeter_id.clone(), &[], &[session], None)
                .is_err()
        );
        assert_eq!(fs::read(path).unwrap(), baseline);
    }

    #[test]
    fn canonical_resolver_matches_legacy_before_edits() {
        let dir = temp_dir("resolve");
        write_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let ctx = VideoContext {
            animal: 1,
            stage: 1,
            phase_part: 2,
        };

        let resolved = store.resolve(&ctx).unwrap().unwrap();
        let mut legacy =
            parse_rectangle_csv(&dir.join("plugin_files/perimeter-rectangle-training_1.csv"))
                .unwrap();
        translate_perimeters(&mut legacy, 0.0, 2.0);
        assert_eq!(resolved, legacy);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn pending_session_resolves_to_empty_perimeters() {
        let dir = temp_dir("pending_session");
        let store = CanonicalAnnotationStore::create_or_open(&dir);
        let session = CanonicalSession {
            key: "pending".into(),
            video_number: None,
            animal: None,
            stage: None,
            phase_part: None,
            perimeter_id: "not_written_yet".into(),
            source_perimeter_id: None,
            coordinate_space: CoordinateSpace::Source,
            target_image: None,
        };

        assert!(
            store
                .resolve_session_perimeters(&session)
                .unwrap()
                .is_empty()
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn group_detection_finds_sessions_with_same_source_and_target() {
        let dir = temp_dir("group_detect");
        write_grouped_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let session = store
            .resolve_session(&VideoContext {
                animal: 1,
                stage: 1,
                phase_part: 2,
            })
            .unwrap()
            .unwrap();

        let info = store.group_info_for_session(&session).unwrap().unwrap();

        assert_eq!(info.key.source_perimeter_id, "training_1");
        assert_eq!(info.key.target_image, "A1_training_1");
        assert_eq!(info.members.len(), 2);
        assert_eq!(
            session.correction_kind(),
            CanonicalCorrectionKind::SourceShared
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn assign_sessions_to_source_group_creates_shared_group_key() {
        let dir = temp_dir("assign_group");
        let store = CanonicalAnnotationStore::create_or_open(&dir);
        fs::create_dir_all(store.root.join("perimeters")).unwrap();
        fs::create_dir_all(store.root.join("sessions")).unwrap();
        fs::create_dir_all(store.root.join("references")).unwrap();
        store.write_sync_state().unwrap();
        store
            .write_perimeter(
                "training_1",
                &[PerimeterSpec::Rectangle {
                    label: "familiar".into(),
                    center_x: 48.0,
                    center_y: 49.0,
                    width: 25.0,
                    height: 38.0,
                }],
            )
            .unwrap();
        store
            .write_perimeter(
                "training_2",
                &[PerimeterSpec::Rectangle {
                    label: "familiar".into(),
                    center_x: 60.0,
                    center_y: 61.0,
                    width: 25.0,
                    height: 38.0,
                }],
            )
            .unwrap();
        store
            .write_session(&CanonicalSession {
                key: "session_1".into(),
                video_number: Some(1),
                animal: Some(1),
                stage: Some(1),
                phase_part: Some(2),
                perimeter_id: "training_1".into(),
                source_perimeter_id: None,
                coordinate_space: CoordinateSpace::Source,
                target_image: None,
            })
            .unwrap();
        store
            .write_session(&CanonicalSession {
                key: "session_2".into(),
                video_number: Some(2),
                animal: Some(2),
                stage: Some(1),
                phase_part: Some(2),
                perimeter_id: "training_2".into(),
                source_perimeter_id: None,
                coordinate_space: CoordinateSpace::Source,
                target_image: None,
            })
            .unwrap();

        let updated = store
            .assign_sessions_to_source_group(
                &["session_1".into(), "session_2".into()],
                "training_1",
                "A1_training_1",
            )
            .unwrap();

        assert_eq!(updated.len(), 2);
        assert!(
            updated
                .iter()
                .all(|session| session.perimeter_id == "training_1")
        );
        assert!(
            updated
                .iter()
                .all(|session| session.source_perimeter_id.is_none())
        );
        assert!(
            updated
                .iter()
                .all(|session| session.coordinate_space == CoordinateSpace::Source)
        );
        assert!(
            updated
                .iter()
                .all(|session| session.target_image.as_deref() == Some("A1_training_1"))
        );
        let reloaded = store.load_session("session_1").unwrap();
        let info = store.group_info_for_session(&reloaded).unwrap().unwrap();
        assert_eq!(info.key.source_perimeter_id, "training_1");
        assert_eq!(info.key.target_image, "A1_training_1");
        assert_eq!(info.members.len(), 2);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn individual_rectification_updates_only_current_group_member() {
        let dir = temp_dir("group_individual");
        write_grouped_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let ctx_one = VideoContext {
            animal: 1,
            stage: 1,
            phase_part: 2,
        };
        let ctx_two = VideoContext {
            animal: 2,
            stage: 1,
            phase_part: 2,
        };
        let mut resolved = store.resolve(&ctx_one).unwrap().unwrap();
        translate_perimeters(&mut resolved, 7.0, 3.0);
        let session = store.resolve_session(&ctx_one).unwrap().unwrap();

        let updated = store
            .materialize_target_perimeter(&session.key, &resolved)
            .unwrap();
        let other = store.resolve_session(&ctx_two).unwrap().unwrap();

        assert_eq!(
            updated.correction_kind(),
            CanonicalCorrectionKind::IndividualTarget
        );
        assert_eq!(
            other.correction_kind(),
            CanonicalCorrectionKind::SourceShared
        );
        assert_eq!(store.resolve(&ctx_one).unwrap().unwrap(), resolved);
        assert_ne!(store.resolve(&ctx_two).unwrap().unwrap(), resolved);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn group_rectification_updates_all_group_members() {
        let dir = temp_dir("group_rectify");
        write_grouped_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let ctx_one = VideoContext {
            animal: 1,
            stage: 1,
            phase_part: 2,
        };
        let ctx_two = VideoContext {
            animal: 2,
            stage: 1,
            phase_part: 2,
        };
        let mut resolved = store.resolve(&ctx_one).unwrap().unwrap();
        translate_perimeters(&mut resolved, -4.0, 8.0);
        let session = store.resolve_session(&ctx_one).unwrap().unwrap();

        let updated = store
            .materialize_group_target_perimeter(&session.key, &resolved)
            .unwrap();

        assert_eq!(updated.len(), 2);
        assert!(
            updated
                .iter()
                .all(|session| session.correction_kind() == CanonicalCorrectionKind::GroupTarget)
        );
        assert_eq!(store.resolve(&ctx_one).unwrap().unwrap(), resolved);
        assert_eq!(store.resolve(&ctx_two).unwrap().unwrap(), resolved);
        assert!(
            store
                .perimeter_path("training_1__group_target__A1_training_1")
                .exists()
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn export_group_rectification_updates_all_group_metadata_rows() {
        let dir = temp_dir("group_export");
        write_grouped_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let ctx = VideoContext {
            animal: 1,
            stage: 1,
            phase_part: 2,
        };
        let resolved = store.resolve(&ctx).unwrap().unwrap();
        let session = store.resolve_session(&ctx).unwrap().unwrap();
        store
            .materialize_group_target_perimeter(&session.key, &resolved)
            .unwrap();

        store.export_legacy().unwrap();

        let metadata = fs::read_to_string(dir.join("metadata.csv")).unwrap();
        assert_eq!(
            metadata
                .matches("training_1__group_target__A1_training_1")
                .count(),
            2
        );
        assert!(
            dir.join(
                "plugin_files/perimeter-rectangle-training_1__group_target__A1_training_1.csv"
            )
            .exists()
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn target_geometry_materializes_target_perimeter_without_changing_source() {
        let dir = temp_dir("target_materialize");
        write_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let ctx = VideoContext {
            animal: 1,
            stage: 1,
            phase_part: 2,
        };
        let source_before = store.load_perimeter("training_1").unwrap();
        let mut resolved = store.resolve(&ctx).unwrap().unwrap();
        translate_perimeters(&mut resolved, 12.0, -4.0);
        let session = store.resolve_session(&ctx).unwrap().unwrap();

        let updated = store
            .materialize_target_perimeter(&session.key, &resolved)
            .unwrap();
        let target_id = test_individual_target_id("training_1", "A1_training_1");

        assert_eq!(updated.coordinate_space, CoordinateSpace::Target);
        assert_eq!(updated.source_perimeter_id.as_deref(), Some("training_1"));
        assert_eq!(updated.perimeter_id, target_id);
        assert_eq!(store.load_perimeter("training_1").unwrap(), source_before);
        assert_eq!(store.load_perimeter(&target_id).unwrap(), resolved);
        assert_eq!(store.resolve(&ctx).unwrap().unwrap(), resolved);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn export_target_rectification_writes_target_specific_legacy_mapping() {
        let dir = temp_dir("target_export");
        write_legacy_nort(&dir);
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        let ctx = VideoContext {
            animal: 1,
            stage: 1,
            phase_part: 2,
        };
        let mut resolved = store.resolve(&ctx).unwrap().unwrap();
        translate_perimeters(&mut resolved, 5.0, 6.0);
        let session = store.resolve_session(&ctx).unwrap().unwrap();
        store
            .materialize_target_perimeter(&session.key, &resolved)
            .unwrap();

        store.export_legacy().unwrap();

        let target_id = test_individual_target_id("training_1", "A1_training_1");
        let target_path = dir
            .join("plugin_files")
            .join(format!("perimeter-rectangle-{target_id}.csv"));
        assert!(target_path.exists());
        assert_eq!(
            store.load_perimeter("training_1").unwrap(),
            parse_rectangle_csv(&dir.join("plugin_files/perimeter-rectangle-training_1.csv"))
                .unwrap()
        );
        let metadata = fs::read_to_string(dir.join("metadata.csv")).unwrap();
        assert!(metadata.contains(&target_id));
        assert!(!metadata.contains(",,training_1,A1_training_1"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn export_preserves_circle_perimeter_shape() {
        let dir = temp_dir("circle_export");
        let store = CanonicalAnnotationStore::create_or_open(&dir);
        fs::create_dir_all(store.root.join("perimeters")).unwrap();
        fs::create_dir_all(store.root.join("sessions")).unwrap();
        fs::create_dir_all(store.root.join("references")).unwrap();
        store
            .write_perimeter(
                "circle_1",
                &[PerimeterSpec::Circle {
                    label: "object_1".into(),
                    center_x: 20.0,
                    center_y: 30.0,
                    radius: 5.0,
                }],
            )
            .unwrap();

        store.export_legacy().unwrap();

        assert!(
            dir.join("plugin_files/perimeter-circle-circle_1.csv")
                .exists()
        );
        assert!(
            !dir.join("plugin_files/perimeter-rectangle-circle_1.csv")
                .exists()
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn sync_detects_both_sides_changed_as_conflict() {
        let dir = temp_dir("conflict");
        write_legacy_nort(&dir);
        CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1));
        fs::write(
            dir.join("plugin_files/perimeter-rectangle-training_1.csv"),
            "familiar,50,49,25,38,A2_training_1.png,232,216\n",
        )
        .unwrap();
        fs::write(
            dir.join("bekiper/perimeters/training_1.toml"),
            "perimeters = []\n",
        )
        .unwrap();

        let error = CanonicalAnnotationStore::sync(
            &dir,
            SyncOptions {
                mode: SyncMode::TwoWay,
                dry_run: false,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("conflict"));
        assert!(
            !sorted_files(&dir.join("bekiper/conflicts"))
                .unwrap()
                .is_empty()
        );

        fs::remove_dir_all(dir).ok();
    }
}
