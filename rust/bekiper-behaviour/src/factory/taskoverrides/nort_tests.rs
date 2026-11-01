#[cfg(test)]
mod nort_center_tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn arena_without_a_saved_center_does_not_infer_center_measurements() {
        let arena = PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 0.0,
            center_y: 0.0,
            width: 0.4,
            height: 0.4,
        };
        let task = build_object_recognition(&[arena], TaskParams::new(10.0), None, None).unwrap();
        assert!(
            !task
                .heuristics()
                .iter()
                .any(|h| matches!(h.name(), "in_center" | "in_periphery"))
        );
    }

    #[test]
    fn annotated_polygon_center_is_not_enlarged_to_its_bounding_box() {
        let perimeters = vec![
            PerimeterSpec::Rectangle {
                label: "arena".into(),
                center_x: 0.0,
                center_y: 0.0,
                width: 0.4,
                height: 0.4,
            },
            PerimeterSpec::Polygon {
                label: "center_zone".into(),
                vertices: vec![(0.0, -0.1), (0.1, 0.0), (0.0, 0.1), (-0.1, 0.0)],
            },
        ];
        let task =
            build_object_recognition(&perimeters, TaskParams::new(10.0), None, None).unwrap();
        let heuristics = task.heuristics();
        let center = heuristics.iter().find(|h| h.name() == "in_center").unwrap();
        let df = df! { "mouse_center_x" => [0.0, 0.09], "mouse_center_y" => [0.0, 0.09] }.unwrap();
        let scored = center.evaluate(df.lazy()).collect().unwrap();
        let mask: Vec<_> = scored
            .column("in_center")
            .unwrap()
            .bool()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(mask, [true, false]);
    }
}
#[cfg(test)]
mod task_override_tests {
    use super::*;
    use bekiper_feature::heuristic::combined::exploration::BoutFilterMode;
    use bekiper_feature::heuristic::combined::merge::MergeRule;
    use bekiper_feature::tolerance::TemporalPolicy;

    /// Inherits every default: each task keeps its built-in parameters.
    struct NoOverrides;
    impl TaskOverrides for NoOverrides {}

    /// Overrides every knob with a distinct, recognisable value.
    struct AllOverrides;
    impl TaskOverrides for AllOverrides {
        fn exploration_nose_max_distance(&self) -> Option<f64> {
            Some(0.041)
        }
        fn exploration_contact_distance(&self) -> Option<f64> {
            Some(0.006)
        }
        fn exploration_max_angle_deg(&self) -> Option<f64> {
            Some(45.0)
        }
        fn exploration_max_nose_speed(&self) -> Option<f64> {
            Some(0.07)
        }
        fn exploration_speed_filter_mode(&self) -> Option<BoutFilterMode> {
            Some(BoutFilterMode::Disabled)
        }
        fn exploration_angle_filter_mode(&self) -> Option<BoutFilterMode> {
            Some(BoutFilterMode::Disabled)
        }
        fn exploration_min_bout_seconds(&self) -> Option<f64> {
            Some(0.5)
        }
        fn exploration_max_gap_seconds(&self) -> Option<f64> {
            Some(1.0)
        }
        fn exploration_strict_boundaries(&self) -> Option<bool> {
            Some(false)
        }
        fn exploration_min_total_exploration(&self) -> Option<f64> {
            Some(15.0)
        }
        fn exploration_analysis_window_seconds(&self) -> Option<f64> {
            Some(300.0)
        }
        fn exploration_window_mode(&self) -> Option<NortWindowMode> {
            Some(NortWindowMode::FullAnalysis)
        }
        fn exploration_perimeter_erosion_meters(&self) -> Option<f64> {
            Some(0.0)
        }
        fn exploration_exclude_posture(&self) -> Option<bool> {
            Some(true)
        }
        fn exploration_gap_density_threshold(&self) -> Option<f64> {
            Some(0.25)
        }
        fn qualia_enable(&self) -> Option<bool> {
            Some(false)
        }
        fn qualia_body_proximity_distance(&self) -> Option<f64> {
            Some(0.051)
        }
        fn qualia_whisker_distance(&self) -> Option<f64> {
            Some(0.036)
        }
        fn qualia_whisker_angle(&self) -> Option<f64> {
            Some(46.0)
        }
        fn qualia_olfaction_distance(&self) -> Option<f64> {
            Some(0.052)
        }
        fn qualia_olfaction_angle(&self) -> Option<f64> {
            Some(61.0)
        }
        fn qualia_merge_rules(&self) -> Option<std::collections::HashMap<String, MergeRule>> {
            Some([("close".to_string(), MergeRule::default())].into())
        }
        fn ymaze_temporal_policy(&self) -> Option<TemporalPolicy> {
            Some(TemporalPolicy::default())
        }
        fn ymaze_proximal_threshold(&self) -> Option<f64> {
            Some(0.4)
        }
        fn ymaze_start_mode(&self) -> Option<crate::radial_arm::YMazeStartMode> {
            Some(crate::radial_arm::YMazeStartMode::VideoStart)
        }
        fn ymaze_apply_dwell_filter(&self) -> Option<bool> {
            Some(false)
        }
        fn ymaze_arm_membership_hysteresis(&self) -> Option<bool> {
            Some(true)
        }
        fn ymaze_zone_max_gap_seconds(&self) -> Option<f64> {
            Some(0.2)
        }
        fn ymaze_pose_gap_fill_seconds(&self) -> Option<f64> {
            Some(0.3)
        }
        fn ymaze_zone_occupancy(&self) -> Option<bekiper_feature::zone::ZoneOccupancy> {
            Some(bekiper_feature::zone::ZoneOccupancy::Raw)
        }
        fn rearing_max_speed(&self) -> Option<f64> {
            Some(0.11)
        }
        fn rearing_max_body_length_ratio(&self) -> Option<f64> {
            Some(0.71)
        }
        fn rearing_max_body_length(&self) -> Option<f64> {
            Some(0.051)
        }
        fn rearing_forepaw_proximity(&self) -> Option<f64> {
            Some(0.021)
        }
        fn grooming_max_speed(&self) -> Option<f64> {
            Some(0.012)
        }
        fn grooming_max_body_length(&self) -> Option<f64> {
            Some(0.061)
        }
        fn grooming_min_body_length(&self) -> Option<f64> {
            Some(0.031)
        }
        fn grooming_forepaw_proximity(&self) -> Option<f64> {
            Some(0.022)
        }
    }

    fn nort_perimeters() -> Vec<PerimeterSpec> {
        vec![
            PerimeterSpec::Triangle {
                label: "object_1".into(),
                vertices: [(0.0, 0.0), (0.02, 0.0), (0.0, 0.02)],
            },
            PerimeterSpec::Circle {
                label: "object_2".into(),
                center_x: 0.2,
                center_y: 0.2,
                radius: 0.02,
            },
            // A Y-maze outline is never an object.
            PerimeterSpec::RadialMaze {
                label: "maze".into(),
                center_vertices: vec![(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)],
                arms: vec![],
            },
        ]
    }

    fn oft_perimeters() -> Vec<PerimeterSpec> {
        vec![PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 0.2,
            center_y: 0.2,
            width: 0.4,
            height: 0.4,
        }]
    }

    #[test]
    fn nort_overrides_replace_every_exploration_and_qualia_knob() {
        let task = object_recognition_task(
            &nort_perimeters(),
            TaskParams::new(30.0),
            Some(ObjectRecognitionProtocol::NortTraining),
            Some(&AllOverrides),
        );
        assert_eq!(task.protocol, ObjectRecognitionProtocol::NortTraining);
        let labels: Vec<_> = task
            .object_perimeters
            .iter()
            .map(|p| p.label.as_str())
            .collect();
        assert_eq!(labels, ["object_1", "object_2"]);
        assert!(matches!(
            task.object_perimeters[0].shape,
            ObjectShape::Triangle(_)
        ));
        assert_eq!(
            (
                task.nose_max_distance,
                task.contact_distance,
                task.max_angle_deg
            ),
            (0.041, 0.006, 45.0)
        );
        assert_eq!(task.max_nose_speed, 0.07);
        assert_eq!(task.speed_filter_mode, BoutFilterMode::Disabled);
        assert_eq!(task.angle_filter_mode, BoutFilterMode::Disabled);
        assert_eq!((task.min_bout_seconds, task.max_gap_seconds), (0.5, 1.0));
        assert!(!task.strict_boundaries);
        assert_eq!(task.min_total_exploration, 15.0);
        assert_eq!(task.analysis_window_seconds, 300.0);
        assert_eq!(task.window_mode, NortWindowMode::FullAnalysis);
        assert_eq!(task.perimeter_erosion_meters, 0.0);
        assert!(task.exclude_posture);
        let names: Vec<_> = task
            .heuristics()
            .iter()
            .map(|h| h.name().to_string())
            .collect();
        for posture in [
            "rearing",
            "grooming",
            crate::object_recognition::POSTURE_COLUMN,
        ] {
            assert!(names.iter().any(|name| name == posture), "{names:?}");
        }
        assert_eq!(task.gap_density_threshold, 0.25);
        assert!(!task.enable_qualia);
        assert_eq!(
            (
                task.qualia_body_proximity_distance,
                task.qualia_whisker_distance,
                task.qualia_whisker_angle,
                task.qualia_olfaction_distance,
                task.qualia_olfaction_angle,
            ),
            (0.051, 0.036, 46.0, 0.052, 61.0)
        );
        assert!(task.merge_rules.contains_key("close"));
        assert!(task.center_perimeter.is_none());
    }

    #[test]
    fn empty_overrides_keep_every_task_default() {
        assert!(NoOverrides.exploration_whisker().is_none());
        let params = TaskParams::new(30.0);
        let defaults = object_recognition_task(&nort_perimeters(), params.clone(), None, None);
        let inherited =
            object_recognition_task(&nort_perimeters(), params.clone(), None, Some(&NoOverrides));
        assert_eq!(inherited.nose_max_distance, defaults.nose_max_distance);
        assert_eq!(inherited.min_bout_seconds, defaults.min_bout_seconds);
        assert_eq!(inherited.window_mode, defaults.window_mode);
        assert_eq!(inherited.protocol, defaults.protocol);
        assert!(!inherited.exclude_posture);

        let oft = open_field_task(&oft_perimeters(), params.clone(), None).unwrap();
        let oft_inherited =
            open_field_task(&oft_perimeters(), params.clone(), Some(&NoOverrides)).unwrap();
        assert_eq!(oft_inherited.rearing_max_speed, oft.rearing_max_speed);
        assert_eq!(
            oft_inherited.grooming_min_body_length,
            oft.grooming_min_body_length
        );

        let maze = [PerimeterSpec::RadialMaze {
            label: "maze".into(),
            center_vertices: vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            arms: vec![vec![(0.1, -0.1), (1.0, -0.1), (1.0, 0.1), (0.1, 0.1)]],
        }];
        for overrides in [
            None,
            Some(&NoOverrides as &dyn TaskOverrides),
            Some(&AllOverrides),
        ] {
            let task = build_task(
                TaskType::YMaze,
                &maze,
                params.clone(),
                None,
                None,
                overrides,
            );
            assert!(task.is_ok());
        }
    }

    #[test]
    fn oft_overrides_replace_every_rearing_and_grooming_knob() {
        let task = open_field_task(
            &oft_perimeters(),
            TaskParams::new(30.0),
            Some(&AllOverrides),
        )
        .unwrap();
        assert_eq!(
            (
                task.rearing_max_speed,
                task.rearing_max_body_length_ratio,
                task.rearing_max_body_length,
                task.rearing_forepaw_proximity,
            ),
            (0.11, 0.71, 0.051, 0.021)
        );
        assert_eq!(
            (
                task.grooming_max_speed,
                task.grooming_max_body_length,
                task.grooming_min_body_length,
                task.grooming_forepaw_proximity,
            ),
            (0.012, 0.061, 0.031, 0.022)
        );
    }

    #[test]
    fn ymaze_overrides_replace_every_knob() {
        let maze = Perimeter::new(
            "maze",
            RadialMaze::new(
                Polygon::new(vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]),
                vec![Polygon::new(vec![
                    (0.1, -0.1),
                    (1.0, -0.1),
                    (1.0, 0.1),
                    (0.1, 0.1),
                ])],
            ),
        );
        let mut task = YMaze::new(maze, 30.0);
        task.apply_overrides(&AllOverrides).unwrap();
        assert_eq!(task.temporal_policy, Some(TemporalPolicy::default()));
        assert_eq!(task.proximal_threshold, 0.4);
        assert_eq!(
            task.start_mode,
            crate::radial_arm::YMazeStartMode::VideoStart
        );
        assert!(!task.apply_dwell_filter);
        assert!(task.arm_membership_hysteresis);
        assert_eq!(
            (task.zone_max_gap_seconds, task.pose_gap_fill_seconds),
            (0.2, 0.3)
        );
        assert_eq!(
            task.zone_occupancy,
            bekiper_feature::zone::ZoneOccupancy::Raw
        );
    }
}
