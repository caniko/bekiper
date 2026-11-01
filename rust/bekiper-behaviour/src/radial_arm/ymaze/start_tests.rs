#[cfg(test)]
mod ymaze_start_mode_tests {
    use super::YMazeStartMode;

    #[test]
    fn default_is_first_detection() {
        // FirstDetection preserves the legacy hardcoded behaviour and
        // matches the imen project's interpretation of the criteria doc.
        assert_eq!(YMazeStartMode::default(), YMazeStartMode::FirstDetection);
    }

    #[derive(serde::Deserialize)]
    struct Outer {
        start_mode: YMazeStartMode,
    }

    #[test]
    fn deserialises_snake_case_first_detection() {
        let parsed: Outer = toml::from_str(r#"start_mode = "first_detection""#).unwrap();
        assert_eq!(parsed.start_mode, YMazeStartMode::FirstDetection);
    }

    #[test]
    fn deserialises_snake_case_video_start() {
        let parsed: Outer = toml::from_str(r#"start_mode = "video_start""#).unwrap();
        assert_eq!(parsed.start_mode, YMazeStartMode::VideoStart);
    }

    #[test]
    fn rejects_unknown_variant() {
        let result: Result<Outer, _> = toml::from_str(r#"start_mode = "trim_to_first""#);
        assert!(result.is_err(), "unknown variant must fail to parse");
    }
}

#[cfg(test)]
mod ymaze_task_detail_tests {
    use super::*;
    use bekiper_core::video::FrameTiming;
    use bekiper_perimeter::polygon::Polygon;

    fn three_arm_ymaze() -> YMaze {
        let center = Polygon::new(vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]);
        let arm_a = Polygon::new(vec![(0.1, -0.2), (1.1, -0.2), (1.1, 0.2), (0.1, 0.2)]);
        let arm_b = Polygon::new(vec![(-0.2, 0.1), (0.2, 0.1), (0.2, 1.1), (-0.2, 1.1)]);
        let arm_c = Polygon::new(vec![(-1.1, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-1.1, 0.2)]);
        let maze = Perimeter::new("maze", RadialMaze::new(center, vec![arm_a, arm_b, arm_c]));
        let mut task = YMaze::new(maze, 1.0);
        task.proximal_threshold = 0.30;
        task
    }

    /// Head, tail and centroid share one path of `(x, y)` points.
    fn poses(points: &[Option<(f64, f64)>]) -> LazyFrame {
        let xs: Vec<_> = points.iter().map(|p| p.map(|p| p.0)).collect();
        let ys: Vec<_> = points.iter().map(|p| p.map(|p| p.1)).collect();
        df! {
            "center_ear_x" => &xs, "center_ear_y" => &ys,
            "tail_base_x" => &xs, "tail_base_y" => &ys,
            "mouse_center_x" => &xs, "mouse_center_y" => &ys,
        }
        .unwrap()
        .lazy()
    }

    const A: Option<(f64, f64)> = Some((0.9, 0.0));
    const B: Option<(f64, f64)> = Some((0.0, 0.9));
    const C: Option<(f64, f64)> = Some((-0.9, 0.0));
    const MID: Option<(f64, f64)> = Some((0.0, 0.0));

    #[test]
    fn analyze_detailed_starts_at_first_detection_and_reports_typed_detail() {
        let task = three_arm_ymaze();
        let path = [None, None, A, A, MID, B, B, MID, C, C];
        let (analysis, detail) = task.analyze_detailed(poses(&path), &FrameTiming::cfr(1.0));
        assert_eq!(analysis.task_type, TaskType::YMaze);
        let names: Vec<_> = analysis.summaries.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            ["in_arm_A", "in_arm_B", "in_arm_C", "in_center", "freezing"]
        );
        // Arm decompositions carry the threshold-crossing components.
        let arm_a = analysis
            .decompositions
            .iter()
            .find(|d| d.heuristic_name == "in_arm_A")
            .unwrap();
        let components: Vec<_> = arm_a.components.iter().map(|c| c.label.as_str()).collect();
        assert!(
            components.contains(&"head_past_threshold"),
            "{components:?}"
        );
        assert!(
            analysis
                .decompositions
                .iter()
                .any(|d| d.heuristic_name == "in_center")
        );
        // The two leading undetected frames are trimmed before analysis.
        assert_eq!(analysis.evaluated_df.as_ref().unwrap().height(), 8);
        let detail = detail.unwrap();
        assert_eq!(detail.task_type(), TaskType::YMaze);
        let TaskDetail::YMaze(result) = &detail else {
            unreachable!()
        };
        assert_eq!(result.analysis.arm_sequence, ["A", "B", "C"]);
        assert_eq!(result.analysis.correct_alternations, 1);
        assert_eq!(result.analysis_duration_seconds, 8.0);
        assert_eq!(result.total_duration_seconds, 10.0);

        // Historical results without the hysteresis flag default to enabled.
        let mut json = detail.as_json_value();
        json.as_object_mut()
            .unwrap()
            .remove("arm_membership_hysteresis");
        let restored: YMazeResult = serde_json::from_value(json).unwrap();
        assert!(restored.arm_membership_hysteresis);
    }

    #[test]
    fn hysteresis_latches_through_missing_pose_only_when_enabled() {
        let path = [A, A, None, A, MID, B, B];
        let mut task = three_arm_ymaze();
        task.pose_gap_fill_seconds = 0.0;
        task.temporal_policy = Some(bekiper_feature::tolerance::TemporalPolicy::default());
        let in_arm_a = |task: &YMaze| -> Vec<bool> {
            let result = task.full_analyze(poses(&path));
            let df = result.evaluated_df.unwrap();
            df.column("in_arm_A")
                .unwrap()
                .bool()
                .unwrap()
                .into_iter()
                .map(|v| v.unwrap_or(false))
                .collect()
        };
        assert!(in_arm_a(&task)[2], "latched across the pose gap");
        task.arm_membership_hysteresis = false;
        assert!(!in_arm_a(&task)[2], "raw membership breaks on the gap");
    }

    #[test]
    fn recording_without_any_detection_has_zero_speed() {
        for path in [&[None, None, None][..], &[]] {
            let result = three_arm_ymaze().full_analyze(poses(path));
            assert_eq!(result.total_mean_speed, 0.0);
            assert_eq!(result.total_distance_meters, 0.0);
            assert!(result.analysis.arm_sequence.is_empty());
        }
    }
}
