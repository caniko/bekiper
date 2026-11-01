#[cfg(test)]
mod ymaze_arm_start_crossline_tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/golden/crossline_entry_parity.json"))
            .expect("parse shared cross-line fixture")
    }

    fn fixture_maze(fixture: &serde_json::Value) -> Perimeter<RadialMaze> {
        let vertices = |v: &serde_json::Value| -> Polygon {
            Polygon::new(serde_json::from_value::<Vec<(f64, f64)>>(v.clone()).unwrap())
        };
        Perimeter::new(
            "maze",
            RadialMaze::new(
                vertices(&fixture["center"]),
                ["A", "B", "C"]
                    .into_iter()
                    .map(|a| vertices(&fixture["arms"][a]))
                    .collect(),
            ),
        )
    }

    fn crossline_task(p: f64) -> YMaze {
        let mut task = YMaze::new(fixture_maze(&fixture()), 10.0);
        task.threshold_geometry = YMazeThresholdGeometry::ArmStartCrosslineV1;
        task.proximal_threshold = p;
        task.start_mode = YMazeStartMode::VideoStart;
        task.apply_dwell_filter = false;
        task.arm_membership_hysteresis = false;
        task.zone_max_gap_seconds = 0.0;
        task.pose_gap_fill_seconds = 0.0;
        task
    }

    fn frames(points: &[(f64, f64)]) -> LazyFrame {
        let x: Vec<f64> = points.iter().map(|p| p.0).collect();
        let y: Vec<f64> = points.iter().map(|p| p.1).collect();
        df! {
            "center_ear_x" => x.clone(), "center_ear_y" => y.clone(),
            "tail_base_x" => x.clone(), "tail_base_y" => y.clone(),
            "mouse_center_x" => x, "mouse_center_y" => y,
        }
        .unwrap()
        .lazy()
    }

    #[test]
    fn shared_crossline_entry_fixture() {
        let fixture = fixture();
        for case in fixture["cases"].as_array().unwrap() {
            let samples = case["samples"].as_array().unwrap();
            let values =
                |c: usize| -> Vec<Option<f64>> { samples.iter().map(|r| r[c].as_f64()).collect() };
            let input = df! {
                "center_ear_x" => values(1), "center_ear_y" => values(2),
                "tail_base_x" => values(3), "tail_base_y" => values(4),
                "mouse_center_x" => vec![0.9; samples.len()],
                "mouse_center_y" => vec![0.0; samples.len()],
            }
            .unwrap();
            for pct in case["thresholds"].as_array().unwrap() {
                let pct = pct.as_u64().unwrap();
                let task = crossline_task(pct as f64 / 100.0);
                assert!(!task.uses_principal_entry_rule());
                let result = task.full_analyze(input.clone().lazy());
                let expected = &case["expected"][pct.to_string()];
                assert_eq!(
                    json!(result.analysis.arm_sequence),
                    expected["sequence"],
                    "{pct}"
                );
                let p = result.scoring_provenance.as_ref().unwrap();
                for arm in ["A", "B", "C"] {
                    assert_eq!(
                        p["entry_frames"][format!("in_arm_{arm}")],
                        expected["entry_frames"][arm],
                        "{pct} {arm}"
                    );
                }
            }
        }
    }

    #[test]
    fn p25_without_dwell_counts_single_frame_entry_and_records_crossline_provenance() {
        // 0.4 m is past the p=0.25 line (0.3375 at y = 0); 0.2 m is not.
        let result =
            crossline_task(0.25).full_analyze(frames(&[(0.2, 0.0), (0.4, 0.0), (0.2, 0.0)]));
        assert_eq!(result.analysis.arm_sequence, ["A"]);
        let p = result.scoring_provenance.unwrap();
        assert_eq!(p["geometry_version"], "imen_arm_start_crossline_v1");
        assert_eq!(p["minimum_dwell_seconds"], 0.0);
        assert!(p.get("threshold_radii_squared").is_none());
        assert_eq!(p["arm_frames"].as_array().unwrap().len(), 3);
        // Endpoint order follows the canonical vertex order, so compare as a set.
        let ends = &p["crossline_endpoints"][0];
        let (e1, e2) = (ends["p1"].clone(), ends["p2"].clone());
        assert!(
            (e1 == json!([0.35, -0.15]) && e2 == json!([0.325, 0.15]))
                || (e2 == json!([0.35, -0.15]) && e1 == json!([0.325, 0.15])),
            "{ends}"
        );
    }

    #[test]
    fn p0_one_second_dwell_with_one_second_gap_validates_and_bridges() {
        let policy = bekiper_feature::tolerance::TemporalPolicy {
            min_duration_seconds: 1.0,
            max_gap_seconds: 1.0,
            ..Default::default()
        };
        let mut task = crossline_task(0.0);
        task.temporal_policy = Some(policy);
        task.zone_max_gap_seconds = 1.0;
        task.validate_temporal_policy().unwrap();
        // Same policy under the radial principal rule must stay rejected.
        let mut radial = crossline_task(0.25);
        radial.threshold_geometry = YMazeThresholdGeometry::ImenRadialVertexSpanV1;
        radial.temporal_policy = Some(policy);
        assert!(radial.validate_temporal_policy().is_err());
        // 0.6 s in the arm, 0.5 s outside, 0.6 s in the arm at 10 fps.
        let mut points = vec![(0.5, 0.0); 6];
        points.extend(vec![(0.0, -0.5); 5]);
        points.extend(vec![(0.5, 0.0); 6]);
        let bridged = task.full_analyze(frames(&points));
        assert_eq!(bridged.analysis.arm_sequence, ["A"]);
        task.temporal_policy = Some(Default::default());
        task.zone_max_gap_seconds = 0.0;
        let unbridged = task.full_analyze(frames(&points));
        assert_eq!(unbridged.analysis.arm_sequence, ["A", "A"]);
    }

    #[test]
    fn zone_gap_of_one_second_bridges_without_dwell() {
        let mut points = vec![(0.5, 0.0); 3];
        points.extend(vec![(0.0, -0.5); 5]);
        points.extend(vec![(0.5, 0.0); 3]);
        let mut task = crossline_task(0.25);
        assert_eq!(
            task.full_analyze(frames(&points)).analysis.arm_sequence,
            ["A", "A"]
        );
        task.zone_max_gap_seconds = 1.0;
        let result = task.full_analyze(frames(&points));
        assert_eq!(result.analysis.arm_sequence, ["A"]);
        assert_eq!(result.zone_max_gap_seconds, 1.0);
    }

    #[test]
    fn crossline_requires_a_unique_start_edge() {
        let mut task = crossline_task(0.25);
        task.maze.shape.arms[0] =
            Polygon::new(vec![(2.0, -1.0), (2.0, 1.0), (5.0, 3.0), (5.0, -3.0)]);
        assert!(task.validate_temporal_policy().is_err());
    }

    #[test]
    fn crossline_rejects_thresholds_outside_the_unit_interval() {
        let mut task = crossline_task(0.25);
        task.proximal_threshold = 1.5;
        assert!(task.arm_thresholds().unwrap_err().contains("[0, 1]"));
    }

    #[test]
    fn principal_thresholds_come_from_overrides() {
        struct Principal;
        impl crate::TaskOverrides for Principal {
            fn ymaze_principal_thresholds(&self) -> Option<Vec<f64>> {
                Some(vec![0.1, 0.9])
            }
        }
        let mut task = crossline_task(0.25);
        task.apply_overrides(&Principal).unwrap();
        assert_eq!(task.principal_thresholds, vec![0.1, 0.9]);
    }

    #[test]
    fn radial_scalar_predicates_follow_polars_nan_semantics() {
        let at = ArmThreshold::Radial {
            center: (0.0, 0.0),
            min_distance_sq: 1.0,
        };
        assert!(at.past_scalar(1.0, 0.0) && !at.before_scalar(1.0, 0.0));
        assert!(!at.past_scalar(0.5, 0.0) && at.before_scalar(0.5, 0.0));
        assert!(at.past_scalar(f64::NAN, 0.0) && !at.before_scalar(f64::NAN, 0.0));
        let nan_limit = ArmThreshold::Radial {
            center: (0.0, 0.0),
            min_distance_sq: f64::NAN,
        };
        assert!(nan_limit.before_scalar(2.0, 0.0));
    }

    #[test]
    fn missing_coordinate_columns_read_as_missing_points() {
        let df = df! { "a" => [1.0, 2.0] }.unwrap();
        let points = pts(&df, &CoordinateColumns::new("x", "y"));
        assert_eq!(points, vec![None, None]);
    }
}
