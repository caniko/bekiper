use bekiper_behaviour::radial_arm::YMaze;
use bekiper_core::video::FrameTiming;
use bekiper_feature::tolerance::{DurationComparison, TemporalPolicy};
use bekiper_perimeter::{perimeter::Perimeter, polygon::Polygon, radial_maze::RadialMaze};
use bekiper_post_analysis::events;
use polars::prelude::*;
use serde_json::{Value, json};

fn geometry() -> (Perimeter<RadialMaze>, Value) {
    let center = vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)];
    let arms = vec![
        vec![(0.1, -0.2), (1.1, -0.2), (1.1, 0.2), (0.1, 0.2)],
        vec![(-0.2, 0.1), (0.2, 0.1), (0.2, 1.1), (-0.2, 1.1)],
        vec![(-1.1, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-1.1, 0.2)],
    ];
    let document = json!({"version":events::GEOMETRY,"canonical_vertices":true,"center":center,
        "arms":{"A":arms[0],"B":arms[1],"C":arms[2]}});
    (
        Perimeter::new(
            "maze",
            RadialMaze::new(
                Polygon::new(center),
                arms.into_iter().map(Polygon::new).collect(),
            ),
        ),
        document,
    )
}

#[test]
fn ymaze_default_accepts_exactly_one_second_and_rejects_shorter_visits() {
    for threshold in [25, 45, 65] {
        let (maze, _) = geometry();
        let x = [0.0, 0.9, 0.0, 0.9, 0.9, 0.0];
        let input = df! {
            "center_ear_x" => x, "center_ear_y" => [0.0;6],
            "tail_base_x" => x, "tail_base_y" => [0.0;6],
            "mouse_center_x" => x, "mouse_center_y" => [0.0;6]
        }
        .unwrap()
        .lazy();
        let mut task = YMaze::new(maze, 2.0);
        task.proximal_threshold = f64::from(threshold) / 100.0;
        let native = task.full_analyze(input);
        assert_eq!(native.analysis.arm_sequence, ["A"]);
        assert_eq!(native.analysis.raw_arm_sequence, ["A", "A"]);
        let provenance = native.scoring_provenance.unwrap();
        assert_eq!(provenance["entry_frames"]["in_arm_A"], json!([3]));
        assert_eq!(provenance["minimum_dwell_seconds"], 1.0);
        assert_eq!(
            provenance["temporal_policy"]["duration_comparison"],
            "at_least"
        );
        assert_eq!(provenance["dwell_bouts"]["in_arm_A"][0]["accepted"], false);
        assert_eq!(
            provenance["dwell_bouts"]["in_arm_A"][1]["crossing_timestamp_s"],
            1.5
        );
        assert_eq!(
            provenance["dwell_bouts"]["in_arm_A"][1]["qualification_timestamp_s"],
            2.5
        );
        assert_eq!(native.zone_summaries[0].entries, 1);
        assert_eq!(native.zone_summaries[0].duration_seconds, 1.0);
    }
}

#[test]
fn native_and_reconstructed_dwell_agree_at_every_threshold_and_boundary() {
    for threshold in [25, 45, 65] {
        for (comparison, expected_frames) in [
            (DurationComparison::AtLeast, vec![1, 4]),
            (DurationComparison::GreaterThan, vec![4]),
        ] {
            let (maze, geometry) = geometry();
            let x = [0.0, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9];
            let input = df! {
                "center_ear_x" => x, "center_ear_y" => [0.0;7],
                "tail_base_x" => x, "tail_base_y" => [0.0;7],
                "mouse_center_x" => x, "mouse_center_y" => [0.0;7]
            }
            .unwrap()
            .lazy();
            let policy = TemporalPolicy {
                min_duration_seconds: 1.0,
                duration_comparison: comparison,
                ..Default::default()
            };
            let mut task = YMaze::new(maze, 2.0);
            task.proximal_threshold = f64::from(threshold) / 100.0;
            task.temporal_policy = Some(policy);
            let native = task.full_analyze(input);
            let samples: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    json!({"timestamp_s":i as f64 / 2.0,
                "head_center_x":x,"head_center_y":0,"tail_base_x":x,"tail_base_y":0})
                })
                .collect();
            let reconstructed = events::ymaze_coordinate_entries_with_policy(
                &samples,
                &geometry,
                threshold,
                &policy,
                Some(3.5),
            )
            .unwrap();
            let provenance = native.scoring_provenance.as_ref().unwrap();
            assert_eq!(
                provenance["entry_frames"]["in_arm_A"],
                json!(expected_frames)
            );
            assert_eq!(provenance["raw_entry_frames"]["in_arm_A"], json!([1, 4]));
            assert_eq!(
                native.analysis.arm_sequence,
                vec!["A"; expected_frames.len()]
            );
            assert_eq!(
                reconstructed["valid_arm_entry_sequence"],
                json!(native.analysis.arm_sequence)
            );
            assert_eq!(reconstructed["raw_arm_entry_sequence"], json!(["A", "A"]));
            assert_eq!(
                reconstructed["removed_short_entries"],
                json!(2 - expected_frames.len())
            );
            assert_eq!(
                native.zone_summaries[0].entries as usize,
                expected_frames.len()
            );
            assert_eq!(
                native.zone_summaries[0].duration_seconds,
                if comparison == DurationComparison::AtLeast {
                    2.5
                } else {
                    1.5
                }
            );
            assert_eq!(
                native.zone_summaries[0].latency_seconds,
                Some(if comparison == DurationComparison::AtLeast {
                    0.5
                } else {
                    2.0
                })
            );
            let verified = events::ymaze_provenance_with_policy(&json!({"threshold_pct":threshold,"scoring_provenance":provenance}),
                &json!({"head_center":{"x":"center_ear_x","y":"center_ear_y"},"tail_base":{"x":"tail_base_x","y":"tail_base_y"}}), &policy).unwrap();
            assert_eq!(verified["status"], "verified_clarified_entry_definition");
            let mismatch = events::ymaze_provenance_with_policy(
                &json!({"scoring_provenance":provenance}),
                &json!({}),
                &TemporalPolicy::default(),
            )
            .unwrap();
            assert_eq!(
                mismatch["status"],
                "requires_regeneration_or_registered_amendment"
            );
            assert!(
                events::ymaze_coordinate_entries_with_policy(
                    &samples, &geometry, threshold, &policy, None
                )
                .is_err()
            );
        }
    }
}

#[test]
fn variable_rate_timing_missing_pose_and_trim_keep_original_frame_lineage() {
    let (maze, geometry) = geometry();
    let x = [None, Some(0.9), Some(0.9), None, Some(0.9), Some(0.9)];
    let timestamps = vec![5.0, 5.25, 5.5, 6.5, 6.75, 7.0];
    let input = df! {"center_ear_x"=>x,"center_ear_y"=>[Some(0.0);6],"tail_base_x"=>x,"tail_base_y"=>[Some(0.0);6],
        "mouse_center_x"=>[0.9;6],"mouse_center_y"=>[0.0;6]}.unwrap().lazy();
    let policy = TemporalPolicy {
        min_duration_seconds: 1.0,
        ..Default::default()
    };
    let mut task = YMaze::new(maze, 4.0);
    task.temporal_policy = Some(policy);
    let timing = FrameTiming {
        fps: 4.0,
        timestamps: timestamps.clone(),
        is_vfr: true,
    };
    let native = task.full_analyze_timed(input, &timing);
    let samples: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, x)| {
            json!({"timestamp_s":timestamps[i],
        "head_center_x":x,"head_center_y":0,"tail_base_x":x,"tail_base_y":0})
        })
        .collect();
    let reconstructed =
        events::ymaze_coordinate_entries_with_policy(&samples, &geometry, 25, &policy, Some(7.25))
            .unwrap();
    assert_eq!(native.analysis.arm_sequence, ["A"]);
    assert_eq!(native.zone_summaries[0].duration_seconds, 1.25);
    assert_eq!(
        native.scoring_provenance.as_ref().unwrap()["entry_frames"]["in_arm_A"],
        json!([1])
    );
    assert_eq!(
        native.scoring_provenance.as_ref().unwrap()["dwell_bouts"]["in_arm_A"][0]["qualification_timestamp_s"],
        6.5
    );
    assert_eq!(reconstructed["valid_arm_entry_sequence"], json!(["A"]));
    assert_eq!(reconstructed["removed_short_entries"], 1);
    assert_eq!(
        reconstructed["event_debug"][9]["reason"],
        "tracking_missing"
    );
    assert_eq!(reconstructed["event_debug"][12]["reason"], "minimum_dwell");
}

#[test]
fn trimmed_cfr_preserves_crossing_and_zero_dwell_qualifies_immediately() {
    for minimum in [0.0, 1.0] {
        for comparison in [DurationComparison::AtLeast, DurationComparison::GreaterThan] {
            let (maze, geometry) = geometry();
            let x = [None, None, Some(0.9), Some(0.9), Some(0.0)];
            let input = df! {
                "center_ear_x"=>x,"center_ear_y"=>[0.0;5],
                "tail_base_x"=>x,"tail_base_y"=>[0.0;5],
                "mouse_center_x"=>[0.9;5],"mouse_center_y"=>[0.0;5]
            }
            .unwrap()
            .lazy();
            let policy = TemporalPolicy {
                min_duration_seconds: minimum,
                duration_comparison: comparison,
                ..Default::default()
            };
            let mut task = YMaze::new(maze, 2.0);
            task.temporal_policy = Some(policy);
            let native = task.full_analyze(input);
            let provenance = native.scoring_provenance.as_ref().unwrap();
            assert_eq!(provenance["analysis_start_frame"], 2);
            assert_eq!(provenance["analysis_start_timestamp_s"], 1.0);
            assert_eq!(provenance["sample_end_timestamp_s"], 2.5);
            let bout = &provenance["dwell_bouts"]["in_arm_A"][0];
            assert_eq!(bout["start_frame"], 2);
            assert_eq!(bout["crossing_timestamp_s"], 1.0);
            assert_eq!(bout["duration_s"], 1.0);
            let accepted = minimum == 0.0 || comparison == DurationComparison::AtLeast;
            assert_eq!(bout["accepted"], accepted);
            assert_eq!(
                provenance["entry_frames"]["in_arm_A"],
                if accepted { json!([2]) } else { json!([]) }
            );
            let samples: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    json!({
                        "timestamp_s":i as f64 / 2.0,"head_center_x":x,"head_center_y":0,
                        "tail_base_x":x,"tail_base_y":0
                    })
                })
                .collect();
            let reconstructed = events::ymaze_coordinate_entries_with_policy(
                &samples,
                &geometry,
                25,
                &policy,
                Some(2.5),
            )
            .unwrap();
            assert_eq!(
                reconstructed["valid_arm_entry_sequence"],
                json!(native.analysis.arm_sequence)
            );
            assert_eq!(
                bout["qualification_timestamp_s"],
                reconstructed["dwell_bouts"][0]["qualification_timestamp_s"]
            );
            if minimum == 0.0 {
                assert_eq!(
                    bout["qualification_timestamp_s"],
                    bout["crossing_timestamp_s"]
                );
            }
        }
    }
}
