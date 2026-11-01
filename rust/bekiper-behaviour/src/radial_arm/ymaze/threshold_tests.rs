#[cfg(test)]
mod ymaze_threshold_geometry_tests {
    use super::{
        DEFAULT_POSE_GAP_FILL_SECONDS, DEFAULT_PROXIMAL_THRESHOLD, DEFAULT_ZONE_MAX_GAP_SECONDS,
        YMaze, apply_ttl_forward_fill, ttl_frames_from_seconds,
    };
    use bekiper_perimeter::perimeter::Perimeter;
    use bekiper_perimeter::polygon::Polygon;
    use bekiper_perimeter::radial_maze::RadialMaze;
    use polars::prelude::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn synthetic_ymaze() -> YMaze {
        let center = Polygon::new(vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]);
        let arm = Polygon::new(vec![(0.1, 0.0), (0.6, 0.05), (1.1, 0.0), (0.6, -0.05)]);
        let maze = Perimeter::new("maze", RadialMaze::new(center, vec![arm]));
        // Historical mitigation tests use an explicitly non-principal threshold.
        let mut task = YMaze::new(maze, 1.0);
        task.proximal_threshold = 0.30;
        task
    }

    fn synthetic_three_arm_ymaze() -> YMaze {
        let center = Polygon::new(vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]);
        let arm_a = Polygon::new(vec![(0.1, -0.2), (1.1, -0.2), (1.1, 0.2), (0.1, 0.2)]);
        let arm_b = Polygon::new(vec![(-0.2, 0.1), (0.2, 0.1), (0.2, 1.1), (-0.2, 1.1)]);
        let arm_c = Polygon::new(vec![(-1.1, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-1.1, 0.2)]);
        let maze = Perimeter::new("maze", RadialMaze::new(center, vec![arm_a, arm_b, arm_c]));
        let mut task = YMaze::new(maze, 1.0);
        task.proximal_threshold = 0.30;
        task
    }

    fn zone_gap_bridge_input() -> LazyFrame {
        df! {
            "center_ear_x" => [0.9, 0.9, 0.0, 0.9, 0.9],
            "center_ear_y" => [0.0, 0.0, 2.0, 0.0, 0.0],
            "tail_base_x" => [0.9, 0.9, 0.0, 0.9, 0.9],
            "tail_base_y" => [0.0, 0.0, 2.0, 0.0, 0.0],
            "mouse_center_x" => [0.9, 0.9, 0.0, 0.9, 0.9],
            "mouse_center_y" => [0.0, 0.0, 2.0, 0.0, 0.0],
        }
        .unwrap()
        .lazy()
    }

    fn zone_gap_bridge_golden_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/radial_arm/tests/golden/zone_gap_bridge.json")
    }

    fn arm_membership_hysteresis_golden_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/radial_arm/tests/golden/arm_membership_hysteresis.json")
    }

    fn pose_gap_fill_golden_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/radial_arm/tests/golden/pose_gap_fill.json")
    }

    fn truly_raw_arm_sequence_golden_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/radial_arm/tests/golden/truly_raw_arm_sequence.json")
    }

    fn in_arm_a_values(result: &super::YMazeResult) -> Vec<bool> {
        let df = result
            .evaluated_df
            .as_ref()
            .expect("full_analyze should attach evaluated_df");
        df.column("in_arm_A")
            .expect("in_arm_A column")
            .bool()
            .expect("in_arm_A bool column")
            .into_iter()
            .map(|value| value.unwrap_or(false))
            .collect()
    }

    fn filled_values(values: &[Option<f64>], ttl_frames: u32) -> Vec<Option<f64>> {
        let input = df! {
            "value" => values.to_vec(),
        }
        .expect("build value DataFrame")
        .lazy();
        let output = apply_ttl_forward_fill(input, &["value"], ttl_frames)
            .collect()
            .expect("collect filled values");
        output
            .column("value")
            .expect("value column")
            .f64()
            .expect("value column should be f64")
            .into_iter()
            .collect()
    }

    fn pose_gap_fill_input() -> LazyFrame {
        let mut head_x = vec![Some(0.9); 21];
        head_x[10] = None;
        df! {
            "center_ear_x" => head_x,
            "center_ear_y" => vec![Some(0.0); 21],
            "tail_base_x" => vec![Some(0.9); 21],
            "tail_base_y" => vec![Some(0.0); 21],
            "mouse_center_x" => vec![Some(0.9); 21],
            "mouse_center_y" => vec![Some(0.0); 21],
        }
        .unwrap()
        .lazy()
    }

    fn mid_bout_tail_drop_input() -> LazyFrame {
        df! {
            "center_ear_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "center_ear_y" => [0.0; 11],
            "tail_base_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9],
            "tail_base_y" => [0.0; 11],
            "mouse_center_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.55, 0.9, 0.9, 0.9, 0.9, 0.9],
            "mouse_center_y" => [0.0; 11],
        }
        .unwrap()
        .lazy()
    }

    fn genuine_exit_input() -> LazyFrame {
        df! {
            "center_ear_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "center_ear_y" => [0.0; 11],
            "tail_base_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "tail_base_y" => [0.0; 11],
            "mouse_center_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "mouse_center_y" => [0.0; 11],
        }
        .unwrap()
        .lazy()
    }

    fn polygon_exit_input() -> LazyFrame {
        df! {
            "center_ear_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "center_ear_y" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
            "tail_base_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "tail_base_y" => [0.0; 8],
            "mouse_center_x" => [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "mouse_center_y" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0],
        }
        .unwrap()
        .lazy()
    }

    fn hysteresis_golden_input() -> LazyFrame {
        df! {
            "center_ear_x" => [
                0.8, 0.8, 0.8,
                0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0,
                -0.8, -0.8, -0.8,
            ],
            "center_ear_y" => [
                0.0, 0.0, 0.0,
                0.0, 0.0,
                0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8,
                0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            "tail_base_x" => [
                0.8, 0.8, 0.8,
                0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0,
                -0.8, -0.8, -0.8,
            ],
            "tail_base_y" => [
                0.0, 0.0, 0.0,
                0.0, 0.0,
                0.8, 0.8, 0.8, 0.2,
                0.8, 0.8, 0.8,
                0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            "mouse_center_x" => [
                0.8, 0.8, 0.8,
                0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0,
                -0.8, -0.8, -0.8,
            ],
            "mouse_center_y" => [
                0.0, 0.0, 0.0,
                0.0, 0.0,
                0.8, 0.8, 0.8, 0.5,
                0.8, 0.8, 0.8,
                0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        }
        .unwrap()
        .lazy()
    }

    fn all_mitigations_input() -> LazyFrame {
        // The same 17-frame trajectory supplies all three landmarks, with
        // explicit deviations for a pose gap, boundary disagreement and exit.
        let baseline_x = vec![Some(0.9); 17];
        let mut head_x = baseline_x.clone();
        head_x[5] = None;
        head_x[13] = Some(0.0);
        let mut tail_x = baseline_x.clone();
        tail_x[9] = Some(0.2);
        tail_x[13] = Some(0.0);
        let mut center_x = baseline_x;
        center_x[9] = Some(0.55);
        center_x[13] = Some(0.0);
        let mut baseline_y = vec![Some(0.0); 17];
        baseline_y[13] = Some(2.0);
        let head_y = baseline_y.clone();
        let tail_y = baseline_y.clone();
        let center_y = baseline_y;

        df! {
            "center_ear_x" => head_x,
            "center_ear_y" => head_y,
            "tail_base_x" => tail_x,
            "tail_base_y" => tail_y,
            "mouse_center_x" => center_x,
            "mouse_center_y" => center_y,
        }
        .unwrap()
        .lazy()
    }

    include!("threshold_tests/principal_geometry.rs");

    include!("threshold_tests/dwell_views.rs");

    include!("threshold_tests/pose_fill.rs");

    include!("threshold_tests/hysteresis.rs");

    include!("threshold_tests/golden_snapshots.rs");
}
