#[cfg(test)]
mod ymaze_override_tests {
    use super::*;

    fn ymaze_perimeters() -> Vec<PerimeterSpec> {
        vec![PerimeterSpec::RadialMaze {
            label: "maze".into(),
            center_vertices: vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            arms: vec![vec![(0.1, -0.1), (1.0, -0.1), (1.0, 0.1), (0.1, 0.1)]],
        }]
    }

    #[test]
    fn ymaze_build_keeps_proximal_heuristics() {
        let names = build_task(
            TaskType::YMaze,
            &ymaze_perimeters(),
            TaskParams::new(30.0),
            None,
            None,
            None,
        )
        .unwrap()
        .heuristics()
        .iter()
        .map(|heuristic| heuristic.name().to_string())
        .collect::<Vec<_>>();

        assert!(names.iter().any(|name| name == "in_arm_A"));
    }
}
#[cfg(test)]
mod build_task_dispatch_tests {
    use super::*;

    fn circle(label: &str, center_x: f64, radius: f64) -> PerimeterSpec {
        PerimeterSpec::Circle {
            label: label.into(),
            center_x,
            center_y: 0.0,
            radius,
        }
    }

    #[test]
    fn build_task_dispatches_each_task_type() {
        let params = TaskParams::new(30.0);
        let nort = build_task(
            TaskType::ObjectRecognition,
            &[circle("object_1", 0.0, 0.02)],
            params.clone(),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(nort.task_type(), TaskType::ObjectRecognition);
        let cheeseboard = build_task(
            TaskType::Cheeseboard,
            &[circle("board", 0.0, 1.0), circle("reward_a", 0.5, 0.05)],
            params,
            None,
            Some("start_area"),
            None,
        )
        .unwrap();
        assert_eq!(cheeseboard.task_type(), TaskType::Cheeseboard);
    }

    #[test]
    fn ymaze_requires_a_radial_maze_perimeter() {
        let error = build_task(
            TaskType::YMaze,
            &[circle("arena", 0.0, 1.0)],
            TaskParams::new(30.0),
            None,
            None,
            None,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("requires a RadialMaze perimeter"), "{error}");
    }
}
