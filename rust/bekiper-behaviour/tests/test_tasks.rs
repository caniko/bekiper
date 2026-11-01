use bekiper_behaviour::BehaviouralTask;
use bekiper_behaviour::mapping::all_task_types;
use bekiper_behaviour::object_recognition::ObjectRecognition;
use bekiper_behaviour::radial_arm::YMaze;
use bekiper_behaviour::reward_tracing::Cheeseboard;
use bekiper_core::types::TaskType;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::radial_maze::RadialMaze;
use bekiper_perimeter::rectangle::Rectangle;
use polars::prelude::*;

#[test]
fn all_task_types_contains_expected() {
    let types = all_task_types();
    assert!(types.contains(&TaskType::ObjectRecognition));
    assert!(types.contains(&TaskType::YMaze));
    assert!(types.contains(&TaskType::Cheeseboard));
    assert_eq!(types.len(), 4);
}

#[test]
fn object_recognition_task_type() {
    let task = ObjectRecognition::new(vec![], 30.0);
    assert_eq!(task.task_type(), TaskType::ObjectRecognition);
}

#[test]
fn object_recognition_heuristics_empty_objects() {
    let task = ObjectRecognition::new(vec![], 30.0);
    // 1 heuristic: freezing (no exploration heuristics with 0 objects)
    assert_eq!(task.heuristics().len(), 1);
}

#[test]
fn object_recognition_heuristics_with_circles() {
    let task = ObjectRecognition::new(
        vec![
            Perimeter::new("obj1", ObjectShape::Circle(Circle::new(1.0, 1.0, 0.5))),
            Perimeter::new("obj2", ObjectShape::Circle(Circle::new(3.0, 3.0, 0.5))),
        ],
        30.0,
    );
    // Per object: 1 exploration + 3 qualia (body_prox, olfaction, whiskers) = 4
    // Total: 2*4 + 1 freezing = 9
    assert_eq!(task.heuristics().len(), 9);
}

#[test]
fn object_recognition_heuristics_with_rectangles() {
    let task = ObjectRecognition::new(
        vec![
            Perimeter::new(
                "obj1",
                ObjectShape::Rectangle(Rectangle::new(1.0, 1.0, 0.5, 0.5)),
            ),
            Perimeter::new(
                "obj2",
                ObjectShape::Rectangle(Rectangle::new(3.0, 3.0, 0.5, 0.5)),
            ),
        ],
        30.0,
    );
    // Per object: 1 exploration + 3 qualia = 4; total: 2*4 + 1 freezing = 9
    assert_eq!(task.heuristics().len(), 9);
}

#[test]
fn object_recognition_heuristics_mixed_shapes() {
    let task = ObjectRecognition::new(
        vec![
            Perimeter::new(
                "circle_obj",
                ObjectShape::Circle(Circle::new(1.0, 1.0, 0.5)),
            ),
            Perimeter::new(
                "rect_obj",
                ObjectShape::Rectangle(Rectangle::new(3.0, 3.0, 0.5, 0.5)),
            ),
        ],
        30.0,
    );
    // Per object: 1 exploration + 3 qualia = 4; total: 2*4 + 1 freezing = 9
    assert_eq!(task.heuristics().len(), 9);
}

#[test]
fn y_maze_task_type() {
    let center = Polygon::new(vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]);
    let maze = RadialMaze::new(center, vec![]);
    let task = YMaze::new(Perimeter::new("ymaze", maze), 30.0);
    assert_eq!(task.task_type(), TaskType::YMaze);
    // center zone + freezing = 2 (no arms)
    assert_eq!(task.heuristics().len(), 2);
}

#[test]
fn y_maze_path_crossing_three_arms_produces_nonzero_b_and_c_metrics() {
    let center = Polygon::new(vec![(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)]);
    let arm_a = Polygon::new(vec![(0.2, -0.1), (0.2, 0.1), (1.0, 0.1), (1.0, -0.1)]);
    let arm_b = Polygon::new(vec![(-0.1, 0.2), (0.1, 0.2), (0.1, 1.0), (-0.1, 1.0)]);
    let arm_c = Polygon::new(vec![(-1.0, -0.1), (-1.0, 0.1), (-0.2, 0.1), (-0.2, -0.1)]);
    let maze = RadialMaze::new(center, vec![arm_a, arm_b, arm_c]);
    let task = YMaze::new(Perimeter::new("ymaze", maze), 1.0);

    let df = df! {
        "center_ear_x" => [0.7, 0.7, 0.0, 0.0, -0.7, -0.7],
        "center_ear_y" => [0.0, 0.0, 0.7, 0.7, 0.0, 0.0],
        "tail_base_x" => [0.6, 0.6, 0.0, 0.0, -0.6, -0.6],
        "tail_base_y" => [0.0, 0.0, 0.6, 0.6, 0.0, 0.0],
        "mouse_center_x" => [0.65, 0.65, 0.0, 0.0, -0.65, -0.65],
        "mouse_center_y" => [0.0, 0.0, 0.65, 0.65, 0.0, 0.0],
    }
    .unwrap();

    let result = task.full_analyze(df.lazy());

    assert!(result.analysis.arm_entries[0] > 0);
    assert!(result.analysis.arm_entries[1] > 0);
    assert!(result.analysis.arm_entries[2] > 0);
    assert!(
        result
            .zone_summaries
            .iter()
            .any(|summary| summary.zone_name == "in_arm_B" && summary.duration_seconds > 0.0)
    );
    assert!(
        result
            .zone_summaries
            .iter()
            .any(|summary| summary.zone_name == "in_arm_C" && summary.duration_seconds > 0.0)
    );
}

#[test]
fn cheeseboard_task_type() {
    let task = Cheeseboard {
        start_area: None,
        enclosure: Perimeter::new("board", Circle::new(0.0, 0.0, 5.0)),
        reward_locations: vec![],
    };
    assert_eq!(task.task_type(), TaskType::Cheeseboard);
    assert_eq!(task.heuristics().len(), 1); // enclosure occupancy
}

#[test]
fn analysis_result_serde() {
    let result = bekiper_behaviour::AnalysisResult {
        task_type: TaskType::ObjectRecognition,
        summaries: vec![],
        decompositions: vec![],
        evaluated_df: None,
    };
    let toml_str = toml::to_string(&result).unwrap();
    let deser: bekiper_behaviour::AnalysisResult = toml::from_str(&toml_str).unwrap();
    assert_eq!(deser.task_type, TaskType::ObjectRecognition);
    assert!(deser.evaluated_df.is_none());
}
