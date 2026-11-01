use super::PoseLabelModel;
use bekiper_core::types::TaskType;

fn s(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| v.to_string()).collect()
}

#[test]
fn imen_dlc_required_labels_default_uses_tail() {
    // Task-agnostic default still names `tail` (the OFT/NORT export).
    let labels = PoseLabelModel::ImenDlc.required_source_labels();
    assert!(labels.contains(&"tail"));
    assert!(!labels.contains(&"base_tail"));
}

#[test]
fn imen_dlc_required_labels_for_ymaze_uses_base_tail() {
    // Y-maze DLC export labels the tail keypoint as `base_tail`, not `tail`.
    let labels = PoseLabelModel::ImenDlc.required_source_labels_for_task(TaskType::YMaze);
    assert!(labels.contains(&"base_tail"));
    assert!(!labels.contains(&"tail"));
}

#[test]
fn imen_dlc_required_labels_for_open_field_uses_tail() {
    let labels = PoseLabelModel::ImenDlc.required_source_labels_for_task(TaskType::OpenField);
    assert!(labels.contains(&"tail"));
}

#[test]
fn imen_dlc_missing_for_ymaze_accepts_base_tail_only_pose_file() {
    // Real Imen Y-maze h5 files contain exactly these four labels.
    let labels = s(&["nose", "left_ear", "right_ear", "base_tail"]);
    assert!(
        PoseLabelModel::ImenDlc
            .missing_source_labels_for_task(TaskType::YMaze, &labels)
            .is_empty()
    );
}

#[test]
fn imen_dlc_missing_for_open_field_flags_base_tail_only_pose_file() {
    // Inversely, an OFT pose file with only `base_tail` lacks `tail`.
    let labels = s(&["nose", "left_ear", "right_ear", "base_tail"]);
    let missing =
        PoseLabelModel::ImenDlc.missing_source_labels_for_task(TaskType::OpenField, &labels);
    assert_eq!(missing, vec!["tail"]);
}

#[test]
fn derived_labels_for_imen_dlc_includes_body_center() {
    let labels = PoseLabelModel::ImenDlc.derived_labels();
    assert!(labels.contains(&"center_ear"));
    assert!(labels.contains(&"body_center"));
}

#[test]
fn imen_dlc_model_midpoints_are_task_specific() {
    let open_field = PoseLabelModel::ImenDlc.midpoint_groups_for_task(TaskType::OpenField);
    assert!(open_field.iter().any(|group| {
        group.output_label == "center_ear"
            && group.source_labels == vec!["left_ear".to_string(), "right_ear".to_string()]
    }));
    assert!(open_field.iter().any(|group| {
        group.output_label == "body_center"
            && group.source_labels == vec!["nose".to_string(), "tail".to_string()]
    }));

    let ymaze = PoseLabelModel::ImenDlc.midpoint_groups_for_task(TaskType::YMaze);
    assert!(ymaze.iter().any(|group| {
        group.output_label == "center_ear"
            && group.source_labels == vec!["left_ear".to_string(), "right_ear".to_string()]
    }));
    assert!(
        !ymaze
            .iter()
            .any(|group| group.output_label == "body_center")
    );
}

#[test]
fn model_skeleton_preset_matches_pose_label_model_and_task() {
    assert_eq!(
        PoseLabelModel::SuperanimalTopviewmouse.skeleton_preset_for_task(TaskType::OpenField),
        bekiper_pose::skeleton::SkeletonPreset::MouseTopView
    );
    assert_eq!(
        PoseLabelModel::ImenDlc.skeleton_preset_for_task(TaskType::OpenField),
        bekiper_pose::skeleton::SkeletonPreset::ImenDlcTail
    );
    assert_eq!(
        PoseLabelModel::ImenDlc.skeleton_preset_for_task(TaskType::YMaze),
        bekiper_pose::skeleton::SkeletonPreset::ImenDlcBaseTail
    );
}

#[test]
fn imen_model_skeleton_accepts_derived_midpoint_labels() {
    let labels = s(&["nose", "left_ear", "right_ear", "tail", "center_ear"]);
    let skeleton = PoseLabelModel::ImenDlc
        .skeleton_preset_for_task(TaskType::OpenField)
        .resolve();
    assert!(
        skeleton
            .missing_labels(labels.iter().map(String::as_str))
            .is_empty()
    );

    let ymaze_labels = s(&["nose", "left_ear", "right_ear", "base_tail", "center_ear"]);
    let ymaze_skeleton = PoseLabelModel::ImenDlc
        .skeleton_preset_for_task(TaskType::YMaze)
        .resolve();
    assert!(
        ymaze_skeleton
            .missing_labels(ymaze_labels.iter().map(String::as_str))
            .is_empty()
    );
}
