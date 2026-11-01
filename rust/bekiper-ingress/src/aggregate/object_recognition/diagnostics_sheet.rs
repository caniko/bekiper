use bekiper_behaviour::object_recognition::NortResult as DiagnosticsNortResult;
use bekiper_core::error::Result as DiagnosticsResult;
use bekiper_core::types::{
    ObjectRecognitionProtocol as DiagnosticsObjectRecognitionProtocol,
    TaskType as DiagnosticsTaskType,
};
use indexmap::IndexMap as DiagnosticsIndexMap;
use polars::prelude::DataFrame as DiagnosticsDataFrame;

use super::cell::{
    CellValue as DiagnosticsCellValue, pluck_trial_metadata as diagnostics_pluck_trial_metadata,
};
use super::dataframe::dataframe_from_extracted;
use super::identity::{IDENTITY_COLUMNS, resolve_aggregate_identity};
use super::trial::TrialResult as DiagnosticsTrialResult;

pub(super) const NORT_EXPLORATION_DIAGNOSTICS_SHEET: &str = "nort_exploration_diagnostics";

const DIAGNOSTIC_COLUMNS: &[&str] = &[
    "animal_trial_id",
    "animal_id",
    "trial_id",
    "recording_id",
    "analysis_id",
    "assay",
    "timepoint",
    "stage",
    "variant",
    "animal",
    "experiment",
    "phase",
    "object_label",
    "frames_in_window",
    "frames_pose_ok",
    "frames_outside_object",
    "frames_nose_near",
    "frames_nose_contact",
    "frames_within_angle",
    "frames_contact_or_angle",
    "frames_slow_nose",
    "frames_all_gates_raw",
    "frames_after_tolerance",
    "bouts_pre_filter",
    "bouts_post_filter",
    "bout_lengths_ms_pre",
    "bout_lengths_ms_post",
    "median_nose_speed_per_bout",
    "median_head_angle_per_bout",
    "n_candidate_bouts_pre_merge",
    "n_bouts_post_merge",
    "n_bouts_removed_by_duration_filter",
    "n_bouts_removed_by_speed_filter",
    "n_final_valid_bouts",
    "n_bouts_removed_by_angle_filter",
    "n_candidate_frames_removed_by_angle_criterion",
    "n_candidate_frames_removed_by_posture",
    "n_bouts_rescued_by_contact_override",
];

pub(super) fn build_nort_exploration_diagnostics_dataframe(
    trials: &[DiagnosticsTrialResult],
) -> DiagnosticsResult<Option<DiagnosticsDataFrame>> {
    let mut rows = Vec::new();
    for trial in trials {
        let Some(detail) = nort_detail_for_trial(trial) else {
            continue;
        };
        let Some(phase) = phase_for_protocol(detail.protocol) else {
            continue;
        };
        if detail.exploration_attributions.is_empty() {
            continue;
        }

        let identity_seed = common_row(trial, &detail);
        let identity = resolve_aggregate_identity(trial, &identity_seed)?;
        let identity_columns = identity.columns(trial);

        for (idx, attribution) in detail.exploration_attributions.iter().enumerate() {
            let mut row = DiagnosticsIndexMap::new();
            for column in IDENTITY_COLUMNS {
                row.insert(
                    (*column).to_string(),
                    identity_columns
                        .get(*column)
                        .cloned()
                        .unwrap_or(DiagnosticsCellValue::Null),
                );
            }
            row.insert(
                "animal".into(),
                diagnostics_pluck_trial_metadata(trial, "animal"),
            );
            row.insert(
                "experiment".into(),
                DiagnosticsCellValue::String(trial.experiment.clone()),
            );
            row.insert(
                "phase".into(),
                DiagnosticsCellValue::String(phase.to_string()),
            );
            row.insert(
                "object_label".into(),
                DiagnosticsCellValue::String(match idx {
                    0 => "A".to_string(),
                    1 => "B".to_string(),
                    value => format!("object_{}", value + 1),
                }),
            );
            row.insert(
                "frames_in_window".into(),
                DiagnosticsCellValue::Int(attribution.frames_in_window as i64),
            );
            row.insert(
                "frames_pose_ok".into(),
                DiagnosticsCellValue::Int(attribution.frames_pose_ok as i64),
            );
            row.insert(
                "frames_outside_object".into(),
                DiagnosticsCellValue::Int(attribution.frames_outside_object as i64),
            );
            row.insert(
                "frames_nose_near".into(),
                DiagnosticsCellValue::Int(attribution.frames_nose_near as i64),
            );
            row.insert(
                "frames_nose_contact".into(),
                DiagnosticsCellValue::Int(attribution.frames_nose_contact as i64),
            );
            row.insert(
                "frames_within_angle".into(),
                DiagnosticsCellValue::Int(attribution.frames_within_angle as i64),
            );
            row.insert(
                "frames_contact_or_angle".into(),
                DiagnosticsCellValue::Int(attribution.frames_contact_or_angle as i64),
            );
            row.insert(
                "frames_slow_nose".into(),
                DiagnosticsCellValue::Int(attribution.frames_slow_nose as i64),
            );
            row.insert(
                "frames_all_gates_raw".into(),
                DiagnosticsCellValue::Int(attribution.frames_all_gates_raw as i64),
            );
            row.insert(
                "frames_after_tolerance".into(),
                DiagnosticsCellValue::Int(attribution.frames_after_tolerance as i64),
            );
            row.insert(
                "bouts_pre_filter".into(),
                DiagnosticsCellValue::Int(attribution.bouts_pre_filter as i64),
            );
            row.insert(
                "bouts_post_filter".into(),
                DiagnosticsCellValue::Int(attribution.bouts_post_filter as i64),
            );
            row.insert(
                "bout_lengths_ms_pre".into(),
                DiagnosticsCellValue::String(json_u64_array(&attribution.bout_lengths_ms_pre)),
            );
            row.insert(
                "bout_lengths_ms_post".into(),
                DiagnosticsCellValue::String(json_u64_array(&attribution.bout_lengths_ms_post)),
            );
            row.insert(
                "median_nose_speed_per_bout".into(),
                DiagnosticsCellValue::FloatList(attribution.median_nose_speed_per_bout.clone()),
            );
            row.insert(
                "median_head_angle_per_bout".into(),
                DiagnosticsCellValue::FloatList(attribution.median_head_angle_per_bout.clone()),
            );
            row.insert(
                "n_candidate_bouts_pre_merge".into(),
                DiagnosticsCellValue::Int(attribution.candidate_bouts_pre_merge as i64),
            );
            row.insert(
                "n_bouts_post_merge".into(),
                DiagnosticsCellValue::Int(attribution.bouts_post_merge as i64),
            );
            row.insert(
                "n_bouts_removed_by_duration_filter".into(),
                DiagnosticsCellValue::Int(attribution.bouts_removed_by_duration_filter as i64),
            );
            row.insert(
                "n_bouts_removed_by_speed_filter".into(),
                DiagnosticsCellValue::Int(attribution.bouts_removed_by_speed_filter as i64),
            );
            row.insert(
                "n_final_valid_bouts".into(),
                DiagnosticsCellValue::Int(attribution.final_valid_bouts as i64),
            );
            row.insert(
                "n_bouts_removed_by_angle_filter".into(),
                DiagnosticsCellValue::Int(attribution.bouts_removed_by_angle_filter as i64),
            );
            row.insert(
                "n_candidate_frames_removed_by_angle_criterion".into(),
                DiagnosticsCellValue::Int(
                    attribution.candidate_frames_removed_by_angle_criterion as i64,
                ),
            );
            row.insert(
                "n_candidate_frames_removed_by_posture".into(),
                DiagnosticsCellValue::Int(attribution.candidate_frames_removed_by_posture as i64),
            );
            row.insert(
                "n_bouts_rescued_by_contact_override".into(),
                DiagnosticsCellValue::Int(attribution.bouts_rescued_by_contact_override as i64),
            );
            rows.push(row);
        }
    }

    if rows.is_empty() {
        return Ok(None);
    }

    dataframe_from_extracted(
        DIAGNOSTIC_COLUMNS
            .iter()
            .map(|column| (*column).to_string())
            .collect(),
        rows,
    )
    .map(Some)
}

fn nort_detail_for_trial(trial: &DiagnosticsTrialResult) -> Option<DiagnosticsNortResult> {
    if trial.task_type != DiagnosticsTaskType::ObjectRecognition {
        return None;
    }
    serde_json::from_value(trial.detailed.clone())
        .ok()
        .or_else(|| fallback_detail_for_trial(trial))
}

fn phase_for_protocol(protocol: DiagnosticsObjectRecognitionProtocol) -> Option<&'static str> {
    match protocol {
        DiagnosticsObjectRecognitionProtocol::NortTraining => Some("training"),
        DiagnosticsObjectRecognitionProtocol::NortNovelty => Some("novelty"),
        _ => None,
    }
}

fn json_u64_array(values: &[u64]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}
