use super::*;

fn metadata(animal: &str, phase: &str) -> Record {
    json!({"animal_id":animal,"assay":"oft","session_id":"registered","timepoint":"post","phase":phase,
        "group":"G5","five_xfad":"AD","hcar1":"WT","treatment":"lactate","sex":"F","age_treatment_window":"adult",
        "excluded":true,"exclusion_reason":"registered reason","measurement_available":false})
}

fn oft_trial() -> TrialResult {
    TrialResult {
        experiment: "an_uninformative_name".into(),
        task_type: TaskType::OpenField,
        analysis_role: None,
        fps: 30.0,
        meters_per_pixel: 0.001,
        animal_metadata: json!({"animal":13}),
        session_context: json!({"session_id":"registered","timepoint":"after","phase":"session"}),
        summaries: vec![],
        detailed: json!({"total_duration_seconds":300.0,"total_distance_meters":2.0,"total_mean_speed":0.1,"qc_flags":{"source_warning":true}}),
    }
}

fn nort_trial(training: bool) -> TrialResult {
    let mut trial = oft_trial();
    trial.task_type = TaskType::ObjectRecognition;
    trial.session_context = json!({"session_id":"registered","timepoint":"post","phase":if training {"training"} else {"test"}});
    trial.detailed = json!({"protocol":if training {"nort_training"} else {"nort_novelty"},
        "object_labels":["right","left"],"exploration_times":[8.0,12.0],"bout_counts":[1,2],
        "indices":null,"side_bias":null,"object_bias":null,"latency_to_center":null,
        "freezing_seconds":0.0,"total_distance_meters":2.0,"total_mean_speed":0.01,
        "median_speed_observing":null,"median_speed_not_observing":null,
        "total_duration_seconds":200.0,"first_approach_frame":0,"analysis_window_seconds":360.0,"excluded":false});
    trial
}

fn nort_plan(training: bool) -> Record {
    let phase = if training { "training" } else { "test" };
    let mapping = json!({phase: if training {json!({"TA":"left","TB":"right"})} else {json!({"TN":"left","TF":"right"})}});
    json!({"run_mode":"synthetic","synthetic_fixture":true,"inputs":{"units_and_calibration":{"nort":{"time":"s","distance":"m","speed":"m/s"}}},
        "assays":{"nort":{"object_role_and_location_mapping":mapping}},
        "decision_resolutions":{"D10":{"origin":"development_fixture","rationale":"synthetic regression only",
            "approval_or_plan_reference":"F4 handoff regression","scope":["assays"],
            "value":{"assays.nort.object_role_and_location_mapping":mapping}}}})
}

include!("tests/identity.rs");
include!("tests/descriptive.rs");
include!("tests/ymaze.rs");
include!("tests/workbook.rs");
include!("tests/nort.rs");
include!("tests/edges.rs");
