// Mirror the audit task-phase split for Y-maze only: the controlled trial
// naming convention uses `_pre_` / `_post_` substrings, and the first match
// wins when a string contains both.
fn ymaze_timepoint(experiment: &str) -> CellValue {
    CellValue::String(
        ymaze_recording_timepoint(experiment)
            .unwrap_or("unknown")
            .into(),
    )
}

/// `pre` / `post` from the trial naming convention, if present.
pub(crate) fn ymaze_recording_timepoint(experiment: &str) -> Option<&'static str> {
    if experiment.contains("_pre_") {
        Some("pre")
    } else if experiment.contains("_post_") {
        Some("post")
    } else {
        None
    }
}
