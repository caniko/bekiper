fn is_native_ymaze_comparison(trial: &TrialResult) -> bool {
    trial.task_type == TaskType::YMaze
        && trial.analysis_role.as_deref().map_or_else(
            || {
                trial
                    .experiment
                    .rsplit_once("__")
                    .is_some_and(|(_, variant)| {
                        ["compare_0pct_dwell1s", "compare_0pct_dwell0s"].contains(&variant)
                    })
            },
            // Native task roles use `y_maze`; historical aggregate roles used
            // `ymaze`. Preserve the original role in published lineage.
            |role| {
                [
                    "y_maze__compare_0pct_dwell1s",
                    "y_maze__compare_0pct_dwell0s",
                    "ymaze__compare_0pct_dwell1s",
                    "ymaze__compare_0pct_dwell0s",
                ]
                .contains(&role)
            },
        )
}

fn complete_registered_session_key(row: &mut Record, metadata: &[Record]) -> Result<()> {
    // A session ID must be supplied or explicitly mapped by the plan. Missing
    // phase/timepoint can then come from its unique registered metadata row.
    if row["session_id"].is_null() || (!row["timepoint"].is_null() && !row["phase"].is_null()) {
        return Ok(());
    }
    let candidates: Vec<_> = metadata
        .iter()
        .filter(|meta| {
            ["animal_id", "assay", "session_id"]
                .iter()
                .all(|field| row[*field] == meta[*field])
                && ["timepoint", "phase"]
                    .iter()
                    .all(|field| row[*field].is_null() || row[*field] == meta[*field])
        })
        .collect();
    require(
        candidates.len() == 1,
        format!(
            "native session mapping needs exactly one registered phase/timepoint; found {} for animal {}, assay {}, session {}, timepoint {}, phase {}",
            candidates.len(),
            row["animal_id"],
            row["assay"],
            row["session_id"],
            row["timepoint"],
            row["phase"]
        ),
    )?;
    for field in ["timepoint", "phase"] {
        if row[field].is_null() {
            row[field] = candidates[0][field].clone();
        }
    }
    Ok(())
}
