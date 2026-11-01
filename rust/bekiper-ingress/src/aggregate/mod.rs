//! Consolidated cross-trial data export.
//!
//! Parses per-trial `result.toml` files into a shared [`TrialResult`] struct
//! and dispatches on [`bekiper_core::types::TaskType`] to per-task [`Experiment`] implementations
//! that pluck the right fields into an ordered, flat row for the consolidated
//! spreadsheet.
//!
//! This module contains the shared infrastructure, per-task extractors for
//! NORT/OUL, Open Field, Y-maze, and Cheeseboard, plus a generic fallback
//! that flattens arbitrary `detailed` JSON into dot-pathed columns. Aggregate
//! sheets preserve one row per trial/session-period; animal-level statistics
//! belong in explicit downstream reporting, not this export stage.

mod cell;
mod cheeseboard;
mod dataframe;
mod experiment;
mod generic;
mod identity;
pub(crate) use identity::session_timepoint;
pub(crate) mod metadata;
mod nort;
pub mod object_recognition;
mod oft;
mod pipeline;
mod post_analysis;
mod run;
mod trial;
mod ymaze;

pub use cell::CellValue;
pub use cheeseboard::CheeseboardExperiment;
pub use experiment::{Experiment, ExperimentId, SheetKey, VariantTag, experiment_for_trial};
pub use generic::GenericExperiment;
pub use metadata::{metadata_workspace_bytes, metadata_xml_bytes};
pub use nort::NortExperiment;
pub use object_recognition::ObjectRecognitionExperiment;
pub use oft::OftExperiment;
pub use pipeline::{AggregateSheet, build_sheet_dataframes};
pub use post_analysis::{
    export_post_analysis_trials, export_post_analysis_trials_with_cancel, post_analysis_output_path,
};
pub use run::{
    aggregate_output_path_for_config, aggregate_state_path_for_config, export_aggregate_trials,
    export_aggregate_trials_with_cancel, is_aggregate_fresh, run_aggregate,
};
pub use trial::{TrialResult, parse_recipe_from_experiment};
pub use ymaze::YMazeExperiment;

#[cfg(test)]
mod tests;
