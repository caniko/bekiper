use crate::analysis::{augment, behaviour, helpers, summary};
mod detect;
mod inputs;
mod inspect;
pub use crate::analysis::staged;
pub(crate) mod pose;

include!("mod/exp_output_dir.rs");
include!("mod/plan_stage_execution.rs");
