//! Scientific decisions are registered, dependency-scoped inputs; unavailable
//! results are explicit and never replaced with invented observations.
pub mod contracts;
pub mod events;
pub mod export;
mod fingerprint;
pub mod fixtures;
pub mod genotype;
pub mod inputs;
pub mod maintenance;
pub mod metrics;
pub mod pipeline;
pub mod plotting;
pub mod report;
pub mod source;
mod source_tree;
pub mod statistics;
pub mod verification;
pub use contracts::{Record, RunOptions};
pub use source::{SOURCE_RECIPE, YMAZE_AMENDMENT, code_identity};
