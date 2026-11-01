//! Configurable boolean combination of heuristic columns.
//!
//! Replaces the legacy PEG grammar parser (`parse_heuristic_merge_equation`)
//! with a structured, type-safe combinator that's configurable via TOML:
//!
//! ```toml
//! [task_params.observation]
//! require = ["olfaction", "proximity"]        # AND: all must be true
//! any_of  = ["whiskers"]                      # OR: at least one must be true
//! exclude = ["body_proximity"]                # NOT: none may be true
//! ```
//!
//! The final observation mask is:
//!   (ALL of `require`) AND (ANY of `any_of`, if non-empty) AND (NONE of `exclude`)

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;

use crate::heuristic::Heuristic;

/// Structured boolean combination rule for merging heuristic outputs.
///
/// This is the improved replacement for the legacy string-based DSL.
/// It supports the three most common combination patterns without
/// needing a parser or runtime evaluation of arbitrary expressions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeRule {
    /// Columns that must ALL be true (logical AND).
    #[serde(default)]
    pub require: Vec<String>,
    /// At least ONE of these columns must be true (logical OR).
    /// Ignored if empty.
    #[serde(default)]
    pub any_of: Vec<String>,
    /// NONE of these columns may be true (logical NOT).
    #[serde(default)]
    pub exclude: Vec<String>,
}

impl MergeRule {
    /// Build a Polars expression that evaluates this merge rule.
    ///
    /// Returns `None` if the rule is empty (no columns specified).
    pub fn to_expr(&self) -> Option<Expr> {
        let mut parts: Vec<Expr> = Vec::new();

        // require: AND all
        for col_name in &self.require {
            parts.push(col(col_name));
        }

        // any_of: OR any
        if !self.any_of.is_empty() {
            let or_parts: Vec<Expr> = self.any_of.iter().map(col).collect();
            parts.push(
                polars::prelude::any_horizontal(or_parts)
                    .expect("any_of requires at least one column"),
            );
        }

        // exclude: NOT any
        for col_name in &self.exclude {
            parts.push(col(col_name).not());
        }

        if parts.is_empty() {
            return None;
        }

        Some(
            polars::prelude::all_horizontal(parts)
                .expect("merge rule requires at least one column"),
        )
    }

    /// Whether this rule has any columns specified.
    pub fn is_empty(&self) -> bool {
        self.require.is_empty() && self.any_of.is_empty() && self.exclude.is_empty()
    }
}

/// Heuristic that combines pre-evaluated boolean columns using a `MergeRule`.
///
/// All referenced columns must already exist in the LazyFrame (i.e., the
/// constituent heuristics must have been evaluated first).
pub struct MergedHeuristic {
    pub name: String,
    pub rule: MergeRule,
    pub decomposition: Option<HeuristicDecompositionSpec>,
}

impl Heuristic for MergedHeuristic {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        match self.rule.to_expr() {
            Some(expr) => lf.with_column(expr.alias(&self.name)),
            None => lf,
        }
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        self.decomposition.clone()
    }
}
