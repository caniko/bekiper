//! Descriptive fusion/tolerance comparisons using the same native bout policy.
//! These four masks are inspection outputs, never alternative scored results.

use super::exploration::{BoutFilterSpec, object_exploration_with_timing};
use super::merge::MergeRule;
use crate::tolerance::BoutTiming;
use polars::prelude::*;

pub fn temporal_fusion_states(
    df: &DataFrame,
    rule: &MergeRule,
    timing: &BoutTiming,
    min_seconds: f64,
    max_gap_seconds: f64,
    gap_density: f64,
) -> PolarsResult<Vec<BooleanChunked>> {
    let Some(raw_expr) = rule.to_expr() else {
        return Ok(vec![]);
    };
    let raw = df
        .clone()
        .lazy()
        .select([raw_expr.clone().alias("fusion")])
        .collect()?;
    let raw_mask = raw.column("fusion")?.bool()?.clone();
    let tolerate = |frame: &DataFrame, column: &str| {
        object_exploration_with_timing(
            frame,
            column,
            timing,
            min_seconds,
            max_gap_seconds,
            gap_density,
            BoutFilterSpec::unfiltered(),
        )
        .2
    };
    let after = tolerate(&raw, "fusion");
    let mut within = df.clone();
    let mut columns = rule
        .require
        .iter()
        .chain(&rule.any_of)
        .chain(&rule.exclude)
        .collect::<Vec<_>>();
    columns.sort();
    columns.dedup();
    for column in columns {
        let mut mask = tolerate(df, column);
        mask.rename(column.as_str().into());
        within.with_column(mask.into_column())?;
    }
    let within = within.lazy().select([raw_expr.alias("fusion")]).collect()?;
    let both = tolerate(&within, "fusion");
    Ok(vec![
        raw_mask,
        after,
        within.column("fusion")?.bool()?.clone(),
        both,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use bekiper_core::video::FrameTiming;

    #[test]
    fn channel_tolerance_and_fusion_tolerance_have_distinct_lineage() {
        let df = df!("left" => [true, true, false, false, false],
                     "right" => [false, false, true, true, false])
        .unwrap();
        let clock = BoutTiming::frames(&FrameTiming::cfr(10.0), 5).unwrap();
        let states = temporal_fusion_states(
            &df,
            &MergeRule {
                any_of: vec!["left".into(), "right".into()],
                ..Default::default()
            },
            &clock,
            0.4,
            0.0,
            0.0,
        )
        .unwrap();
        assert_eq!(
            states
                .iter()
                .map(|m| m.sum().unwrap_or(0))
                .collect::<Vec<_>>(),
            vec![4, 4, 0, 0]
        );
        let empty =
            temporal_fusion_states(&df, &MergeRule::default(), &clock, 0.4, 0.0, 0.0).unwrap();
        assert!(empty.is_empty());
    }
}
