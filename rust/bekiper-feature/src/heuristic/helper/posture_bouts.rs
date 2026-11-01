use polars::prelude::*;

use crate::heuristic::Heuristic;

/// Helper heuristic: true on frames inside a run of any source column that
/// lasts at least that source's minimum duration (e.g. grooming ≥1 s,
/// rearing ≥0.5 s). NORT uses it to void nose-contact rescue in posture frames.
pub struct PostureBoutHeuristic {
    /// Output column.
    pub name: String,
    /// `(boolean source column, minimum run seconds)`.
    pub sources: Vec<(String, f64)>,
    pub fps: f64,
}

impl Heuristic for PostureBoutHeuristic {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        // ponytail: run length in frames at the nominal fps; use per-frame
        // timestamps (TemporalPolicy) if a VFR NORT dataset needs exact seconds.
        let bout = |(source, seconds): &(String, f64)| {
            let frames = (seconds * self.fps - 1e-9).ceil().max(1.0) as u32;
            let on = col(source.as_str()).fill_null(lit(false));
            on.clone().and(len().over([on.rle_id()]).gt_eq(lit(frames)))
        };
        let any = self
            .sources
            .iter()
            .map(bout)
            .reduce(|a, b| a.or(b))
            .unwrap_or(lit(false));
        lf.with_column(any.alias(self.name.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_runs_reaching_their_minimum_duration_count() {
        let frame = df! {
            "grooming" => [true, true, true, false, true, true, false, false],
            "rearing" => [Some(false), Some(false), Some(false), Some(false), Some(false), Some(false), Some(true), None],
        }
        .unwrap();
        let heuristic = PostureBoutHeuristic {
            name: "posture".into(),
            // 10 fps: grooming needs 3 frames (0.3 s), rearing 1 frame.
            sources: vec![("grooming".into(), 0.3), ("rearing".into(), 0.1)],
            fps: 10.0,
        };
        let out = heuristic.evaluate(frame.lazy()).collect().unwrap();
        let posture: Vec<_> = out["posture"]
            .bool()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            posture,
            vec![true, true, true, false, false, false, true, false]
        );
        assert_eq!(heuristic.name(), "posture");

        let none = PostureBoutHeuristic {
            name: "posture".into(),
            sources: vec![],
            fps: 10.0,
        };
        let out = none
            .evaluate(df! {"x" => [1.0]}.unwrap().lazy())
            .collect()
            .unwrap();
        assert_eq!(out["posture"].bool().unwrap().get(0), Some(false));
    }
}
