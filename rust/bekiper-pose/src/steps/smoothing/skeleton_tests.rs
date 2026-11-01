#[cfg(test)]
mod skeleton_constraint_tests {
    use super::*;

    fn cols(label: &str) -> KalmanSmootherColumns {
        KalmanSmootherColumns {
            label: label.into(),
            x_col: format!("{label}_x"),
            y_col: format!("{label}_y"),
            likelihood_col: Some(format!("{label}_likelihood")),
        }
    }

    fn line(labels: &[&str], edges: &[(&str, &str)]) -> Skeleton {
        Skeleton::new(
            "line",
            labels
                .iter()
                .map(|l| crate::skeleton::Node::new(*l))
                .collect(),
            edges
                .iter()
                .map(|(s, t)| crate::skeleton::Edge::new(*s, *t))
                .collect(),
        )
    }

    #[test]
    fn short_gaps_need_trusted_frames_on_both_sides_within_the_limit() {
        let trusted = [true, false, false, false, true];
        // Every frame of a bounded three-frame gap is short, whichever
        // direction the gap boundary search walks.
        for row in 1..=3 {
            assert!(is_short_gap(&trusted, row, 3), "row {row}");
            assert!(!is_short_gap(&trusted, row, 2), "row {row}");
        }
        assert!(!is_short_gap(&trusted, 0, 3));
        // Leading and trailing gaps are never bounded.
        assert!(!is_short_gap(&[false, true], 0, 10));
        assert!(!is_short_gap(&[true, false, false], 2, 10));
    }

    #[test]
    fn projection_skips_missing_or_degenerate_pairs() {
        let series = |mx: Option<f64>, ax: Option<f64>| {
            vec![
                PositionSeries {
                    xs: vec![mx],
                    ys: vec![Some(0.0)],
                },
                PositionSeries {
                    xs: vec![ax],
                    ys: vec![Some(0.0)],
                },
            ]
        };
        for (mut case, length) in [
            (series(Some(3.0), None), 1.0),      // anchor missing
            (series(Some(3.0), Some(3.0)), 1.0), // coincident points
            (series(Some(3.0), Some(0.0)), 0.0), // zero bone length
        ] {
            let before = case[0].xs[0];
            project_endpoint(&mut case, 0, 1, 0, length, 1.0);
            assert_eq!(case[0].xs[0], before);
        }
        let mut moved = series(Some(4.0), Some(0.0));
        project_endpoint(&mut moved, 0, 1, 0, 2.0, 0.5);
        assert_eq!(moved[0].xs[0], Some(3.0));
    }

    #[test]
    fn baselines_skip_edges_without_columns_or_trusted_frames() {
        let df = df! {
            "a_x" => [0.0, 0.0], "a_y" => [0.0, 0.0], "a_likelihood" => [0.9, 0.9],
            "b_x" => [3.0, 3.0], "b_y" => [4.0, 4.0], "b_likelihood" => [0.0, 0.0],
            "c_x" => [0.0, 8.0], "c_y" => [8.0, 0.0], "c_likelihood" => [0.9, 0.9],
        }
        .unwrap();
        let skeleton = line(&["a", "b", "c", "d"], &[("a", "b"), ("a", "c"), ("a", "d")]);
        let baselines =
            fit_skeleton_edge_lengths(&df, &skeleton, &[cols("a"), cols("b"), cols("c")], 0.5)
                .unwrap();
        // a-b has no trusted frame, a-d has no columns; only a-c is fitted.
        assert_eq!(baselines.len(), 1);
        assert_eq!(baselines[&("a".to_string(), "c".to_string())], 8.0);
    }

    #[test]
    fn constraints_move_whichever_endpoint_is_in_a_short_gap() {
        // `a` sits in a short gap on row 1 while `b` is trusted; the a-b
        // edge must project the source. Edge b-c has no fitted baseline.
        let mut df = df! {
            "a_x" => [0.0, 10.0, 0.0], "a_y" => [0.0, 0.0, 0.0],
            "b_x" => [2.0, 2.0, 2.0], "b_y" => [0.0, 0.0, 0.0],
            "c_x" => [5.0, 5.0, 5.0], "c_y" => [0.0, 0.0, 0.0],
        }
        .unwrap();
        let skeleton = line(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
        let trusted = vec![
            TrustedMask {
                label: "a".into(),
                trusted: vec![true, false, true],
            },
            TrustedMask {
                label: "b".into(),
                trusted: vec![true; 3],
            },
            TrustedMask {
                label: "c".into(),
                trusted: vec![true; 3],
            },
        ];
        let baselines = [(("a".to_string(), "b".to_string()), 2.0)].into();
        let columns = [cols("a"), cols("b"), cols("c")];
        apply_skeleton_constraints(
            &mut df, &skeleton, &columns, &trusted, &baselines, 5, 1, 1.0,
        )
        .unwrap();
        let a_x = df.column("a_x").unwrap().f64().unwrap();
        assert_eq!(a_x.get(1), Some(4.0));
        assert_eq!(a_x.get(0), Some(0.0));
        assert_eq!(median_f64(&[]), None);
    }

    #[test]
    fn baselines_trust_missing_likelihood_columns_and_skip_null_positions() {
        let df = df! {
            "a_x" => [0.0; 4], "a_y" => [0.0; 4], "a_likelihood" => [0.9; 4],
            "e_x" => [Some(3.0), Some(4.0), Some(5.0), None], "e_y" => [0.0; 4],
        }
        .unwrap();
        let mut e = cols("e");
        e.likelihood_col = None;
        let skeleton = line(&["a", "e"], &[("a", "e")]);
        let baselines = fit_skeleton_edge_lengths(&df, &skeleton, &[cols("a"), e], 0.5).unwrap();
        // Rows 0..3 are trusted by default; the null row is skipped.
        assert_eq!(baselines[&("a".to_string(), "e".to_string())], 4.0);
    }
}
