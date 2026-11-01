impl BehaviouralTask for Cheeseboard {
    fn task_type(&self) -> TaskType {
        TaskType::Cheeseboard
    }

    fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        let mut heuristics: Vec<Box<dyn Heuristic>> =
            Vec::with_capacity(2 + self.reward_locations.len());
        if let Some(spec) = &self.start_area {
            heuristics.push(Box::new(ConfinementSpecHeuristic {
                name: START_AREA_COLUMN.into(),
                spec: spec.clone(),
                coords: CoordinateColumns::new("mouse_center_x", "mouse_center_y"),
            }));
        }
        heuristics.push(Box::new(ConfinementHeuristic {
            name: ENCLOSURE_COLUMN.into(),
            perimeter: self.enclosure.clone(),
            coords: CoordinateColumns::new("mouse_center_x", "mouse_center_y"),
        }));

        for (i, reward) in self.reward_locations.iter().enumerate() {
            heuristics.push(Box::new(ConfinementHeuristic {
                name: reward_column_name(i),
                perimeter: reward.clone(),
                coords: CoordinateColumns::new("mouse_center_x", "mouse_center_y"),
            }));
        }

        heuristics
    }

    fn analyze_detailed(
        &self,
        lf: LazyFrame,
        timing: &FrameTiming,
    ) -> (crate::AnalysisResult, Option<TaskDetail>) {
        let heuristics = self.heuristics();
        let decompositions = crate::decompositions_from_heuristics(&heuristics);
        let result = self.full_analyze(lf, timing);
        let names = self.heuristic_names();
        let summary_names: Vec<&str> = names.iter().map(|name| name.as_str()).collect();
        let summaries = summaries_from_df(
            result
                .evaluated_df
                .as_ref()
                .expect("evaluated_df is always Some after full_analyze"),
            &summary_names,
            timing,
        );
        let analysis = crate::AnalysisResult {
            task_type: self.task_type(),
            summaries,
            decompositions,
            evaluated_df: result.evaluated_df.clone(),
        };
        (analysis, Some(TaskDetail::Cheeseboard(result)))
    }
}

struct ConfinementSpecHeuristic {
    name: String,
    spec: PerimeterSpec,
    coords: CoordinateColumns,
}

impl Heuristic for ConfinementSpecHeuristic {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let mask = spec_confinement_expr(&self.spec, &self.coords).fill_null(lit(false));
        let component_col = diag_column_name(self.name(), "in_zone");
        lf.with_columns([mask.clone().alias(&component_col), mask.alias(self.name())])
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name.clone(),
            result_column: self.name.clone(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.coords.x)],
            perimeter_label: Some(self.spec.label().to_string()),
            components: vec![decomposition_component(
                "in_zone",
                diag_column_name(self.name(), "in_zone"),
                vec![coordinate_label_from_xy(&self.coords.x)],
                Some(self.spec.label().to_string()),
            )],
            show_final_result: false,
        })
    }
}

struct ConfinementHeuristic {
    name: String,
    perimeter: Perimeter<Circle>,
    coords: CoordinateColumns,
}

impl Heuristic for ConfinementHeuristic {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let mask = self
            .perimeter
            .confinement_mask(&self.coords)
            .fill_null(lit(false));
        let component_col = diag_column_name(self.name(), "in_zone");
        lf.with_columns([mask.clone().alias(&component_col), mask.alias(self.name())])
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name.clone(),
            result_column: self.name.clone(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.coords.x)],
            perimeter_label: Some(self.perimeter.label.clone()),
            components: vec![decomposition_component(
                "in_zone",
                diag_column_name(self.name(), "in_zone"),
                vec![coordinate_label_from_xy(&self.coords.x)],
                Some(self.perimeter.label.clone()),
            )],
            show_final_result: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_df(x: &[f64], y: &[f64]) -> DataFrame {
        let cols = vec![
            Float64Chunked::from_slice("mouse_center_x".into(), x).into_column(),
            Float64Chunked::from_slice("mouse_center_y".into(), y).into_column(),
        ];
        DataFrame::new(x.len(), cols).unwrap()
    }

    #[test]
    fn heuristics_cover_enclosure_and_rewards() {
        let task = Cheeseboard {
            start_area: Some(PerimeterSpec::Rectangle {
                label: "start_area".into(),
                center_x: 5.5,
                center_y: 0.0,
                width: 1.0,
                height: 1.0,
            }),
            enclosure: Perimeter::new("board", Circle::new(0.0, 0.0, 5.0)),
            reward_locations: vec![
                Perimeter::new("reward_a", Circle::new(3.0, 0.0, 0.5)),
                Perimeter::new("reward_b", Circle::new(-3.0, 0.0, 0.5)),
            ],
        };

        assert_eq!(task.heuristics().len(), 4);
    }

    #[test]
    fn full_analyze_derives_reward_latency_and_search_metrics() {
        let task = Cheeseboard {
            start_area: Some(PerimeterSpec::Rectangle {
                label: "start_area".into(),
                center_x: 6.0,
                center_y: 0.0,
                width: 1.0,
                height: 1.0,
            }),
            enclosure: Perimeter::new("board", Circle::new(0.0, 0.0, 5.0)),
            reward_locations: vec![Perimeter::new("reward_a", Circle::new(3.0, 0.0, 0.5))],
        };

        let timing = FrameTiming::cfr(4.0);
        let df = test_df(
            &[6.0, 6.0, 5.0, 4.0, 3.4, 3.2],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let result = task.full_analyze(df.lazy(), &timing);

        assert_eq!(result.enclosure_label, "board");
        assert_eq!(result.reward_labels, vec!["reward_a"]);
        assert_eq!(result.start_area_exit_frame, Some(2));
        assert_eq!(result.reward_arrival_frame, Some(4));
        assert!((result.start_area_exit_seconds.unwrap() - 0.5).abs() < 1e-10);
        assert!((result.reward_arrival_seconds.unwrap() - 1.0).abs() < 1e-10);
        assert!((result.seconds_to_find_reward.unwrap() - 0.5).abs() < 1e-10);
        assert!((result.seconds_in_reward_area - 0.5).abs() < 1e-10);
        assert_eq!(result.reward_entries, 1);
        assert!((result.search_path_distance_meters.unwrap() - 1.0).abs() < 1e-10);
        assert!((result.search_direct_distance_meters.unwrap() - 1.0).abs() < 1e-10);
        assert!((result.search_mean_speed_mps.unwrap() - 2.0).abs() < 1e-10);
        assert!((result.search_tortuosity.unwrap() - 1.0).abs() < 1e-10);
        assert!(result.total_distance_meters > 0.0);
        assert!(result.total_mean_speed_mps > 0.0);
        assert_eq!(result.reward_summaries.len(), 1);
        assert_eq!(result.reward_summaries[0].zone_name, "reward_a");
    }

    #[test]
    fn full_analyze_requires_reward_after_start_exit() {
        let task = Cheeseboard {
            start_area: Some(PerimeterSpec::Rectangle {
                label: "start_area".into(),
                center_x: 6.0,
                center_y: 0.0,
                width: 1.0,
                height: 1.0,
            }),
            enclosure: Perimeter::new("board", Circle::new(0.0, 0.0, 5.0)),
            reward_locations: vec![Perimeter::new("reward_a", Circle::new(6.0, 0.0, 0.6))],
        };

        let timing = FrameTiming::cfr(4.0);
        let df = test_df(&[6.0, 6.0, 5.0, 4.0, 3.0], &[0.0, 0.0, 0.0, 0.0, 0.0]);

        let result = task.full_analyze(df.lazy(), &timing);

        assert_eq!(result.start_area_exit_frame, Some(2));
        assert_eq!(result.reward_arrival_frame, None);
        assert_eq!(result.reward_arrival_seconds, None);
        assert_eq!(result.seconds_to_find_reward, None);
        assert_eq!(result.search_path_distance_meters, None);
        assert_eq!(result.search_direct_distance_meters, None);
        assert_eq!(result.search_mean_speed_mps, None);
        assert_eq!(result.search_tortuosity, None);
        assert!((result.seconds_in_reward_area - 0.5).abs() < 1e-10);
        assert_eq!(result.reward_entries, 1);
    }

    #[test]
    fn analyze_detailed_returns_typed_task_detail() {
        let task = Cheeseboard {
            start_area: None,
            enclosure: Perimeter::new("board", Circle::new(0.0, 0.0, 5.0)),
            reward_locations: vec![Perimeter::new("reward_a", Circle::new(3.0, 0.0, 0.5))],
        };

        let timing = FrameTiming::cfr(4.0);
        let df = test_df(
            &[6.0, 4.5, 4.0, 3.5, 3.3, 3.2],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let (analysis, detail) = task.analyze_detailed(df.lazy(), &timing);

        assert_eq!(analysis.task_type, TaskType::Cheeseboard);
        assert_eq!(analysis.summaries.len(), 2);
        assert!(matches!(detail, Some(TaskDetail::Cheeseboard(_))));
    }
}
