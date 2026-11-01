impl<S> Heuristic for ExplorationHeuristic<S>
where
    S: Expandable + ClosestPoint + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn name(&self) -> &str {
        &self.object_label
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        // 1. Outside the (optionally eroded) object footprint.
        let erosion = self.footprint_erosion_meters;
        let shared = self.shared.as_deref();
        let outside = match shared.and_then(|s| s.outside_expr(erosion)) {
            Some(outside) => outside,
            None => {
                let inside = if erosion > 0.0 {
                    self.object_perimeter
                        .expanded(-erosion)
                        .confinement_mask(&self.centroid_coords)
                } else {
                    self.object_perimeter
                        .confinement_mask(&self.centroid_coords)
                };
                inside.not().fill_null(lit(false))
            }
        };

        // 2./3. Nose within proximity / contact (expanded perimeter).
        let nose_mask = |distance: f64| -> Expr {
            match shared {
                Some(s) if s.supports_expand(distance) => s.nose_confinement_expr(distance),
                _ => self
                    .object_perimeter
                    .expanded(distance)
                    .confinement_mask(&self.nose_coords),
            }
        };
        let nose_near = nose_mask(self.nose_max_distance).fill_null(lit(false));
        let nose_contact = nose_mask(self.contact_distance).fill_null(lit(false));

        // 4. Head orientation: angle between the head direction and the
        //    vector from the nose to the nearest object-perimeter point.
        let [
            within_angle_deg,
            within_angle_target_x,
            within_angle_target_y,
        ] = match shared {
            Some(s) => s.angle_exprs(),
            None => self.object_perimeter.shape.nearest_point_heading_exprs(
                &self.nose_coords.x,
                &self.nose_coords.y,
                "snout_dx",
                "snout_dy",
            ),
        };
        let within_angle = within_angle_deg
            .clone()
            .lt_eq(lit(self.max_angle_deg + ANGLE_EPSILON_DEG))
            .fill_null(lit(false));
        let within_angle_margin_deg = lit(self.max_angle_deg) - within_angle_deg.clone();

        // 5. Speed filter: nose speed must be below threshold
        let nose_speed = velocity_expr(&self.nose_coords.x, &self.nose_coords.y);
        let nose_speed_mps = nose_speed.clone() * lit(self.fps);
        let limit = lit(self.max_nose_speed / self.fps);
        let slow_enough = (if self.strict_boundaries {
            nose_speed.lt(limit)
        } else {
            nose_speed.lt_eq(limit)
        })
        .fill_null(lit(false));

        let contact_rescue = match &self.posture_col {
            Some(posture) => nose_contact
                .clone()
                .and(col(posture).fill_null(lit(false)).not()),
            None => nose_contact.clone(),
        };
        let contact_or_angle = contact_rescue
            .or(within_angle.clone())
            .fill_null(lit(false));

        // Whisker fan: a hit is a candidate without the nose being near, but
        // grooming/rearing proxy frames still need the head angle.
        let whisker_exprs = self.whisker.as_ref().map(|fan| {
            let [hit, side, distance] = shared
                .and_then(|s| s.fan_exprs(&fan.params))
                .unwrap_or_else(|| fan.hit_exprs(&self.head_base_coords, &self.nose_coords));
            let candidate = match &self.posture_col {
                Some(posture) => hit.clone().and(col(posture).fill_null(lit(false)).not()),
                None => hit.clone(),
            };
            (hit, side, distance, candidate)
        });

        // Combined candidate mask. Per-bout median mode removes only the
        // corresponding strict per-frame gate; contact-or-angle remains the
        // loose frame-level trigger for bout admission.
        let mut nose_candidate = nose_near.clone().and(contact_or_angle.clone());
        if let Some((_, _, _, candidate)) = &whisker_exprs {
            nose_candidate = nose_candidate.or(candidate.clone());
        }
        let mut exploring = outside.clone().and(nose_candidate).fill_null(lit(false));
        if self.speed_filter_mode == BoutFilterMode::PerFrame {
            exploring = exploring.and(slow_enough.clone()).fill_null(lit(false));
        }

        let raw_col = self.raw_col();
        let raw_diag_col = self.raw_diag_col();
        let mut whisker_columns = Vec::new();
        if let Some((hit, side, distance, candidate)) = whisker_exprs {
            whisker_columns = vec![
                hit.fill_null(lit(false)).alias(self.whisker_hit_col()),
                side.alias(self.whisker_side_col()),
                distance.alias(self.whisker_nearest_distance_col()),
                candidate.clone().alias(self.whisker_candidate_col()),
                nose_contact
                    .clone()
                    .or(candidate)
                    .alias(self.bout_contact_col()),
            ];
        }
        lf.with_columns(whisker_columns).with_columns([
            outside.alias(self.outside_col()),
            nose_near.clone().alias(self.nose_near_col()),
            nose_contact.clone().alias(self.nose_contact_col()),
            within_angle.clone().alias(self.within_angle_col()),
            within_angle_deg.alias(self.within_angle_min_deg_col()),
            within_angle_margin_deg.alias(self.within_angle_margin_deg_col()),
            within_angle_target_x.alias(self.within_angle_target_x_col()),
            within_angle_target_y.alias(self.within_angle_target_y_col()),
            contact_or_angle.alias(self.contact_or_angle_col()),
            slow_enough.alias(self.slow_nose_col()),
            nose_speed_mps.alias(self.nose_speed_mps_col()),
            exploring.clone().alias(raw_diag_col.as_str()),
            exploring.alias(raw_col.as_str()),
        ])
    }

    fn summary(
        &self,
        df: &DataFrame,
        timing: &bekiper_core::video::FrameTiming,
    ) -> crate::heuristic::HeuristicSummary {
        // Apply tolerance (bout merging + minimum duration) on collected data
        let raw_col = self.raw_col();
        let clock = BoutTiming::frames(timing, df.height()).expect("valid exploration timing");

        let (true_frames, total_frames, seconds) = if let Ok(series) = df.column(&raw_col)
            && let Ok(bools) = series.bool()
        {
            let nose_speed = df
                .column(&self.nose_speed_mps_col())
                .ok()
                .and_then(|series| series.f64().ok());
            let angle = df
                .column(&self.within_angle_min_deg_col())
                .ok()
                .and_then(|series| series.f64().ok());
            let nose_contact = bool_column(df, &self.bout_contact_col());
            let toleranced = filter_selected_bouts(
                bools,
                select_exploration_bouts_for_column(
                    df,
                    &raw_col,
                    bools,
                    &clock,
                    self.min_bout_seconds,
                    (self.max_gap_seconds, self.strict_boundaries),
                    self.gap_density_threshold,
                ),
                BoutFilterSpec {
                    strict_boundaries: self.strict_boundaries,
                    speed_mode: self.speed_filter_mode,
                    angle_mode: self.angle_filter_mode,
                    max_nose_speed: self.max_nose_speed,
                    max_angle_deg: self.max_angle_deg,
                    nose_speed_mps: nose_speed,
                    angle_deg: angle,
                    nose_contact: nose_contact.as_ref(),
                },
            )
            .0;
            let total = toleranced.len() as u64;
            let trues = toleranced.sum().unwrap_or(0) as u64;
            let values: Vec<_> = toleranced.into_iter().map(|v| v.unwrap_or(false)).collect();
            (trues, total, clock.mask_seconds(&values))
        } else {
            (0, 0, 0.0)
        };

        crate::heuristic::HeuristicSummary {
            name: format!("exploring_{}", self.object_label),
            true_frames,
            total_frames,
            seconds,
        }
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let mut decomposition = HeuristicDecompositionSpec {
            heuristic_name: self.heuristic_name(),
            result_column: self.final_toleranced_col(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.nose_coords.x)],
            perimeter_label: Some(self.object_perimeter.label.clone()),
            components: vec![
                {
                    let outside = decomposition_component(
                        "outside_object",
                        self.outside_col(),
                        vec![coordinate_label_from_xy(&self.centroid_coords.x)],
                        Some(self.object_perimeter.label.clone()),
                    );
                    if self.footprint_erosion_meters > 0.0 {
                        outside.with_buffer(-self.footprint_erosion_meters)
                    } else {
                        outside
                    }
                },
                decomposition_component(
                    "nose_near",
                    self.nose_near_col(),
                    vec![coordinate_label_from_xy(&self.nose_coords.x)],
                    Some(self.object_perimeter.label.clone()),
                )
                .with_buffer(self.nose_max_distance),
                decomposition_component(
                    "nose_contact",
                    self.nose_contact_col(),
                    vec![coordinate_label_from_xy(&self.nose_coords.x)],
                    Some(self.object_perimeter.label.clone()),
                )
                .with_buffer(self.contact_distance),
                decomposition_component(
                    "within_angle",
                    self.within_angle_col(),
                    vec![coordinate_label_from_xy(&self.nose_coords.x)],
                    Some(self.object_perimeter.label.clone()),
                )
                .with_ray(
                    &bekiper_core::types::RaySpec {
                        label: "head".into(),
                        origin: self.nose_coords.clone(),
                        direction: CoordinateColumns::new("snout_dx", "snout_dy"),
                    },
                    self.within_angle_target_x_col(),
                    self.within_angle_target_y_col(),
                )
                .with_values(self.within_angle_min_deg_col(), "degrees"),
                decomposition_component(
                    "contact_or_angle",
                    self.contact_or_angle_col(),
                    vec![coordinate_label_from_xy(&self.nose_coords.x)],
                    Some(self.object_perimeter.label.clone()),
                ),
                decomposition_component(
                    "slow_nose",
                    self.slow_nose_col(),
                    vec![coordinate_label_from_xy(&self.nose_coords.x)],
                    None,
                ),
                decomposition_component(
                    "raw_combined",
                    self.raw_diag_col(),
                    vec![coordinate_label_from_xy(&self.nose_coords.x)],
                    Some(self.object_perimeter.label.clone()),
                ),
            ]
            .into_iter()
            .map(|c| c.with_geometry(&self.object_perimeter.shape, &self.object_perimeter.label))
            .collect(),
            show_final_result: true,
        };
        if self.whisker.is_some() {
            decomposition.components.push(
                decomposition_component(
                    "whisker_hit",
                    self.whisker_hit_col(),
                    vec![coordinate_label_from_xy(&self.head_base_coords.x)],
                    Some(self.object_perimeter.label.clone()),
                )
                .with_geometry(&self.object_perimeter.shape, &self.object_perimeter.label),
            );
        }
        if self.speed_filter_mode == BoutFilterMode::Disabled {
            decomposition
                .components
                .retain(|component| component.label != "slow_nose");
        }
        Some(decomposition)
    }
}
