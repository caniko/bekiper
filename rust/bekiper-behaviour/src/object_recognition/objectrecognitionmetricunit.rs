impl ObjectRecognitionMetricUnit for ZoneMotionUnit {
    fn compute(
        &self,
        ctx: &ObjectRecognitionMetricContext<'_>,
        metrics: &mut ObjectRecognitionMetrics,
    ) {
        let Some(arena) = &ctx.arena else {
            return;
        };

        let center_summary = ctx.task.center_perimeter.as_ref().map(|_| {
            zone_summary(
                ctx.df,
                "in_center",
                &ctx.task.params.centroid_coords.x,
                &ctx.task.params.centroid_coords.y,
                ctx.task.params.fps,
                ctx.task.params.min_dwell_frames(),
            )
        });
        let center_motion_summary = ctx.task.center_perimeter.as_ref().map(|_| {
            motion_summary_for_zone(
                ctx.df,
                "in_center",
                &ctx.task.params.centroid_coords.x,
                &ctx.task.params.centroid_coords.y,
                ctx.task.params.fps,
                ctx.task.params.min_dwell_frames(),
                "freezing",
            )
        });
        let periphery_summary = ctx.task.center_perimeter.as_ref().map(|_| {
            zone_summary(
                ctx.df,
                "in_periphery",
                &ctx.task.params.centroid_coords.x,
                &ctx.task.params.centroid_coords.y,
                ctx.task.params.fps,
                ctx.task.params.min_dwell_frames(),
            )
        });
        let periphery_motion_summary = ctx.task.center_perimeter.as_ref().map(|_| {
            motion_summary_for_zone(
                ctx.df,
                "in_periphery",
                &ctx.task.params.centroid_coords.x,
                &ctx.task.params.centroid_coords.y,
                ctx.task.params.fps,
                ctx.task.params.min_dwell_frames(),
                "freezing",
            )
        });
        let all_mask = BooleanChunked::full("all".into(), true, ctx.df.height());
        let all_motion_summary = Some(motion_summary_from_mask(
            ctx.df,
            "all",
            &all_mask,
            &ctx.task.params.centroid_coords.x,
            &ctx.task.params.centroid_coords.y,
            ctx.task.params.fps,
            "freezing",
        ));

        let bb = arena.shape.bounding_box();
        let quadrant_labels: Vec<String> = generate_grid_zones(
            (bb.min_x + bb.max_x) / 2.0,
            (bb.min_y + bb.max_y) / 2.0,
            bb.max_x - bb.min_x,
            bb.max_y - bb.min_y,
            OBJECT_RECOGNITION_QUADRANT_ROWS,
            OBJECT_RECOGNITION_QUADRANT_COLS,
        )
        .into_iter()
        .map(|(label, _)| label)
        .collect();

        let quadrant_summaries = quadrant_labels
            .iter()
            .map(|label| {
                zone_summary(
                    ctx.df,
                    label,
                    &ctx.task.params.centroid_coords.x,
                    &ctx.task.params.centroid_coords.y,
                    ctx.task.params.fps,
                    ctx.task.params.min_dwell_frames(),
                )
            })
            .collect();
        let quadrant_motion_summaries = quadrant_labels
            .iter()
            .map(|label| {
                motion_summary_for_zone(
                    ctx.df,
                    label,
                    &ctx.task.params.centroid_coords.x,
                    &ctx.task.params.centroid_coords.y,
                    ctx.task.params.fps,
                    ctx.task.params.min_dwell_frames(),
                    "freezing",
                )
            })
            .collect();

        metrics.zone_metrics = Some(ObjectRecognitionZoneMetrics {
            all_motion_summary,
            center_summary,
            periphery_summary,
            quadrant_summaries,
            center_motion_summary,
            periphery_motion_summary,
            quadrant_motion_summaries,
        });
    }
}
