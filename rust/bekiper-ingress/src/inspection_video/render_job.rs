/// Timeline strip geometry below scored videos (pixels).
const SCORED_STRIP_TOP: u32 = 2;
const SCORED_ROW_HEIGHT: u32 = 7;
const SCORED_ROW_PITCH: u32 = 8;
/// Qualia videos are upscaled to at least this many pixels wide.
const QUALIA_MIN_WIDTH: u32 = 640;
/// Largest accepted `--upscale` factor (keeps `width * upscale` far from u32 overflow).
const MAX_UPSCALE: u32 = 8;

fn render_job(
    bundle: &LoadedInspection,
    theme: &VideoTheme,
    opts: &RenderOptions,
    job: &RenderJob,
    cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    let mut decoder = FfmpegDecoder::start(&bundle.video_path, &job.output_path, cancelled)?;
    // Qualia frames are upscaled by an integer factor first, so overlays, text
    // and strip are drawn at output resolution (default: at least 640 px wide).
    let upscale = match job.target {
        JobTarget::Qualia(_) => opts
            .upscale
            .unwrap_or_else(|| QUALIA_MIN_WIDTH.div_ceil(decoder.width.max(1))),
        _ => 1,
    }
    .max(1);
    let (frame_width, frame_height) = (decoder.width * upscale, decoder.height * upscale);
    let scale = frame_scale(&bundle.manifest, frame_width, frame_height);
    let mut state = JobState::build(bundle, job, &scale)?;
    // Ticks are drawn over decoded frames, so they follow the decoder rate.
    state.fps = decoder.fps;
    if matches!(job.target, JobTarget::Qualia(_)) {
        state.layout = qualia_layout(upscale);
    }
    let scored = matches!(job.target, JobTarget::Scored | JobTarget::Qualia(_));
    let layout = state.layout;
    // Scored videos append the strip and pad both dimensions to even sizes
    // (yuv420p); other modes encode the source frame size unchanged.
    let (width, height, strip) = if scored {
        let width = frame_width + frame_width % 2;
        let rows = layout.top + state.scored.len() as u32 * layout.pitch;
        let strip_height = rows + layout.footer + (frame_height + rows + layout.footer) % 2;
        let frames = decoder
            .frame_count
            .unwrap_or(0)
            .max(state.frame_indices.last().map_or(1, |last| last + 1));
        let mut strip = state.scored_strip(width, strip_height, frames);
        if let JobTarget::Qualia(feature) = &job.target {
            draw_qualia_legend(
                &mut strip,
                feature,
                rows + layout.axis + layout.text,
                layout.text,
            );
        }
        (width, frame_height + strip_height, Some((strip, frames)))
    } else {
        (frame_width, frame_height, None)
    };
    let crf = encoder_supports_crf(map_codec(&opts.codec)).then(|| opts.crf.unwrap_or(DEFAULT_CRF));
    let mut encoder = FfmpegEncoder::start(
        &job.output_path,
        &opts.codec,
        &theme.ffmpeg_preset,
        crf,
        width,
        height,
        decoder.fps,
    )?;
    let poster = job.output_path.with_extension("poster.png");

    let mut frame_index = 0;
    while let Some(mut rgb) = decoder.next_frame(cancelled)? {
        if cancelled() {
            return Err(BekiperError::Other("inspection video cancelled".into()));
        }
        if upscale > 1 {
            rgb = image::imageops::resize(
                &rgb,
                frame_width,
                frame_height,
                image::imageops::FilterType::Triangle,
            );
        }
        render_frame(&mut rgb, frame_index, bundle, theme, job, &state);
        if let Some((strip, frames)) = &strip {
            rgb = compose_scored(&rgb, strip, frame_index as u64, *frames, &state);
        }
        if frame_index == 0 {
            rgb.save(&poster)
                .map_err(|error| BekiperError::Data(format!("poster: {error}")))?;
        }
        encoder.write_frame(&rgb, cancelled)?;
        frame_index += 1;
    }
    if frame_index == 0 {
        return Err(BekiperError::Data(
            "inspection source video decoded no frames".into(),
        ));
    }

    encoder.finish(cancelled)?;
    let mut metadata = serde_json::json!({
        "schema_version": 1, "status": "complete", "source_video": bundle.video_path,
        "recorded_source_video": bundle.recorded_video_path,
        "source_fps": decoder.fps, "encoded_fps": decoder.fps,
        "resolution": [width, height], "frames_written": frame_index,
        "codec": opts.codec, "ffmpeg_preset": theme.ffmpeg_preset, "crf": crf,
        "faststart": faststart_container(&job.output_path), "poster": poster,
    });
    if scored {
        metadata["frame_size"] = serde_json::json!([decoder.width, decoder.height]);
        metadata["upscale"] = upscale.into();
        metadata["strip_height_px"] = (height - decoder.height).into();
        metadata["strip_row_top_px"] = layout.top.into();
        metadata["strip_row_height_px"] = layout.row_height.into();
        metadata["strip_row_pitch_px"] = layout.pitch.into();
        metadata["strip_gutter_px"] = layout.gutter.into();
        metadata["scored_rows"] = state
            .scored
            .iter()
            .map(|row| {
                let mut entry = serde_json::json!({
                    "heuristic": row.heuristic, "result_column": row.result_column,
                    "perimeter": row.perimeter,
                });
                // Qualia rows are also drawn with a label; scored receipts keep their shape.
                if layout.gutter > 0 {
                    entry["label"] = row.label.clone().into();
                }
                entry
            })
            .collect();
    }
    let path = job.output_path.with_extension("video.json");
    let temporary = path.with_extension("json.pending");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&metadata)
            .map_err(|error| BekiperError::Data(error.to_string()))?,
    )?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// The rendered frame above the precomputed strip, with the playhead at
/// `frame_index`. Rows are copied as slices: this runs once per output frame.
fn compose_scored(
    frame: &RgbImage,
    strip: &RgbImage,
    frame_index: u64,
    frames: u64,
    state: &JobState,
) -> RgbImage {
    let layout = state.layout;
    let row = strip.width() as usize * 3;
    let pad = parse_color(themekit::DARK.background).0;
    let mut bytes = Vec::with_capacity(row * (frame.height() + strip.height()) as usize);
    for line in frame.as_raw().chunks_exact(frame.width() as usize * 3) {
        bytes.extend_from_slice(line);
        // Odd source widths gain one background column.
        for _ in (line.len()..row).step_by(3) {
            bytes.extend_from_slice(&pad);
        }
    }
    bytes.extend_from_slice(strip.as_raw());
    let mut canvas = RgbImage::from_raw(strip.width(), frame.height() + strip.height(), bytes)
        .expect("canvas rows match its dimensions");
    // At least two pixels wide so chroma subsampling keeps the playhead legible.
    let thick = layout.unit.max(2);
    let x = (layout.gutter + strip_x(frame_index, strip.width() - layout.gutter, frames))
        .min(strip.width().saturating_sub(layout.unit.max(2)));
    let ink = parse_color(themekit::DARK.ink);
    let rows_bottom = frame.height() + layout.top + state.scored.len() as u32 * layout.pitch;
    for y in frame.height()..rows_bottom.min(canvas.height()) {
        for x in x..(x + thick).min(strip.width()) {
            canvas.put_pixel(x, y, ink);
        }
    }
    canvas
}

fn render_frame(
    img: &mut RgbImage,
    frame_idx: usize,
    bundle: &LoadedInspection,
    theme: &VideoTheme,
    job: &RenderJob,
    state: &JobState,
) {
    match &job.target {
        JobTarget::Overview => render_overview(img, frame_idx, bundle, theme, state, false),
        JobTarget::Scored => render_overview(img, frame_idx, bundle, theme, state, true),
        JobTarget::Qualia(feature) => draw_detection_fields(img, frame_idx, bundle, state, feature),
        JobTarget::Feature(feature) => {
            render_feature(img, frame_idx, bundle, theme, state, feature)
        }
    }
}

fn render_overview(
    img: &mut RgbImage,
    frame_idx: usize,
    bundle: &LoadedInspection,
    theme: &VideoTheme,
    state: &JobState,
    scored: bool,
) {
    let perimeter_color = parse_color(&theme.video_perimeter_color);
    let thickness = theme_thickness(theme.linewidth);
    // Scored outlines draw last so coincident unscored perimeters never hide them.
    let mut perimeters: Vec<_> = state
        .perimeters
        .iter()
        .map(|(label, perimeter)| (state.scored_outline(label, frame_idx), label, perimeter))
        .collect();
    perimeters.sort_by(|a, b| (a.0.is_some(), a.1).cmp(&(b.0.is_some(), b.1)));
    for (mask, label, perimeter) in perimeters {
        match mask {
            Some(color) => draw_perimeter(img, perimeter, color, thickness, 0.95),
            // Unscored context stays faint and neutral beside mask colours.
            None if scored => draw_perimeter(
                img,
                perimeter,
                parse_color(themekit::DARK.reference),
                1,
                0.6,
            ),
            None => {
                let color = bundle
                    .manifest
                    .label_colors
                    .get(label)
                    .map(|c| parse_color(c))
                    .unwrap_or(perimeter_color);
                draw_perimeter(img, perimeter, color, thickness, 0.95);
            }
        }
    }
    for quadrant in state.quadrants.values() {
        draw_quadrant_grid(img, quadrant, perimeter_color, thickness.max(1), None);
    }
    draw_current_points(img, frame_idx, bundle, theme, state, None, false);
}

fn render_feature(
    img: &mut RgbImage,
    frame_idx: usize,
    bundle: &LoadedInspection,
    theme: &VideoTheme,
    state: &JobState,
    feature: &FeatureInspection,
) {
    match feature {
        FeatureInspection::Perimeter {
            perimeter_label, ..
        } => {
            if let Some(perimeter) = state.perimeters.get(perimeter_label) {
                draw_perimeter(
                    img,
                    perimeter,
                    resolved_feature_color(
                        feature,
                        bundle
                            .manifest
                            .label_colors
                            .get(perimeter_label)
                            .unwrap_or(&theme.video_perimeter_color),
                    ),
                    resolved_feature_thickness(feature, theme.linewidth),
                    0.95,
                );
            }
            draw_current_points(img, frame_idx, bundle, theme, state, None, false);
        }
        FeatureInspection::Zone {
            perimeter_label, ..
        }
        | FeatureInspection::Heuristic {
            perimeter_label: Some(perimeter_label),
            ..
        } => {
            let active = state.bool_at(frame_idx).unwrap_or(false);
            let color = bool_color(theme, active);
            if let Some(perimeter) = state.perimeters.get(perimeter_label) {
                draw_perimeter(
                    img,
                    perimeter,
                    color,
                    resolved_feature_thickness(feature, theme.linewidth),
                    0.95,
                );
            }
            draw_current_points(img, frame_idx, bundle, theme, state, Some(color), true);
        }
        FeatureInspection::Heuristic { .. } => {
            let active = state.bool_at(frame_idx).unwrap_or(false);
            draw_current_points(
                img,
                frame_idx,
                bundle,
                theme,
                state,
                Some(bool_color(theme, active)),
                true,
            );
        }
        FeatureInspection::Quadrant { quadrant_label, .. } => {
            let primary = state
                .labels
                .first()
                .and_then(|label| state.coord_at(label, frame_idx));
            let highlight = primary.and_then(|(x, y)| {
                state
                    .quadrants
                    .get(quadrant_label)
                    .and_then(|quad| current_cell(quad, x, y))
            });
            if let Some(quadrant) = state.quadrants.get(quadrant_label) {
                draw_quadrant_grid(
                    img,
                    quadrant,
                    parse_color(&theme.video_perimeter_color),
                    theme_thickness(theme.linewidth).max(1),
                    highlight,
                );
            }
            draw_current_points(img, frame_idx, bundle, theme, state, None, false);
        }
        FeatureInspection::Angle {
            from_coordinate_label,
            to_coordinate_label,
            ..
        } => {
            if let (Some((fx, fy)), Some((tx, ty))) = (
                state.coord_at(from_coordinate_label, frame_idx),
                state.coord_at(to_coordinate_label, frame_idx),
            ) {
                let color = resolved_feature_color(feature, &theme.video_true_color);
                draw_arrow_alpha(
                    img,
                    fx,
                    fy,
                    tx,
                    ty,
                    color,
                    resolved_feature_thickness(feature, theme.linewidth),
                    1.0,
                );
                draw_marker(img, fx, fy, parse_color(&theme.video_point_color), 1.0);
                draw_marker(img, tx, ty, parse_color(&theme.video_point_color), 0.9);
            }
        }
        FeatureInspection::Trajectory { .. } => {
            let color = resolved_feature_color(feature, &theme.video_true_color);
            let thickness = resolved_feature_thickness(feature, theme.linewidth);
            for label in &state.labels {
                if let Some(segments) = state.segments_until(label, frame_idx) {
                    for points in segments {
                        if points.len() > 1 {
                            draw_polyline_alpha(img, &points, color, thickness, false, 0.9);
                        } else if let Some(&(x, y)) = points.first() {
                            draw_marker(img, x, y, color, 0.9);
                        }
                    }
                    if let Some((x, y)) = state.coord_at(label, frame_idx) {
                        draw_marker(img, x, y, color, 1.0);
                    }
                }
            }
        }
        FeatureInspection::RayCast {
            origin_label,
            angle_column,
            length_meters,
            cone_width_deg,
            ..
        } => {
            if let (Some((ox, oy)), Some(angle_rad)) = (
                state.coord_at(origin_label, frame_idx),
                state.scalar_at(angle_column, frame_idx),
            ) {
                let color = resolved_feature_color(feature, &theme.video_true_color);
                let radius = metric_length_to_pixels(
                    *length_meters,
                    bundle.manifest.settings.meters_per_pixel,
                    img.width() as f64 / bundle.manifest.video.resolution.0.max(1) as f64,
                );
                let angle_deg = angle_rad.to_degrees();
                draw_wedge_alpha(
                    img,
                    ox,
                    oy,
                    radius,
                    angle_deg - cone_width_deg / 2.0,
                    angle_deg + cone_width_deg / 2.0,
                    color,
                    resolved_feature_thickness(feature, theme.linewidth),
                    0.95,
                    theme.quiver_alpha,
                );
                let tx = ox + radius * angle_rad.cos();
                let ty = oy + radius * angle_rad.sin();
                draw_arrow_alpha(
                    img,
                    ox,
                    oy,
                    tx,
                    ty,
                    color,
                    resolved_feature_thickness(feature, theme.linewidth),
                    0.95,
                );
                draw_marker(img, ox, oy, parse_color(&theme.video_point_color), 1.0);
            }
        }
        FeatureInspection::Proximity {
            perimeter_label,
            buffer_meters,
            ..
        } => {
            let active = state.bool_at(frame_idx).unwrap_or(false);
            let color = bool_color(theme, active);
            if let Some(perimeter) = state.perimeters.get(perimeter_label) {
                draw_perimeter(
                    img,
                    perimeter,
                    color,
                    resolved_feature_thickness(feature, theme.linewidth),
                    0.95,
                );
                if let Some(expanded) = expand_perimeter_pixels(
                    perimeter,
                    *buffer_meters,
                    bundle,
                    (img.width(), img.height()),
                ) {
                    draw_perimeter(
                        img,
                        &expanded,
                        color,
                        resolved_feature_thickness(feature, theme.linewidth).max(1),
                        0.45,
                    );
                }
            }
            draw_current_points(img, frame_idx, bundle, theme, state, Some(color), true);
        }
        FeatureInspection::HeuristicDecomposition { components, .. } => {
            for component in components {
                let active = state.component_at(&component.result_column, frame_idx);
                let color = active
                    .map(|active| bool_color(theme, active))
                    .unwrap_or(image::Rgb([160, 160, 160]));
                if let Some(geometry) = &component.evaluated_perimeter {
                    let factor = if bundle.manifest.settings.meters_per_pixel > 0.0 {
                        1.0 / bundle.manifest.settings.meters_per_pixel
                    } else {
                        1.0
                    };
                    let scale = frame_scale(&bundle.manifest, img.width(), img.height());
                    let perimeter = scale_perimeter(&geometry.scale(factor), scale.sx, scale.sy);
                    draw_perimeter(
                        img,
                        &perimeter,
                        color,
                        theme_thickness(theme.linewidth),
                        0.85,
                    );
                    if let Some(buffer) = &component.buffer_perimeter {
                        draw_perimeter(
                            img,
                            &scale_perimeter(&buffer.scale(factor), scale.sx, scale.sy),
                            color,
                            1,
                            0.4,
                        );
                    }
                }
                if let Some(origin) = &component.ray_origin_label {
                    let value = |column: &Option<String>| {
                        column.as_ref().and_then(|c| state.scalar_at(c, frame_idx))
                    };
                    if let (Some((ox, oy)), Some(dx), Some(dy)) = (
                        state.coord_at(origin, frame_idx),
                        value(&component.ray_direction_x_column),
                        value(&component.ray_direction_y_column),
                    ) {
                        let scale = frame_scale(&bundle.manifest, img.width(), img.height());
                        let (dx, dy) = (dx * scale.sx, dy * scale.sy);
                        let norm = dx.hypot(dy);
                        if norm.is_finite() && norm > 0.0 {
                            let length = img.width().min(img.height()) as f64 * 0.045;
                            draw_arrow_alpha(
                                img,
                                ox,
                                oy,
                                ox + dx / norm * length,
                                oy + dy / norm * length,
                                color,
                                theme_thickness(theme.linewidth),
                                theme.quiver_alpha,
                            );
                            if let (Some(tx), Some(ty)) = (
                                value(&component.ray_target_x_column),
                                value(&component.ray_target_y_column),
                            ) && tx.is_finite()
                                && ty.is_finite()
                            {
                                let mpp = bundle.manifest.settings.meters_per_pixel;
                                draw_marker(
                                    img,
                                    metric_coord_to_pixels(tx, mpp, scale.sx),
                                    metric_coord_to_pixels(ty, mpp, scale.sy),
                                    parse_color(&theme.video_point_color),
                                    0.65,
                                );
                            }
                        }
                    }
                } else {
                    for label in &component.coordinate_labels {
                        if let Some((x, y)) = state.coord_at(label, frame_idx) {
                            draw_marker(img, x, y, color, 1.0);
                        }
                    }
                }
            }
        }
        FeatureInspection::DetectionFields { .. } => {
            draw_detection_fields(img, frame_idx, bundle, state, feature)
        }
        FeatureInspection::ScaleDerivation { .. }
        | FeatureInspection::HeuristicCloud { .. }
        | FeatureInspection::TimeSeries { .. }
        | FeatureInspection::NortDiscrimination { .. }
        | FeatureInspection::ArmSequence { .. }
        | FeatureInspection::YMazeArmSections { .. }
        | FeatureInspection::CenterPeriphery { .. }
        | FeatureInspection::StateDistribution { .. } => {}
    }
}

fn draw_current_points(
    img: &mut RgbImage,
    frame_idx: usize,
    bundle: &LoadedInspection,
    theme: &VideoTheme,
    state: &JobState,
    forced_color: Option<image::Rgb<u8>>,
    use_bool_color: bool,
) {
    for label in &state.labels {
        if let Some((x, y)) = state.coord_at(label, frame_idx) {
            let color = forced_color.unwrap_or_else(|| {
                if use_bool_color {
                    bool_color(theme, state.bool_at(frame_idx).unwrap_or(false))
                } else {
                    bundle
                        .manifest
                        .label_colors
                        .get(label)
                        .map(|value| parse_color(value))
                        .unwrap_or_else(|| parse_color(&theme.video_point_color))
                }
            });
            draw_marker(img, x, y, color, 1.0);
        }
    }
}

fn draw_marker(img: &mut RgbImage, x: f64, y: f64, color: image::Rgb<u8>, alpha: f32) {
    draw_circle_alpha(img, x, y, 4.0, color, 2, alpha);
    draw_crosshair_alpha(img, x, y, 4, color, 1, alpha);
}

fn draw_quadrant_grid(
    img: &mut RgbImage,
    quad: &ScaledQuadrant,
    color: image::Rgb<u8>,
    thickness: i32,
    highlight: Option<(u32, u32)>,
) {
    let cell_w = (quad.x_max - quad.x_min) / quad.cols.max(1) as f64;
    let cell_h = (quad.y_max - quad.y_min) / quad.rows.max(1) as f64;

    if let Some((row, col)) = highlight {
        let x = quad.x_min + cell_w * col as f64;
        let y = quad.y_min + cell_h * row as f64;
        fill_rect_alpha(img, x, y, cell_w, cell_h, color, 0.20);
    }

    for col in 0..=quad.cols {
        let x = quad.x_min + cell_w * col as f64;
        draw_line_alpha(img, x, quad.y_min, x, quad.y_max, color, thickness, 0.85);
    }
    for row in 0..=quad.rows {
        let y = quad.y_min + cell_h * row as f64;
        draw_line_alpha(img, quad.x_min, y, quad.x_max, y, color, thickness, 0.85);
    }
}

fn current_cell(quad: &ScaledQuadrant, x: f64, y: f64) -> Option<(u32, u32)> {
    if x < quad.x_min || x > quad.x_max || y < quad.y_min || y > quad.y_max {
        return None;
    }
    let cell_w = (quad.x_max - quad.x_min) / quad.cols.max(1) as f64;
    let cell_h = (quad.y_max - quad.y_min) / quad.rows.max(1) as f64;
    let col = ((x - quad.x_min) / cell_w).floor() as u32;
    let row = ((y - quad.y_min) / cell_h).floor() as u32;
    Some((
        row.min(quad.rows.saturating_sub(1)),
        col.min(quad.cols.saturating_sub(1)),
    ))
}

fn frame_scale(manifest: &InspectionManifest, frame_width: u32, frame_height: u32) -> FrameScale {
    let base_w = manifest.video.resolution.0.max(1) as f64;
    let base_h = manifest.video.resolution.1.max(1) as f64;
    // Metric / mpp_x is square-pixel space; undo non-square source pixels.
    FrameScale {
        sx: frame_width as f64 / base_w,
        sy: frame_height as f64 / base_h / manifest.y_aspect(),
    }
}

fn load_coordinate_series(
    df: &DataFrame,
    spec: &CoordinateColumnSpec,
    meters_per_pixel: f64,
    scale: &FrameScale,
) -> Result<Vec<Option<(f64, f64)>>> {
    let xs = df.column(&spec.x)?.f64()?.clone();
    let ys = df.column(&spec.y)?.f64()?.clone();
    let len = xs.len().max(ys.len());
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        match (xs.get(idx), ys.get(idx)) {
            (Some(x), Some(y)) if x.is_finite() && y.is_finite() => {
                let px = metric_coord_to_pixels(x, meters_per_pixel, scale.sx);
                let py = metric_coord_to_pixels(y, meters_per_pixel, scale.sy);
                out.push(Some((px, py)));
            }
            _ => out.push(None),
        }
    }
    Ok(out)
}

fn load_bool_column(df: &DataFrame, name: &str) -> Result<Vec<Option<bool>>> {
    let bools = df.column(name)?.bool()?.clone();
    Ok(bools.into_iter().collect())
}

fn load_f64_column(df: &DataFrame, name: &str) -> Result<Vec<Option<f64>>> {
    let values = df.column(name)?.f64()?.clone();
    Ok(values.into_iter().collect())
}
