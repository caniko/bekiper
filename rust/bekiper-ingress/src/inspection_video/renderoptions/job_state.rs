#[derive(Debug, Clone)]
struct ScaledQuadrant {
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
    rows: u32,
    cols: u32,
}

#[derive(Debug, Clone)]
struct FrameScale {
    sx: f64,
    sy: f64,
}

/// A scored heuristic: one timeline row, plus an outline colour when its
/// perimeter is drawn.
#[derive(Debug, Clone)]
struct ScoredRow {
    /// Text drawn in the strip's label gutter (qualia videos only).
    label: String,
    /// True-cell colour (qualia rows); scored rows use the mask palette.
    color: Option<image::Rgb<u8>>,
    heuristic: String,
    result_column: String,
    perimeter: Option<String>,
    mask: Vec<Option<bool>>,
}

#[derive(Debug, Clone)]
struct JobState {
    frame_indices: Vec<u64>,
    labels: Vec<Label>,
    coords: HashMap<Label, Vec<Option<(f64, f64)>>>,
    bools: Option<Vec<Option<bool>>>,
    angles: HashMap<String, Vec<Option<f64>>>,
    perimeters: HashMap<String, PerimeterSpec>,
    quadrants: HashMap<String, ScaledQuadrant>,
    component_bools: HashMap<String, Vec<Option<bool>>>,
    scored: Vec<ScoredRow>,
    layout: StripLayout,
    fps: f64,
    /// Whisker side strings, keyed by the whisker-hit column they qualify.
    sides: HashMap<String, Vec<Option<String>>>,
}

impl JobState {
    fn build(bundle: &LoadedInspection, job: &RenderJob, scale: &FrameScale) -> Result<Self> {
        let labels = labels_for_job(&bundle.manifest, &job.target);

        let coords = labels
            .iter()
            .filter_map(|label| {
                bundle.manifest.coordinate_columns.get(label).map(|spec| {
                    load_coordinate_series(
                        &bundle.df,
                        spec,
                        bundle.manifest.settings.meters_per_pixel,
                        scale,
                    )
                    .map(|values| (label.clone(), values))
                })
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let bools = match &job.target {
            JobTarget::Feature(feature) => feature
                .result_column()
                .map(|name| load_bool_column(&bundle.df, name))
                .transpose()?,
            JobTarget::Overview | JobTarget::Scored | JobTarget::Qualia(_) => None,
        };

        let mut angles = HashMap::new();
        if let JobTarget::Feature(FeatureInspection::RayCast { angle_column, .. }) = &job.target {
            angles.insert(
                angle_column.clone(),
                load_f64_column(&bundle.df, angle_column)?,
            );
        }

        let perimeter_labels = perimeter_labels_for_job(&bundle.manifest, &job.target);
        let perimeters: HashMap<String, PerimeterSpec> = perimeter_labels
            .into_iter()
            .filter_map(|label| {
                scaled_perimeter(&bundle.manifest, &label, scale).map(|spec| (label, spec))
            })
            .collect();
        let scored = match &job.target {
            JobTarget::Scored => scored_heuristics(&bundle.manifest)
                .into_iter()
                .map(|(heuristic, result_column, perimeter)| {
                    Ok(ScoredRow {
                        label: heuristic.clone(),
                        color: None,
                        mask: load_bool_column(&bundle.df, &result_column)?,
                        perimeter: perimeter.filter(|label| perimeters.contains_key(label)),
                        heuristic,
                        result_column,
                    })
                })
                .collect::<Result<_>>()?,
            JobTarget::Qualia(feature) => qualia_rows(&bundle.df, feature)?,
            _ => Vec::new(),
        };
        let layout = if matches!(job.target, JobTarget::Qualia(_)) {
            qualia_layout(1)
        } else {
            SCORED_LAYOUT
        };

        let quadrant_labels = quadrant_labels_for_job(&bundle.manifest, &job.target);
        let quadrants = quadrant_labels
            .into_iter()
            .filter_map(|label| {
                scaled_quadrant(&bundle.manifest, &label, scale).map(|quad| (label, quad))
            })
            .collect();

        let mut component_bools = HashMap::new();
        if let JobTarget::Feature(FeatureInspection::HeuristicDecomposition {
            components, ..
        }) = &job.target
        {
            for component in components {
                component_bools.insert(
                    component.result_column.clone(),
                    load_bool_column(&bundle.df, &component.result_column)?,
                );
                for column in [
                    &component.ray_direction_x_column,
                    &component.ray_direction_y_column,
                    &component.ray_target_x_column,
                    &component.ray_target_y_column,
                ]
                .into_iter()
                .flatten()
                {
                    angles.insert(column.clone(), load_f64_column(&bundle.df, column)?);
                }
            }
        }
        let mut sides = HashMap::new();
        if let JobTarget::Feature(FeatureInspection::DetectionFields {
            nose_near_columns,
            nose_gate_columns,
            whisker_hit_columns,
            ..
        })
        | JobTarget::Qualia(FeatureInspection::DetectionFields {
            nose_near_columns,
            nose_gate_columns,
            whisker_hit_columns,
            ..
        }) = &job.target
        {
            // Hit columns are optional diagnostics: absent ones draw as false.
            for column in nose_near_columns
                .iter()
                .chain(nose_gate_columns)
                .chain(whisker_hit_columns)
            {
                if let Ok(values) = load_bool_column(&bundle.df, column) {
                    component_bools.insert(column.clone(), values);
                }
            }
            for column in whisker_hit_columns {
                let side = column.replace("__whisker_hit", "__whisker_side");
                if let Some(values) = bundle.df.column(&side).ok().and_then(|c| c.str().ok()) {
                    sides.insert(
                        column.clone(),
                        values.into_iter().map(|v| v.map(str::to_owned)).collect(),
                    );
                }
            }
        }
        Ok(Self {
            frame_indices: source_frame_indices(&bundle.df)?,
            labels,
            coords,
            bools,
            angles,
            perimeters,
            quadrants,
            component_bools,
            scored,
            layout,
            fps: bundle.manifest.video.fps,
            sides,
        })
    }

    fn row_at(&self, frame_idx: usize) -> Option<usize> {
        self.frame_indices.binary_search(&(frame_idx as u64)).ok()
    }

    fn bool_at(&self, frame_idx: usize) -> Option<bool> {
        let row = self.row_at(frame_idx)?;
        self.bools
            .as_ref()
            .and_then(|values| values.get(row).copied())
            .flatten()
    }

    fn coord_at(&self, label: &str, frame_idx: usize) -> Option<(f64, f64)> {
        let row = self.row_at(frame_idx)?;
        self.coords
            .get(label)
            .and_then(|series| series.get(row))
            .copied()
            .flatten()
    }

    fn scalar_at(&self, column: &str, frame_idx: usize) -> Option<f64> {
        let row = self.row_at(frame_idx)?;
        self.angles.get(column)?.get(row).copied().flatten()
    }

    fn component_at(&self, column: &str, frame_idx: usize) -> Option<bool> {
        let row = self.row_at(frame_idx)?;
        self.component_bools
            .get(column)?
            .get(row)
            .copied()
            .flatten()
    }

    fn side_at(&self, column: &str, frame_idx: usize) -> Option<&str> {
        let row = self.row_at(frame_idx)?;
        self.sides.get(column)?.get(row)?.as_deref()
    }

    /// Mask colour for a perimeter that a scored heuristic recolours.
    fn scored_outline(&self, label: &str, frame_idx: usize) -> Option<image::Rgb<u8>> {
        let row = self
            .scored
            .iter()
            .find(|row| row.perimeter.as_deref() == Some(label))?;
        let value = self
            .row_at(frame_idx)
            .and_then(|index| row.mask.get(index).copied().flatten());
        Some(mask_color(value))
    }

    /// Full-recording mask rows, rasterised once. Pixel column `x` spans
    /// source frames `[x·frames/width, (x+1)·frames/width)` (at least one) and
    /// shows true when any of them is true, so sub-pixel bouts stay visible;
    /// otherwise false, else unavailable.
    fn scored_strip(&self, width: u32, height: u32, frames: u64) -> RgbImage {
        let layout = self.layout;
        let ink = parse_color(themekit::DARK.ink);
        let mut strip = RgbImage::from_pixel(width, height, parse_color(themekit::DARK.background));
        let (bars, frames) = (u64::from(width - layout.gutter), frames.max(1));
        for (index, row) in self.scored.iter().enumerate() {
            let mut by_frame = vec![None; frames as usize];
            for (&frame, &value) in self.frame_indices.iter().zip(&row.mask) {
                if let Some(slot) = by_frame.get_mut(frame as usize) {
                    *slot = value;
                }
            }
            let top = layout.top + index as u32 * layout.pitch;
            if layout.gutter > 0 {
                draw_text(&mut strip, layout.text, top, &row.label, ink, layout.text);
            }
            for x in 0..bars {
                let start = x * frames / bars;
                let end = ((x + 1) * frames / bars).max(start + 1);
                let span = &by_frame[start as usize..end as usize];
                let value = if span.contains(&Some(true)) {
                    Some(true)
                } else if span.contains(&Some(false)) {
                    Some(false)
                } else {
                    None
                };
                let color = match row.color {
                    // Qualia: object colour when true, dim when false, blank when unavailable.
                    Some(true_color) => match value {
                        Some(true) => true_color,
                        Some(false) => parse_color(themekit::DARK.border),
                        None => parse_color(themekit::DARK.background),
                    },
                    None => mask_color(value),
                };
                for y in top..top + layout.row_height {
                    strip.put_pixel(layout.gutter + x as u32, y, color);
                }
            }
        }
        if layout.axis > 0 {
            self.draw_time_axis(&mut strip, frames);
        }
        strip
    }

    /// A tick every 60 s of recording time under the rows, labelled in minutes.
    fn draw_time_axis(&self, strip: &mut RgbImage, frames: u64) {
        let layout = self.layout;
        let (ink, s) = (parse_color(themekit::DARK.muted), layout.text);
        let top = layout.top + self.scored.len() as u32 * layout.pitch + s;
        let bars = strip.width() - layout.gutter;
        for minute in 1.. {
            let frame = (f64::from(minute) * 60.0 * self.fps).round() as u64;
            if self.fps <= 0.0 || frame >= frames {
                break;
            }
            let x = layout.gutter + strip_x(frame, bars, frames);
            for (dx, dy) in (0..s.div_ceil(2)).flat_map(|dx| (0..2 * s).map(move |dy| (dx, dy))) {
                strip.put_pixel((x + dx).min(strip.width() - 1), top + dy, ink);
            }
            // Labels that would run off the right edge sit left of their tick.
            let label = format!("{minute}M");
            let width = label.len() as u32 * 4 * s;
            let lx = if x + s + width <= strip.width() {
                x + s
            } else {
                x.saturating_sub(width)
            };
            draw_text(strip, lx, top + 2 * s, &label, ink, s);
        }
    }

    fn segments_until(&self, label: &str, frame_idx: usize) -> Option<Vec<Vec<(f64, f64)>>> {
        let series = self.coords.get(label)?;
        let mut segments = Vec::new();
        let mut segment = Vec::new();
        let mut previous = None;
        for (&frame, &point) in self.frame_indices.iter().zip(series) {
            if frame > frame_idx as u64 {
                break;
            }
            if (point.is_none()
                || previous.is_some_and(|prev: u64| prev.checked_add(1) != Some(frame)))
                && !segment.is_empty()
            {
                segments.push(std::mem::take(&mut segment));
            }
            if let Some(point) = point {
                segment.push(point);
            }
            previous = Some(frame);
        }
        if !segment.is_empty() {
            segments.push(segment);
        }
        Some(segments)
    }
}
