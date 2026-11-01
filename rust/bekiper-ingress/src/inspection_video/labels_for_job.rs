fn labels_for_job(manifest: &InspectionManifest, target: &JobTarget) -> Vec<Label> {
    let mut labels: BTreeSet<Label> = BTreeSet::new();
    match target {
        JobTarget::Overview | JobTarget::Scored | JobTarget::Qualia(_) => {
            for label in manifest.coordinate_columns.keys() {
                labels.insert(label.clone());
            }
        }
        JobTarget::Feature(feature) => match feature {
            FeatureInspection::HeuristicDecomposition {
                components,
                coordinate_labels,
                ..
            } => {
                labels.extend(coordinate_labels.iter().cloned());
                for component in components {
                    labels.extend(component.coordinate_labels.iter().cloned());
                    labels.extend(component.ray_origin_label.iter().cloned());
                }
            }
            FeatureInspection::Zone {
                coordinate_labels, ..
            }
            | FeatureInspection::Quadrant {
                coordinate_labels, ..
            }
            | FeatureInspection::Heuristic {
                coordinate_labels, ..
            }
            | FeatureInspection::Trajectory {
                coordinate_labels, ..
            }
            | FeatureInspection::YMazeArmSections {
                coordinate_labels, ..
            }
            | FeatureInspection::Proximity {
                coordinate_labels, ..
            } => {
                for label in coordinate_labels {
                    labels.insert(label.clone());
                }
            }
            FeatureInspection::Angle {
                from_coordinate_label,
                to_coordinate_label,
                ..
            } => {
                labels.insert(from_coordinate_label.clone());
                labels.insert(to_coordinate_label.clone());
            }
            FeatureInspection::RayCast { origin_label, .. } => {
                labels.insert(origin_label.clone());
            }
            _ => {
                for label in manifest.coordinate_columns.keys() {
                    labels.insert(label.clone());
                }
            }
        },
    }
    labels.into_iter().collect()
}

fn perimeter_labels_for_job(manifest: &InspectionManifest, target: &JobTarget) -> Vec<String> {
    match target {
        JobTarget::Overview | JobTarget::Scored => manifest
            .perimeters
            .iter()
            .map(|perimeter| perimeter.label().to_string())
            .collect(),
        JobTarget::Qualia(feature) => {
            perimeter_labels_for_job(manifest, &JobTarget::Feature(feature.clone()))
        }
        JobTarget::Feature(feature) => match feature {
            FeatureInspection::DetectionFields { object_labels, .. } => object_labels.clone(),
            FeatureInspection::HeuristicDecomposition { components, .. } => components
                .iter()
                .filter_map(|c| c.perimeter_label.clone())
                .collect(),
            FeatureInspection::Perimeter {
                perimeter_label, ..
            }
            | FeatureInspection::Zone {
                perimeter_label, ..
            }
            | FeatureInspection::YMazeArmSections {
                perimeter_label, ..
            }
            | FeatureInspection::Proximity {
                perimeter_label, ..
            } => vec![perimeter_label.clone()],
            FeatureInspection::Heuristic {
                perimeter_label: Some(perimeter_label),
                ..
            } => vec![perimeter_label.clone()],
            _ => Vec::new(),
        },
    }
}

/// Diagnostic channels the composite figures exclude from scored heuristics.
const DIAGNOSTIC_PREFIXES: [&str; 3] = ["body_proximity_", "olfaction_", "whisker_interaction_"];

/// `(heuristic, result column, perimeter label)` for each scored heuristic,
/// mirroring the composite figures: manifest heuristics without diagnostic
/// prefixes, plus grooming/rearing when only their decomposition is emitted.
/// The perimeter comes from the primary decomposition, else the heuristic feature.
fn scored_heuristics(manifest: &InspectionManifest) -> Vec<(String, String, Option<String>)> {
    let primary = |name: &str| {
        manifest
            .feature_inspections
            .iter()
            .find_map(|feature| match feature {
                FeatureInspection::HeuristicDecomposition {
                    label,
                    heuristic_name,
                    result_column,
                    perimeter_label,
                    ..
                } if heuristic_name == name
                    && *label == format!("heuristic_decomposition_{name}") =>
                {
                    Some((result_column.clone(), perimeter_label.clone()))
                }
                _ => None,
            })
    };
    let mut names: Vec<(String, String)> = manifest
        .heuristics
        .iter()
        .map(|meta| (meta.name.clone(), meta.result_column.clone()))
        .collect();
    for posture in ["grooming", "rearing"] {
        if !names.iter().any(|(name, _)| name == posture)
            && let Some((column, _)) = primary(posture)
        {
            names.push((posture.into(), column));
        }
    }
    names
        .into_iter()
        .filter(|(name, _)| !DIAGNOSTIC_PREFIXES.iter().any(|p| name.starts_with(p)))
        .map(|(name, column)| {
            let perimeter = primary(&name).and_then(|(_, label)| label).or_else(|| {
                manifest
                    .feature_inspections
                    .iter()
                    .find_map(|feature| match feature {
                        FeatureInspection::Heuristic {
                            result_column,
                            perimeter_label,
                            ..
                        } if *result_column == column => perimeter_label.clone(),
                        _ => None,
                    })
            });
            (name, column, perimeter)
        })
        .collect()
}

fn mask_color(value: Option<bool>) -> image::Rgb<u8> {
    parse_color(match value {
        Some(true) => themekit::DARK.true_mask,
        Some(false) => themekit::DARK.false_mask,
        None => themekit::DARK.unavailable,
    })
}

/// Strip column of a source frame over a recording of `frames` frames, matching
/// the bar spans `[floor(xF/B), max(floor((x+1)F/B), floor(xF/B)+1))`: the bar
/// holding the frame when `F > B`, else the first of the bars showing it.
fn strip_x(frame: u64, width: u32, frames: u64) -> u32 {
    let (f, b, n) = (
        u128::from(frame),
        u128::from(width),
        u128::from(frames.max(1)),
    );
    let x = if n > b {
        ((f + 1) * b - 1) / n
    } else {
        (f * b).div_ceil(n)
    };
    x.min(b - 1) as u32
}

/// Scored videos omit quadrant grids: their perimeter-blue lines coincide with
/// arena outlines and read as true-mask colour.
fn quadrant_labels_for_job(manifest: &InspectionManifest, target: &JobTarget) -> Vec<String> {
    match target {
        JobTarget::Overview => manifest
            .quadrants
            .iter()
            .map(|quadrant| quadrant.label.clone())
            .collect(),
        JobTarget::Feature(FeatureInspection::Quadrant { quadrant_label, .. }) => {
            vec![quadrant_label.clone()]
        }
        _ => Vec::new(),
    }
}

fn scaled_perimeter(
    manifest: &InspectionManifest,
    label: &str,
    scale: &FrameScale,
) -> Option<PerimeterSpec> {
    let pixel = manifest
        .pixel_perimeters
        .iter()
        .find(|perimeter| perimeter.label() == label)
        .cloned()
        .or_else(|| {
            manifest
                .perimeters
                .iter()
                .find(|perimeter| perimeter.label() == label)
                .map(|perimeter| {
                    if manifest.settings.meters_per_pixel > 0.0 {
                        perimeter.scale(1.0 / manifest.settings.meters_per_pixel)
                    } else {
                        perimeter.clone()
                    }
                })
        })?;

    Some(scale_perimeter(&pixel, scale.sx, scale.sy))
}

fn scaled_quadrant(
    manifest: &InspectionManifest,
    label: &str,
    scale: &FrameScale,
) -> Option<ScaledQuadrant> {
    let quad = manifest
        .pixel_quadrants
        .iter()
        .find(|quadrant| quadrant.label == label)
        .cloned()
        .or_else(|| {
            manifest
                .quadrants
                .iter()
                .find(|quadrant| quadrant.label == label)
                .map(|quadrant| {
                    let factor = if manifest.settings.meters_per_pixel > 0.0 {
                        1.0 / manifest.settings.meters_per_pixel
                    } else {
                        1.0
                    };
                    let mut scaled = quadrant.clone();
                    scaled.x_min *= factor;
                    scaled.y_min *= factor;
                    scaled.x_max *= factor;
                    scaled.y_max *= factor;
                    scaled
                })
        })?;

    Some(ScaledQuadrant {
        x_min: quad.x_min * scale.sx,
        y_min: quad.y_min * scale.sy,
        x_max: quad.x_max * scale.sx,
        y_max: quad.y_max * scale.sy,
        rows: quad.rows,
        cols: quad.cols,
    })
}

fn expand_perimeter_pixels(
    perimeter: &PerimeterSpec,
    buffer_meters: f64,
    bundle: &LoadedInspection,
    frame_size: (u32, u32),
) -> Option<PerimeterSpec> {
    // Expand in calibration space, then apply the full display transform.
    // Offsetting an already-distorted polygon changes the native buffer.
    let source = scaled_perimeter(
        &bundle.manifest,
        perimeter.label(),
        &FrameScale { sx: 1.0, sy: 1.0 },
    )?;
    let distance = metric_length_to_pixels(
        buffer_meters,
        bundle.manifest.settings.meters_per_pixel,
        1.0,
    );
    let expanded = match &source {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => Some(PerimeterSpec::Circle {
            label: label.clone(),
            center_x: *center_x,
            center_y: *center_y,
            radius: radius + distance,
        }),
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } => {
            let poly = Rectangle::new(*center_x, *center_y, *width, *height).expand(distance);
            Some(PerimeterSpec::Polygon {
                label: label.clone(),
                vertices: poly.vertices,
            })
        }
        PerimeterSpec::Polygon { label, vertices } => {
            let poly = Polygon::new(vertices.clone()).expand(distance);
            Some(PerimeterSpec::Polygon {
                label: label.clone(),
                vertices: poly.vertices,
            })
        }
        PerimeterSpec::Triangle { label, vertices } => {
            let poly = Triangle::new(vertices[0], vertices[1], vertices[2]).expand(distance);
            Some(PerimeterSpec::Polygon {
                label: label.clone(),
                vertices: poly.vertices,
            })
        }
        PerimeterSpec::RadialMaze { .. } => None,
    };
    let scale = frame_scale(&bundle.manifest, frame_size.0, frame_size.1);
    expanded.map(|perimeter| scale_perimeter(&perimeter, scale.sx, scale.sy))
}

fn metric_coord_to_pixels(value: f64, meters_per_pixel: f64, scale: f64) -> f64 {
    if meters_per_pixel > 0.0 {
        (value / meters_per_pixel) * scale
    } else {
        value * scale
    }
}

fn metric_length_to_pixels(value: f64, meters_per_pixel: f64, scale: f64) -> f64 {
    if meters_per_pixel > 0.0 {
        (value / meters_per_pixel) * scale
    } else {
        value * scale
    }
}

fn bool_color(theme: &VideoTheme, active: bool) -> image::Rgb<u8> {
    if active {
        parse_color(&theme.video_true_color)
    } else {
        parse_color(&theme.video_false_color)
    }
}

fn theme_thickness(width: f64) -> i32 {
    width.round().max(1.0) as i32
}

fn resolved_feature_thickness(feature: &FeatureInspection, default: f64) -> i32 {
    theme_thickness(feature.linewidth().unwrap_or(default))
}

fn resolved_feature_color(feature: &FeatureInspection, default: &str) -> image::Rgb<u8> {
    feature
        .color()
        .map(parse_color)
        .unwrap_or_else(|| parse_color(default))
}

fn validate_options(opts: &RenderOptions) -> Result<()> {
    let selector_count = opts.heuristic.iter().count()
        + opts.feature.iter().count()
        + opts.all_features as usize
        + opts.scored as usize
        + opts.qualia as usize;
    if selector_count > 1 {
        return Err(BekiperError::Config(
            "--heuristic, --feature, --all-features, --scored, and --qualia are mutually exclusive"
                .into(),
        ));
    }
    if opts.upscale.is_some() && !opts.qualia {
        return Err(BekiperError::Config("--upscale needs --qualia".into()));
    }
    if opts.crf.is_some() && !encoder_supports_crf(map_codec(&opts.codec)) {
        return Err(BekiperError::Config(format!(
            "--crf is not supported by codec '{}'",
            opts.codec
        )));
    }
    if let Some(upscale) = opts.upscale
        && !(1..=MAX_UPSCALE).contains(&upscale)
    {
        return Err(BekiperError::Config(format!(
            "--upscale must be 1..={MAX_UPSCALE}, got {upscale}"
        )));
    }
    if let Some(crf) = opts.crf {
        let max = crf_max(map_codec(&opts.codec));
        if crf > max {
            return Err(BekiperError::Config(format!(
                "--crf {crf} is out of range 0..={max} for codec '{}'",
                opts.codec
            )));
        }
    }
    if opts.output.is_some() && opts.output_dir.is_some() {
        return Err(BekiperError::Config(
            "--output and --output-dir are mutually exclusive".into(),
        ));
    }
    if opts.all_features {
        if opts.output.is_some() {
            return Err(BekiperError::Config(
                "--output cannot be used with --all-features".into(),
            ));
        }
    } else if opts.output_dir.is_some() {
        return Err(BekiperError::Config(
            "--output-dir requires --all-features".into(),
        ));
    }
    Ok(())
}

fn build_jobs(bundle: &LoadedInspection, opts: &RenderOptions) -> Result<Vec<RenderJob>> {
    let default_dir = bundle.dir.join("videos");
    if opts.all_features {
        let mut skipped = Vec::new();
        let mut jobs = vec![RenderJob {
            target: JobTarget::Overview,
            output_path: (opts
                .output_dir
                .clone()
                .unwrap_or_else(|| default_dir.clone()))
            .join("overview.inspection.mp4"),
        }];

        for feature in &bundle.manifest.feature_inspections {
            if is_renderable_feature(feature) {
                jobs.push(RenderJob {
                    target: JobTarget::Feature(feature.clone()),
                    output_path: (opts
                        .output_dir
                        .clone()
                        .unwrap_or_else(|| default_dir.clone()))
                    .join(format!(
                        "feature_{}.inspection.mp4",
                        sanitize_for_filename(feature.label())
                    )),
                });
            } else {
                skipped.push(feature.label().to_string());
            }
        }

        if !skipped.is_empty() {
            tracing::warn!(skipped = ?skipped, "skipping unsupported inspection-video features");
        }

        if jobs.len() == 1 {
            return Err(BekiperError::Config(
                "no renderable feature videos available".into(),
            ));
        }

        return Ok(jobs);
    }

    if let Some(feature_label) = &opts.feature {
        let feature = bundle
            .manifest
            .feature_inspections
            .iter()
            .find(|feature| feature.label() == feature_label)
            .ok_or_else(|| {
                BekiperError::Config(format!("unknown feature label: {feature_label}"))
            })?;
        if !is_renderable_feature(feature) {
            return Err(BekiperError::Config(format!(
                "feature '{}' is not supported by v1 inspection video",
                feature.label()
            )));
        }
        let output_path = opts.output.clone().unwrap_or_else(|| {
            default_dir.join(format!(
                "feature_{}.inspection.mp4",
                sanitize_for_filename(feature.label())
            ))
        });
        return Ok(vec![RenderJob {
            target: JobTarget::Feature(feature.clone()),
            output_path,
        }]);
    }

    if let Some(heuristic_name) = &opts.heuristic {
        if let Some(feature) = bundle
            .manifest
            .feature_inspections
            .iter()
            .find(|f| f.label() == format!("multiobject_{heuristic_name}"))
        {
            return Ok(vec![RenderJob {
                target: JobTarget::Feature(feature.clone()),
                output_path: opts.output.clone().unwrap_or_else(|| {
                    default_dir.join(format!(
                        "heuristic_{}.inspection.mp4",
                        sanitize_for_filename(heuristic_name)
                    ))
                }),
            }]);
        }
        let heuristic = bundle
            .manifest
            .heuristics
            .iter()
            .find(|meta| meta.name == *heuristic_name)
            .ok_or_else(|| BekiperError::Config(format!("unknown heuristic: {heuristic_name}")))?;

        if let Some(feature) = bundle.manifest.feature_inspections.iter().find(|feature| {
            feature.label() == format!("heuristic_decomposition_{heuristic_name}")
                && matches!(feature, FeatureInspection::HeuristicDecomposition { components, .. }
                    if components.iter().any(|component| component.ray_origin_label.is_some()))
        }) {
            let mut feature = feature.clone();
            if let FeatureInspection::HeuristicDecomposition {
                components,
                result_column,
                show_final_result,
                ..
            } = &mut feature
            {
                components.retain(|component| component.ray_origin_label.is_some());
                for component in components {
                    component.result_column = heuristic.result_column.clone();
                }
                *result_column = heuristic.result_column.clone();
                *show_final_result = false;
            }
            return Ok(vec![RenderJob {
                target: JobTarget::Feature(feature),
                output_path: opts.output.clone().unwrap_or_else(|| {
                    default_dir.join(format!(
                        "heuristic_{}.inspection.mp4",
                        sanitize_for_filename(heuristic_name)
                    ))
                }),
            }]);
        }

        let matches: Vec<_> = bundle
            .manifest
            .feature_inspections
            .iter()
            .filter(|feature| {
                is_renderable_feature(feature)
                    && !matches!(feature, FeatureInspection::HeuristicDecomposition { .. })
                    && feature.result_column() == Some(heuristic.result_column.as_str())
            })
            .cloned()
            .collect();

        if matches.is_empty() {
            return Err(BekiperError::Config(format!(
                "heuristic '{}' has no renderable inspection-video feature",
                heuristic_name
            )));
        }
        if matches.len() > 1 {
            let candidates = matches
                .iter()
                .map(|feature| feature.label().to_string())
                .collect::<Vec<_>>();
            return Err(BekiperError::Config(format!(
                "heuristic '{}' is ambiguous; choose one of: {}",
                heuristic_name,
                candidates.join(", ")
            )));
        }

        let output_path = opts.output.clone().unwrap_or_else(|| {
            default_dir.join(format!(
                "heuristic_{}.inspection.mp4",
                sanitize_for_filename(heuristic_name)
            ))
        });
        return Ok(vec![RenderJob {
            target: JobTarget::Feature(matches.into_iter().next().unwrap()),
            output_path,
        }]);
    }

    let (target, name) = if opts.qualia {
        let fields = bundle
            .manifest
            .feature_inspections
            .iter()
            .find(|feature| matches!(feature, FeatureInspection::DetectionFields { .. }))
            .ok_or_else(|| {
                BekiperError::Config(
                    "--qualia needs a `detection_fields` inspection (NORT trials only)".into(),
                )
            })?;
        (JobTarget::Qualia(fields.clone()), "qualia.inspection.mp4")
    } else if opts.scored {
        (JobTarget::Scored, "scored.inspection.mp4")
    } else {
        (JobTarget::Overview, "overview.inspection.mp4")
    };
    Ok(vec![RenderJob {
        target,
        output_path: opts
            .output
            .clone()
            .unwrap_or_else(|| default_dir.join(name)),
    }])
}

fn is_renderable_feature(feature: &FeatureInspection) -> bool {
    matches!(
        feature,
        FeatureInspection::Perimeter { .. }
            | FeatureInspection::Zone { .. }
            | FeatureInspection::Quadrant { .. }
            | FeatureInspection::Heuristic { .. }
            | FeatureInspection::Angle { .. }
            | FeatureInspection::Trajectory { .. }
            | FeatureInspection::RayCast { .. }
            | FeatureInspection::Proximity { .. }
            | FeatureInspection::HeuristicDecomposition { .. }
    )
}
