#[cfg(feature = "pose-gpu")]
fn apply_redetected_candidates_with<F>(
    fitted: &FittedSkeletonRepair,
    requests: &[FrameSkeletonViolations],
    likelihood_threshold: f64,
    repaired_df: &mut DataFrame,
    final_eval: &mut SkeletonViolationEvaluation,
    redetected_rows: &mut BTreeSet<usize>,
    mut predictor: F,
) -> Result<()>
where
    F: FnMut(i64) -> Result<Option<HashMap<String, RedetectedKeypoint>>>,
{
    let mut updated_columns: HashMap<String, Vec<Option<f64>>> = HashMap::new();

    for request in requests {
        redetected_rows.insert(request.row_index);

        let Some(prediction) = predictor(request.frame)? else {
            continue;
        };

        let mut candidates: HashMap<usize, RedetectedKeypoint> = HashMap::new();
        for label in &request.labels {
            let Some(label_idx) = fitted.label_index(label) else {
                continue;
            };
            let Some(candidate) = prediction.get(label) else {
                continue;
            };
            if candidate.likelihood >= likelihood_threshold {
                candidates.insert(label_idx, *candidate);
            }
        }

        if candidates.is_empty() {
            continue;
        }

        let flagged_indices: Vec<usize> = request
            .labels
            .iter()
            .filter_map(|label| fitted.label_index(label))
            .collect();
        let mut accepted: BTreeSet<usize> = candidates.keys().copied().collect();

        loop {
            let mut row_coords = fitted.row_coordinates(repaired_df, request.row_index)?;
            for label_idx in &flagged_indices {
                row_coords[*label_idx] = None;
            }
            for label_idx in &accepted {
                let candidate = candidates[label_idx];
                row_coords[*label_idx] = Some((candidate.x, candidate.y));
            }

            let violated = fitted.evaluate_frame(&row_coords);
            let next_accepted: BTreeSet<usize> = accepted
                .iter()
                .copied()
                .filter(|label_idx| !violated.contains(label_idx))
                .collect();

            if next_accepted == accepted {
                break;
            }
            accepted = next_accepted;
            if accepted.is_empty() {
                break;
            }
        }

        for label_idx in accepted {
            let label = fitted.label_name(label_idx);
            let candidate = candidates[&label_idx];

            set_numeric_value(
                repaired_df,
                &mut updated_columns,
                &format!("{label}_x"),
                request.row_index,
                candidate.x,
            )?;
            set_numeric_value(
                repaired_df,
                &mut updated_columns,
                &format!("{label}_y"),
                request.row_index,
                candidate.y,
            )?;
            set_numeric_value(
                repaired_df,
                &mut updated_columns,
                &format!("{label}_likelihood"),
                request.row_index,
                candidate.likelihood,
            )?;

            final_eval.clear_label(request.row_index, label);
        }
    }

    for (column_name, values) in updated_columns {
        repaired_df.with_column(Column::new(column_name.into(), values))?;
    }

    Ok(())
}

#[cfg(feature = "pose-gpu")]
fn set_numeric_value(
    df: &DataFrame,
    updated_columns: &mut HashMap<String, Vec<Option<f64>>>,
    column_name: &str,
    row_index: usize,
    value: f64,
) -> Result<()> {
    if !updated_columns.contains_key(column_name) {
        let values = df
            .column(column_name)?
            .f64()
            .map_err(|_| BekiperError::Data(format!("{column_name} must be Float64")))?
            .into_iter()
            .collect();
        updated_columns.insert(column_name.to_string(), values);
    }

    if let Some(column) = updated_columns.get_mut(column_name) {
        column[row_index] = Some(value);
    }

    Ok(())
}

#[cfg(feature = "pose-gpu")]
fn redetect_frame(
    decoder: &mut VideoDecoder,
    model: &mut OnnxPoseModel,
    crop_region: Option<bekiper_pose::infer::CropRegion>,
    frame: i64,
) -> Result<Option<HashMap<String, RedetectedKeypoint>>> {
    let frame = u64::try_from(frame).map_err(|_| {
        BekiperError::Data(format!("cannot re-detect negative frame index {frame}"))
    })?;
    decoder.seek_to(frame)?;

    let Some((_, image)) = decoder.next_frame()? else {
        return Ok(None);
    };

    let image = if let Some(crop) = crop_region {
        crop.crop(&image)
    } else {
        image
    };
    let keypoints = model.infer_frame(&image)?;

    Ok(Some(
        model
            .labels()
            .iter()
            .zip(keypoints.iter())
            .map(|(label, keypoint)| (label.clone(), offset_keypoint(*keypoint, crop_region)))
            .collect(),
    ))
}

#[cfg(feature = "pose-gpu")]
fn offset_keypoint(
    keypoint: Keypoint,
    crop_region: Option<bekiper_pose::infer::CropRegion>,
) -> RedetectedKeypoint {
    let (offset_x, offset_y) = crop_region
        .map(|crop| (crop.x as f64, crop.y as f64))
        .unwrap_or((0.0, 0.0));

    RedetectedKeypoint {
        x: keypoint.x as f64 + offset_x,
        y: keypoint.y as f64 + offset_y,
        likelihood: keypoint.confidence as f64,
    }
}

/// Extract the arena perimeter as an `ObjectShape` for confinement nulling.
/// Returns `None` if no perimeter labeled "arena" exists.
fn find_arena_shape(
    perimeters: &[PerimeterSpec],
) -> Option<bekiper_perimeter::object_shape::ObjectShape> {
    use bekiper_perimeter::circle::Circle;
    use bekiper_perimeter::object_shape::ObjectShape;
    use bekiper_perimeter::polygon::Polygon;
    use bekiper_perimeter::rectangle::Rectangle;
    use bekiper_perimeter::triangle::Triangle;

    perimeters.iter().find_map(|spec| match spec {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } if label == "arena" => Some(ObjectShape::Circle(Circle::new(
            *center_x, *center_y, *radius,
        ))),
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } if label == "arena" => Some(ObjectShape::Rectangle(Rectangle::new(
            *center_x, *center_y, *width, *height,
        ))),
        PerimeterSpec::Polygon { label, vertices } if label == "arena" => {
            Some(ObjectShape::Polygon(Polygon::new(vertices.clone())))
        }
        PerimeterSpec::Triangle { label, vertices } if label == "arena" => Some(
            ObjectShape::Triangle(Triangle::new(vertices[0], vertices[1], vertices[2])),
        ),
        _ => None,
    })
}
