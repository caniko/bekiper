fn build_object_recognition(
    perimeters: &[PerimeterSpec],
    params: TaskParams,
    object_protocol: Option<ObjectRecognitionProtocol>,
    overrides: Option<&dyn TaskOverrides>,
) -> Result<Box<dyn BehaviouralTask>> {
    let mut task = object_recognition_task(perimeters, params, object_protocol, overrides);
    if let Some(whisker) = overrides.and_then(|ov| ov.exploration_whisker()) {
        task.whisker_fan = whisker.params().map_err(BekiperError::Config)?;
        if task.whisker_fan.is_some() {
            task.whisker_age_days = whisker.age_days;
            task.whisker_length_source = whisker.length_source;
        }
    }
    Ok(Box::new(task))
}

fn object_recognition_task(
    perimeters: &[PerimeterSpec],
    params: TaskParams,
    object_protocol: Option<ObjectRecognitionProtocol>,
    overrides: Option<&dyn TaskOverrides>,
) -> ObjectRecognition {
    let object_perimeters: Vec<Perimeter<ObjectShape>> = perimeters
        .iter()
        .filter_map(|spec| {
            let (label, shape): (_, ObjectShape) = match spec {
                PerimeterSpec::Circle {
                    label,
                    center_x,
                    center_y,
                    radius,
                } => (label, Circle::new(*center_x, *center_y, *radius).into()),
                PerimeterSpec::Rectangle {
                    label,
                    center_x,
                    center_y,
                    width,
                    height,
                } => (
                    label,
                    Rectangle::new(*center_x, *center_y, *width, *height).into(),
                ),
                PerimeterSpec::Polygon { label, vertices } => {
                    (label, Polygon::new(vertices.clone()).into())
                }
                PerimeterSpec::Triangle { label, vertices } => (
                    label,
                    Triangle::new(vertices[0], vertices[1], vertices[2]).into(),
                ),
                PerimeterSpec::RadialMaze { .. } => return None,
            };
            Some(Perimeter::new(label.clone(), shape))
        })
        .collect();

    let mut task = ObjectRecognition::new(object_perimeters, params.fps);
    task.params = params;
    if let Some(protocol) = object_protocol {
        task.protocol = protocol;
    }

    // Apply exploration overrides
    if let Some(ov) = overrides {
        if let Some(v) = ov.exploration_nose_max_distance() {
            task.nose_max_distance = v;
        }
        if let Some(v) = ov.exploration_contact_distance() {
            task.contact_distance = v;
        }
        if let Some(v) = ov.exploration_max_angle_deg() {
            task.max_angle_deg = v;
        }
        if let Some(v) = ov.exploration_max_nose_speed() {
            task.max_nose_speed = v;
        }
        if let Some(v) = ov.exploration_speed_filter_mode() {
            task.speed_filter_mode = v;
        }
        if let Some(v) = ov.exploration_strict_boundaries() {
            task.strict_boundaries = v;
        }
        if let Some(v) = ov.exploration_angle_filter_mode() {
            task.angle_filter_mode = v;
        }
        if let Some(v) = ov.exploration_min_bout_seconds() {
            task.min_bout_seconds = v;
        }
        if let Some(v) = ov.exploration_max_gap_seconds() {
            task.max_gap_seconds = v;
        }
        if let Some(v) = ov.exploration_min_total_exploration() {
            task.min_total_exploration = v;
        }
        if let Some(v) = ov.exploration_analysis_window_seconds() {
            task.analysis_window_seconds = v;
        }
        if let Some(v) = ov.exploration_window_mode() {
            task.window_mode = v;
        }
        if let Some(v) = ov.exploration_perimeter_erosion_meters() {
            task.perimeter_erosion_meters = v;
        }
        if let Some(v) = ov.exploration_exclude_posture() {
            task.exclude_posture = v;
        }
        if let Some(v) = ov.exploration_gap_density_threshold() {
            task.gap_density_threshold = v;
        }
        if let Some(v) = ov.qualia_enable() {
            task.enable_qualia = v;
        }
        if let Some(v) = ov.qualia_body_proximity_distance() {
            task.qualia_body_proximity_distance = v;
        }
        if let Some(v) = ov.qualia_whisker_distance() {
            task.qualia_whisker_distance = v;
        }
        if let Some(v) = ov.qualia_whisker_angle() {
            task.qualia_whisker_angle = v;
        }
        if let Some(v) = ov.qualia_olfaction_distance() {
            task.qualia_olfaction_distance = v;
        }
        if let Some(v) = ov.qualia_olfaction_angle() {
            task.qualia_olfaction_angle = v;
        }
        if let Some(v) = ov.qualia_merge_rules() {
            task.merge_rules = v;
        }
    }

    // Center measurements require a saved outline; arena dimensions and scale
    // alone do not establish a reviewed center convention.
    if let Some(center) = task
        .object_perimeters
        .iter()
        .find(|p| p.label == "center_zone")
    {
        task.center_perimeter = Some(Perimeter::new("in_center", center.shape.clone()));
    }

    task
}
