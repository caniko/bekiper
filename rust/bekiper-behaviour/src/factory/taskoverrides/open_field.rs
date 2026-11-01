/// Resolve the OFT center perimeter and effective peripheral band.
///
/// Prefers an explicit `center_zone` perimeter (e.g. produced by the
/// `arena_calibration_cv` backend from `center_zone_meters`). Falls back to
/// deriving the center from `peripheral_band_fraction × arena_side`. The
/// returned `peripheral_band` is recomputed from whichever center wins so
/// downstream OFT code stays self-consistent.
///
/// Returns `(center_cx, center_cy, center_w, center_h, peripheral_band)`.
pub(crate) fn resolve_oft_center(
    perimeters: &[PerimeterSpec],
    arena_center: (f64, f64),
    arena_side: f64,
    peripheral_band_fraction: f64,
) -> (f64, f64, f64, f64, f64) {
    let (cx, cy, w, h) = extract_rect_dims(perimeters, "center_zone").unwrap_or_else(|| {
        let peripheral_band = arena_side * peripheral_band_fraction;
        let center_side = arena_side - 2.0 * peripheral_band;
        (arena_center.0, arena_center.1, center_side, center_side)
    });
    let peripheral_band = (arena_side - w.min(h)) / 2.0;
    (cx, cy, w, h, peripheral_band)
}

fn build_open_field(
    perimeters: &[PerimeterSpec],
    params: TaskParams,
    overrides: Option<&dyn TaskOverrides>,
) -> Result<Box<dyn BehaviouralTask>> {
    open_field_task(perimeters, params, overrides).map(|task| Box::new(task) as _)
}

fn open_field_task(
    perimeters: &[PerimeterSpec],
    params: TaskParams,
    overrides: Option<&dyn TaskOverrides>,
) -> Result<OpenFieldTest> {
    let (cx, cy, w, h) = extract_arena_dims(perimeters).ok_or_else(|| {
        BekiperError::Config("open_field_test requires a perimeter labeled 'arena'".into())
    })?;

    let arena_side = w.min(h);
    let (center_cx, center_cy, center_w, center_h, peripheral_band) = resolve_oft_center(
        perimeters,
        (cx, cy),
        arena_side,
        params.peripheral_band_fraction,
    );

    let arena = Perimeter::new("arena", Rectangle::new(cx, cy, w, h));
    let center = Perimeter::new(
        "in_center",
        Rectangle::new(center_cx, center_cy, center_w, center_h),
    );

    let mut task = OpenFieldTest::standard((cx, cy), arena_side, peripheral_band, params.fps);
    task.params = params;
    task.arena = arena;
    task.center = center;

    if let Some(ov) = overrides {
        if let Some(v) = ov.rearing_max_speed() {
            task.rearing_max_speed = v;
        }
        if let Some(v) = ov.rearing_max_body_length_ratio() {
            task.rearing_max_body_length_ratio = v;
        }
        if let Some(v) = ov.rearing_max_body_length() {
            task.rearing_max_body_length = v;
        }
        if let Some(v) = ov.rearing_forepaw_proximity() {
            task.rearing_forepaw_proximity = v;
        }
        if let Some(v) = ov.grooming_max_speed() {
            task.grooming_max_speed = v;
        }
        if let Some(v) = ov.grooming_max_body_length() {
            task.grooming_max_body_length = v;
        }
        if let Some(v) = ov.grooming_min_body_length() {
            task.grooming_min_body_length = v;
        }
        if let Some(v) = ov.grooming_forepaw_proximity() {
            task.grooming_forepaw_proximity = v;
        }
    }

    Ok(task)
}
