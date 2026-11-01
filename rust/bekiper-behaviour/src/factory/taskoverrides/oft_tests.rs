#[cfg(test)]
mod resolve_oft_center_tests {
    use super::{PerimeterSpec, resolve_oft_center};
    use bekiper_core::types::TaskType;

    fn rect(label: &str, side: f64) -> PerimeterSpec {
        PerimeterSpec::Rectangle {
            label: label.into(),
            center_x: 0.0,
            center_y: 0.0,
            width: side,
            height: side,
        }
    }

    #[test]
    fn explicit_center_zone_perimeter_wins_over_peripheral_band_fraction() {
        // arena = 40 cm, explicit center = 24 cm (criteria-doc value).
        // peripheral_band_fraction = 0.5 would imply a degenerate fallback,
        // so if the explicit center isn't honoured this test fails loudly.
        let perimeters = [rect("arena", 0.40), rect("center_zone", 0.24)];
        let (cx, cy, w, h, band) =
            resolve_oft_center(&perimeters, (0.0, 0.0), 0.40, /* fallback */ 0.5);
        assert_eq!((cx, cy), (0.0, 0.0));
        assert!((w - 0.24).abs() < 1e-9, "center width = {w}");
        assert!((h - 0.24).abs() < 1e-9, "center height = {h}");
        // Band recomputed from chosen center, not the (deliberately wrong)
        // fallback fraction.
        assert!((band - 0.08).abs() < 1e-9, "peripheral band = {band}");
    }

    #[test]
    fn falls_back_to_peripheral_band_fraction_when_no_center_zone() {
        // arena = 40 cm, peripheral_band_fraction = 0.2 → center 24 cm.
        let perimeters = [rect("arena", 0.40)];
        let (_, _, w, h, band) = resolve_oft_center(&perimeters, (0.0, 0.0), 0.40, 0.2);
        let expected_center = 0.40 - 2.0 * (0.40 * 0.2);
        assert!((w - expected_center).abs() < 1e-9);
        assert!((h - expected_center).abs() < 1e-9);
        assert!((band - 0.08).abs() < 1e-9);
    }

    #[test]
    fn ignores_non_center_zone_perimeters() {
        // Foreign labels (objects, start areas) must not be treated as the
        // center perimeter — only "center_zone" should win.
        let perimeters = [
            rect("arena", 0.40),
            rect("object_1", 0.05),
            rect("start_area", 0.10),
        ];
        let (_, _, w, _, _) = resolve_oft_center(&perimeters, (0.0, 0.0), 0.40, 0.2);
        let expected_center = 0.40 - 2.0 * (0.40 * 0.2);
        assert!(
            (w - expected_center).abs() < 1e-9,
            "should fall back, not pick a foreign perimeter"
        );
    }

    #[test]
    fn build_open_field_requires_arena_perimeter() {
        let params = super::TaskParams::new(30.0);
        let result = super::build_task(TaskType::OpenField, &[], params, None, None, None);
        assert!(result.is_err());
    }
}
