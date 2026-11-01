#[cfg(test)]
mod ymaze_geometry_identity_tests {
    use super::YMazeOverrides;

    #[test]
    fn geometry_assertion_accepts_imen_and_rejects_conflicting_conventions() {
        let valid: YMazeOverrides = toml::from_str(
            "threshold_geometry = 'imen_radial_vertex_span_v1'\nproximal_threshold = 0.45",
        )
        .unwrap();
        assert!(valid.threshold_geometry.is_some());
        assert!(
            toml::from_str::<YMazeOverrides>("threshold_geometry = 'fraction_of_center_to_tip'")
                .is_err()
        );
    }
}
