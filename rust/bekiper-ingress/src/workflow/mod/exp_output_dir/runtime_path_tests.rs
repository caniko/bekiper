#[cfg(test)]
mod runtime_path_tests {
    use super::*;

    #[test]
    fn preflight_diagnostics_keep_fifty_details_and_stage_specific_remainders() {
        let errors: Vec<_> = (0..52).map(|index| format!("BROKEN[{index}]")).collect();
        for (format, remainder) in [
            (
                format_detect_output_errors as fn(&[String]) -> String,
                "missing checkpoint(s)",
            ),
            (format_no_detect_errors, "missing input(s)"),
            (format_no_auto_detect_errors, "input(s)"),
        ] {
            let message = format(&errors);
            assert_eq!(message.lines().count(), 52);
            assert!(message.contains("BROKEN[0]\n"));
            assert!(message.contains("BROKEN[49]\n"));
            assert!(!message.contains("BROKEN[50]"));
            assert!(message.ends_with(&format!("... and 2 more {remainder}")));
            assert_eq!(format(&errors[..50]).lines().count(), 51);
            assert!(!format(&errors[..50]).contains("... and"));
            assert_eq!(format(&[]).lines().count(), 1);
        }
    }

    #[test]
    fn resolve_runtime_path_rebases_relative_paths() {
        let base_dir = Path::new("/tmp/bekiper-config-root");
        assert_eq!(
            resolve_runtime_path(base_dir, Path::new("NORT videos/video.mp4")),
            base_dir.join("NORT videos/video.mp4")
        );
    }

    #[test]
    fn resolve_runtime_path_preserves_absolute_paths() {
        let base_dir = Path::new("/tmp/bekiper-config-root");
        let absolute = Path::new("/data/videos/video.mp4");
        assert_eq!(
            resolve_runtime_path(base_dir, absolute),
            PathBuf::from("/data/videos/video.mp4")
        );
    }

    #[test]
    fn resolve_runtime_path_preserves_already_rebased_relative_paths() {
        let base_dir = Path::new("examples/neo/imen");
        let rebased = Path::new("examples/neo/imen/NORT videos/video.mp4");
        assert_eq!(resolve_runtime_path(base_dir, rebased), rebased);
    }
}
