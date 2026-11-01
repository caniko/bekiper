#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn shared_radial_edge_keeps_degenerate_fallback_and_last_tie_behavior() {
        assert_eq!(radial_arm_distal_edge(&[], &[]), None);
        assert_eq!(
            radial_arm_distal_edge(&[], &[(2.0, 3.0)]),
            Some(((2.0, 3.0), (2.0, 3.0)))
        );
        assert_eq!(
            radial_arm_distal_edge(&[], &[(2.0, 3.0), (4.0, 5.0)]),
            Some(((2.0, 3.0), (4.0, 5.0)))
        );
        let square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let left = Some(((-1.0, 1.0), (-1.0, -1.0)));
        assert_eq!(radial_arm_distal_edge(&[], &square), left);
        assert_eq!(radial_arm_distal_edge(&[(4.0, 0.0)], &square), left);
        assert_eq!(
            radial_arm_distal_edge(&[(-4.0, 0.0)], &square),
            Some(((1.0, -1.0), (1.0, 1.0)))
        );
    }

    include!("tests/make_temp_dir.rs");
    include!("tests/session_refresh_updates_metadata_mapping_mode.rs");

    #[test]
    fn target_edits_are_deferred_and_preserve_shape_type() {
        let dir = make_temp_dir();
        let plugin = dir.join("plugin_files");
        std::fs::create_dir_all(&plugin).unwrap();
        std::fs::write(dir.join("metadata.csv"), "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName\nA,2,47,1,1,1,,training_1,A1\n").unwrap();
        std::fs::write(
            plugin.join("perimeter-rectangle-training_1.csv"),
            "familiar,10,20,30,40,A1.png,100,100\n",
        )
        .unwrap();
        let video = Path::new("post.T1.configA.video-1.mp4");
        let mut session = AnnotationSession::load(&dir, video).unwrap();
        let baseline = session.clone();
        let polygon = PerimeterSpec::Polygon {
            label: "variable".into(),
            vertices: vec![(10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0)],
        };
        session
            .set_target_geometry_from_resolved(std::slice::from_ref(&polygon))
            .unwrap();
        assert_eq!(
            AnnotationSession::load(&dir, video).unwrap().resolved_specs,
            baseline.resolved_specs
        );
        assert!(!CanonicalAnnotationStore::exists(&dir));
        session.save().unwrap();
        assert_eq!(
            AnnotationSession::load(&dir, video).unwrap().resolved_specs,
            vec![polygon]
        );
    }
}
