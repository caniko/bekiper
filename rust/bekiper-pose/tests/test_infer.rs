#[cfg(feature = "infer")]
mod infer_tests {
    use bekiper_core::types::MidpointGroup;
    use bekiper_pose::PoseBackend;
    use bekiper_pose::infer::{
        ModelSpec, Normalization, OutputLayout, preprocess_frame, rescale_keypoints,
    };
    use bekiper_pose::skeleton::{Edge, Node, Skeleton};
    use image::{DynamicImage, RgbImage};

    fn dummy_image(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }))
    }

    fn test_spec() -> ModelSpec {
        ModelSpec {
            backend: PoseBackend::DeepLabCut,
            labels: vec!["nose".into(), "tail_base".into()],
            input_height: 64,
            input_width: 64,
            input_channels: 3,
            channels_first: true,
            normalization: Normalization::ImageNet {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
            output_layout: OutputLayout::KeypointsFlat,
            skeleton: Skeleton::new(
                "test",
                vec![Node::new("nose"), Node::new("tail_base")],
                vec![Edge::new("nose", "tail_base")],
            ),
            midpoint_groups: Vec::new(),
            label_aliases: Vec::new(),
        }
    }

    #[test]
    fn preprocess_produces_correct_shape_nchw() {
        let spec = test_spec();
        let image = dummy_image(640, 480);
        let (tensor, original_size) = preprocess_frame(&image, &spec);

        assert_eq!(tensor.shape(), &[1, 3, 64, 64]);
        assert_eq!(original_size, (640, 480));
    }

    #[test]
    fn preprocess_produces_correct_shape_nhwc() {
        let mut spec = test_spec();
        spec.channels_first = false;
        let image = dummy_image(640, 480);
        let (tensor, _) = preprocess_frame(&image, &spec);

        assert_eq!(tensor.shape(), &[1, 64, 64, 3]);
    }

    #[test]
    fn preprocess_imagenet_normalization_range() {
        let spec = test_spec();
        let image = dummy_image(64, 64);
        let (tensor, _) = preprocess_frame(&image, &spec);

        // ImageNet normalized values should be roughly in [-2.5, 2.5]
        let max_val = tensor.iter().cloned().reduce(f32::max).unwrap();
        let min_val = tensor.iter().cloned().reduce(f32::min).unwrap();
        assert!(
            max_val < 3.0,
            "max val {max_val} too high for ImageNet norm"
        );
        assert!(
            min_val > -3.0,
            "min val {min_val} too low for ImageNet norm"
        );
    }

    #[test]
    fn preprocess_zero_one_normalization_range() {
        let mut spec = test_spec();
        spec.normalization = Normalization::ZeroOne;
        let image = dummy_image(64, 64);
        let (tensor, _) = preprocess_frame(&image, &spec);

        let max_val = tensor.iter().cloned().reduce(f32::max).unwrap();
        let min_val = tensor.iter().cloned().reduce(f32::min).unwrap();
        assert!(max_val <= 1.0, "max val {max_val} exceeds 1.0 for 0-1 norm");
        assert!(min_val >= 0.0, "min val {min_val} below 0.0 for 0-1 norm");
    }

    #[test]
    fn rescale_keypoints_identity() {
        let (x, y) = rescale_keypoints(128.0, 128.0, 256, 256, 256, 256);
        assert!((x - 128.0).abs() < 1e-6);
        assert!((y - 128.0).abs() < 1e-6);
    }

    #[test]
    fn rescale_keypoints_upscale() {
        // Model is 256x256, original frame is 640x480
        let (x, y) = rescale_keypoints(128.0, 128.0, 256, 256, 640, 480);
        assert!((x - 320.0).abs() < 1e-3, "expected 320.0, got {x}");
        assert!((y - 240.0).abs() < 1e-3, "expected 240.0, got {y}");
    }

    #[test]
    fn dlc_super_animal_spec() {
        let spec = ModelSpec::dlc_super_animal();
        assert_eq!(spec.backend, PoseBackend::DeepLabCut);
        assert_eq!(spec.input_height, 256);
        assert_eq!(spec.input_width, 256);
        assert_eq!(spec.input_channels, 3);
        assert!(spec.channels_first);
        assert_eq!(spec.labels.len(), 27);
        assert!(matches!(spec.output_layout, OutputLayout::Heatmaps));
    }

    #[test]
    fn sleap_single_instance_spec() {
        let spec = ModelSpec::sleap_single_instance();
        assert_eq!(spec.backend, PoseBackend::Sleap);
        assert_eq!(spec.input_channels, 1);
        assert!(!spec.channels_first);
        assert_eq!(spec.labels.len(), 13);
        assert!(matches!(spec.output_layout, OutputLayout::Heatmaps));
    }

    #[test]
    fn lightning_pose_spec() {
        let spec = ModelSpec::lightning_pose();
        assert_eq!(spec.backend, PoseBackend::LightningPose);
        assert!(spec.channels_first);
        assert_eq!(spec.labels.len(), 27);
    }

    #[test]
    fn model_spec_serialization_roundtrip() {
        let spec = ModelSpec::dlc_super_animal();
        let toml_str = toml::to_string(&spec).unwrap();
        let deserialized: ModelSpec = toml::from_str(&toml_str).unwrap();
        assert_eq!(deserialized.backend, spec.backend);
        assert_eq!(deserialized.labels, spec.labels);
        assert_eq!(deserialized.input_height, spec.input_height);
        assert_eq!(deserialized.skeleton.labels(), spec.skeleton.labels());
        assert_eq!(
            deserialized.midpoint_groups.len(),
            spec.midpoint_groups.len()
        );
    }

    #[test]
    fn built_in_specs_expose_valid_skeletons() {
        for spec in [
            ModelSpec::dlc_super_animal(),
            ModelSpec::sleap_single_instance(),
            ModelSpec::lightning_pose(),
        ] {
            spec.validate().unwrap();
            assert!(!spec.skeleton.nodes.is_empty());
            assert!(
                spec.skeleton
                    .missing_labels(spec.labels.iter().map(String::as_str))
                    .is_empty()
            );
        }
    }

    #[test]
    fn custom_model_spec_without_skeleton_is_rejected() {
        let json = r#"{
            "backend": "deeplabcut",
            "labels": ["nose", "tail_base"],
            "input_height": 64,
            "input_width": 64,
            "input_channels": 3,
            "channels_first": true,
            "normalization": "ZeroOne",
            "output_layout": "KeypointsFlat"
        }"#;
        let result: std::result::Result<ModelSpec, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn custom_model_spec_requires_declared_midpoint_for_derived_skeleton_node() {
        let mut spec = ModelSpec {
            backend: PoseBackend::DeepLabCut,
            labels: vec!["nose".into(), "left_ear".into(), "right_ear".into()],
            input_height: 64,
            input_width: 64,
            input_channels: 3,
            channels_first: true,
            normalization: Normalization::ZeroOne,
            output_layout: OutputLayout::KeypointsFlat,
            skeleton: Skeleton::new(
                "derived",
                vec![
                    Node::new("nose"),
                    Node::new("left_ear"),
                    Node::new("right_ear"),
                    Node::new("center_ear"),
                ],
                vec![
                    Edge::new("nose", "center_ear"),
                    Edge::new("center_ear", "left_ear"),
                    Edge::new("center_ear", "right_ear"),
                ],
            ),
            midpoint_groups: Vec::new(),
            label_aliases: Vec::new(),
        };

        let err = spec.validate().unwrap_err().to_string();
        assert!(err.contains("missing center_ear"));

        spec.midpoint_groups = vec![MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["left_ear".into(), "right_ear".into()],
        }];
        spec.validate().unwrap();
    }

    #[test]
    fn custom_model_spec_midpoint_cannot_duplicate_raw_label() {
        let mut spec = test_spec();
        spec.midpoint_groups = vec![MidpointGroup {
            output_label: "nose".into(),
            source_labels: vec!["tail_base".into()],
        }];

        let err = spec.validate().unwrap_err().to_string();
        assert!(err.contains("duplicates a raw model label"));
    }
}
