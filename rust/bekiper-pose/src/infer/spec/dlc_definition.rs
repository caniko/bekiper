impl PoseModelDefinition for DlcSuperAnimalDefinition {
    fn backend(&self) -> PoseBackend {
        PoseBackend::DeepLabCut
    }

    fn labels(&self) -> Vec<String> {
        self.skeleton().labels()
    }

    fn input_height(&self) -> u32 {
        256
    }

    fn input_width(&self) -> u32 {
        256
    }

    fn input_channels(&self) -> u32 {
        3
    }

    fn channels_first(&self) -> bool {
        true
    }

    fn normalization(&self) -> Normalization {
        Normalization::ImageNet {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    fn output_layout(&self) -> OutputLayout {
        OutputLayout::Heatmaps
    }

    fn skeleton(&self) -> Skeleton {
        Skeleton::mouse_top_view()
    }
}
