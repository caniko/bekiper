impl PoseModelDefinition for SleapSingleInstanceDefinition {
    fn backend(&self) -> PoseBackend {
        PoseBackend::Sleap
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
        1
    }

    fn channels_first(&self) -> bool {
        false
    }

    fn normalization(&self) -> Normalization {
        Normalization::ZeroOne
    }

    fn output_layout(&self) -> OutputLayout {
        OutputLayout::Heatmaps
    }

    fn skeleton(&self) -> Skeleton {
        Skeleton::mouse_full()
    }
}
