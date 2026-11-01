#[derive(Debug, Clone)]
struct LoadedInspection {
    dir: PathBuf,
    manifest: InspectionManifest,
    df: DataFrame,
    video_path: PathBuf,
    /// The manifest's `video_path`, resolved against `dir`, when one is recorded.
    recorded_video_path: Option<PathBuf>,
}

impl LoadedInspection {
    fn load(dir: &Path, source_override: Option<&Path>) -> Result<Self> {
        let (toml_path, parquet_path) = find_inspection_pair(dir)?;
        let manifest = InspectionManifest::from_toml(&toml_path)?;
        let df =
            LazyFrame::scan_parquet(parquet_path.to_str().unwrap().into(), Default::default())?
                .collect()?;

        let recorded_video_path = manifest.video_path.as_ref().map(|path| dir.join(path));
        let resolved_video = source_override
            .map(Path::to_path_buf)
            .or_else(|| recorded_video_path.clone())
            .ok_or_else(|| {
                BekiperError::Data("Video path not set in inspection manifest".into())
            })?;
        let df = crate::inspection::source_overlay_dataframe(&manifest, df)?;
        let manifest = manifest.source_overlay_geometry();
        if !resolved_video.exists() {
            return Err(BekiperError::Data(format!(
                "Video file not found: {}",
                resolved_video.display()
            )));
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            df,
            video_path: resolved_video,
            recorded_video_path,
        })
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum JobTarget {
    Overview,
    Scored,
    /// The manifest's `DetectionFields`, drawn with a per-object channel strip.
    Qualia(FeatureInspection),
    Feature(FeatureInspection),
}

#[derive(Debug, Clone)]
struct RenderJob {
    target: JobTarget,
    output_path: PathBuf,
}

impl RenderJob {
    fn display_name(&self) -> String {
        match &self.target {
            JobTarget::Overview => "overview".into(),
            JobTarget::Scored => "scored".into(),
            JobTarget::Qualia(_) => "qualia".into(),
            JobTarget::Feature(feature) => format!("feature:{}", feature.label()),
        }
    }
}
