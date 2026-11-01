use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, anyhow, bail};
use clap::{Args, Subcommand};

mod export;

// ── `bekiper inspect` CLI defaults ──────────────────────────────────────────
// These are surfaced as clap `default_value = ...` attributes, which must
// be string literals at compile time. They are promoted to named consts
// so the choices are grep-able in one place.

/// Default directory walked by every `bekiper inspect` subcommand.
const DEFAULT_INSPECTION_DIR: &str = "inspection";

/// Default video codec used by `bekiper inspect video`. `h264` is the
/// broadest-compatibility option; users wanting GPU-accelerated output
/// can pass `--codec hevc_nvenc` or `--codec av1_qsv`.
#[cfg(feature = "full")]
const DEFAULT_VIDEO_CODEC: &str = "h264";

#[derive(Debug)]
struct AnnotationValidation {
    dir: PathBuf,
    mapping_rows: usize,
    perimeter_ids: usize,
    loaded_specs: usize,
    rereferenced_rows: usize,
}

fn resolve_annotation_directories(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }

    let file_name = path.file_name().and_then(|name| name.to_str());
    match file_name {
        Some("metadata.csv") | Some("video_map.csv") => {
            return Ok(vec![
                path.parent()
                    .ok_or_else(|| anyhow!("annotation file has no parent directory"))?
                    .to_path_buf(),
            ]);
        }
        _ => {}
    }

    if path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        == Some("plugin_files")
    {
        return Ok(vec![
            path.parent()
                .and_then(|parent| parent.parent())
                .ok_or_else(|| anyhow!("plugin file has no annotation directory parent"))?
                .to_path_buf(),
        ]);
    }

    if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
        let config = bekiper_ingress::config::IngressConfig::from_file(path)
            .with_context(|| format!("failed to parse config at {}", path.display()))?;
        let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let dirs: BTreeSet<PathBuf> = config
            .scan
            .iter()
            .filter_map(|scan| scan.annotations.as_ref().map(|ann| base_dir.join(&ann.dir)))
            .collect();

        if dirs.is_empty() {
            bail!(
                "config '{}' does not declare any scan.annotations directories",
                path.display()
            );
        }

        return Ok(dirs.into_iter().collect());
    }

    Err(anyhow!(
        "could not infer annotation directory from '{}'; pass an annotation directory, metadata/video_map CSV, plugin_files entry, or bekiper_config.toml",
        path.display()
    ))
}

fn validate_annotation_dir(dir: &std::path::Path) -> anyhow::Result<AnnotationValidation> {
    let _resolver = bekiper_ingress::annotation::AnnotationResolver::from_dir(dir)
        .with_context(|| format!("failed to load annotation directory {}", dir.display()))?;
    let plugin_dir = dir.join("plugin_files");

    let video_map_path = dir.join("video_map.csv");
    let metadata_path = dir.join("metadata.csv");

    let mut ids = BTreeSet::new();
    let mut mapping_rows = 0usize;
    let mut rereferenced_rows = 0usize;

    if video_map_path.exists() {
        let map = bekiper_ingress::annotation::read_video_map(&video_map_path)
            .with_context(|| format!("failed to read {}", video_map_path.display()))?;
        mapping_rows = map.len();
        ids.extend(map.into_values());
    } else if metadata_path.exists() {
        let rows = bekiper_ingress::annotation::read_metadata_csv(&metadata_path)
            .with_context(|| format!("failed to read {}", metadata_path.display()))?;
        mapping_rows = rows.len();
        for row in rows {
            let Some(source) = row.source else {
                continue;
            };

            match source {
                bekiper_ingress::annotation::PerimeterSource::Direct { perimeter_id } => {
                    ids.insert(perimeter_id);
                }
                bekiper_ingress::annotation::PerimeterSource::ReReferenced {
                    perimeter_id, ..
                } => {
                    rereferenced_rows += 1;
                    bekiper_ingress::annotation::load_source_reference(&plugin_dir, &perimeter_id)
                        .with_context(|| {
                            format!(
                            "missing source reference for re-referenced perimeter '{perimeter_id}'"
                        )
                        })?;
                    let change_ref =
                        plugin_dir.join(format!("change_reference-{perimeter_id}.csv"));
                    if !change_ref.exists() {
                        bail!(
                            "missing change_reference file for re-referenced perimeter '{}' in {}",
                            perimeter_id,
                            plugin_dir.display()
                        );
                    }
                    ids.insert(perimeter_id);
                }
            }
        }
    }

    let mut loaded_specs = 0usize;
    for id in &ids {
        loaded_specs += bekiper_ingress::annotation::load_perimeter_by_id(&plugin_dir, id)
            .with_context(|| format!("failed to load perimeter '{id}' from {}", dir.display()))?
            .len();
    }

    Ok(AnnotationValidation {
        dir: dir.to_path_buf(),
        mapping_rows,
        perimeter_ids: ids.len(),
        loaded_specs,
        rereferenced_rows,
    })
}

#[derive(Args)]
pub struct InspectArgs {
    #[command(subcommand)]
    pub command: InspectCommand,
}

#[derive(Subcommand)]
pub enum InspectCommand {
    /// Export and verify all declared trial figures into a fresh directory.
    Export(export::ExportArgs),

    /// Read-only verification of a completed native trial-export receipt.
    VerifyExport {
        /// Directory containing plot_manifest.json.
        #[arg(long)]
        input: PathBuf,
    },
    /// Generate all inspection plots from analysis output.
    Plot(PlotArgs),

    /// Generate an inspection video with Rust-native overlays.
    #[cfg(feature = "full")]
    Video {
        /// Path to the inspection output directory.
        #[arg(default_value = DEFAULT_INSPECTION_DIR)]
        dir: PathBuf,

        /// Output video path for single-video mode.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output directory for batch mode (`--all-features`).
        #[arg(long)]
        output_dir: Option<PathBuf>,

        /// Video codec (h264, mpeg4, hevc_nvenc, av1_qsv).
        #[arg(long, default_value = DEFAULT_VIDEO_CODEC)]
        codec: String,

        /// Heuristic selector for compatibility mode.
        #[arg(long)]
        heuristic: Option<String>,

        /// Render the feature with this exact manifest label.
        #[arg(long)]
        feature: Option<String>,

        /// Render overview plus every renderable feature into an output directory.
        #[arg(long, default_value_t = false)]
        all_features: bool,

        /// Optional bekiper_config.toml for `[inspect.plot]` video-theme overrides.
        #[arg(long)]
        config: Option<PathBuf>,

        /// Optional ffmpeg encoder preset override.
        #[arg(long)]
        ffmpeg_preset: Option<String>,

        /// Decode this video instead of the manifest's recorded `video_path`.
        #[arg(long)]
        source_video: Option<PathBuf>,

        /// Constant rate factor for CRF encoders such as libx264 (default 23).
        /// Falls back to `BEKIPER_VIDEO_CRF` for CRF codecs only.
        #[arg(long)]
        crf: Option<u8>,

        /// Overview with scored-heuristic outline colours and a mask timeline strip.
        #[arg(long, default_value_t = false)]
        scored: bool,

        /// NORT qualia overview: detection-field overlay and a per-object channel timeline.
        #[arg(long, default_value_t = false)]
        qualia: bool,

        /// Integer frame upscale for `--qualia` (default: enough for 640 px width).
        #[arg(long, requires = "qualia", value_parser = clap::value_parser!(u32).range(1..=8))]
        upscale: Option<u32>,
    },

    /// Display summary information about inspection output.
    Info {
        /// Path to the inspection output directory.
        #[arg(default_value = DEFAULT_INSPECTION_DIR)]
        dir: PathBuf,
    },

    /// Check annotation consistency.
    CheckAnnotation {
        /// Path to the annotation file or project config.
        #[arg(short, long)]
        path: PathBuf,
    },

    /// Overlay source annotations on a video frame (extracts middle frame).
    ///
    /// Draws perimeters, arm-lines, center polygon, and reference point
    /// onto the frame. Useful for checking if an existing annotation
    /// fits an unannotated video.
    #[cfg(feature = "full")]
    Overlay(super::overlay::OverlayArgs),
}

/// Forward visual choices to the Python renderer without overriding its
/// file/environment defaults when a CLI option was not supplied.
#[derive(Args, Debug, Default)]
pub struct PlotArgs {
    #[arg(default_value = DEFAULT_INSPECTION_DIR)]
    pub dir: PathBuf,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(short, long)]
    pub format: Option<String>,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub dpi: Option<u32>,
    #[arg(long)]
    pub profile: Option<String>,
    #[arg(long)]
    pub only: Option<String>,
    #[arg(long)]
    pub skip: Option<String>,
    #[arg(long, conflicts_with = "no_features")]
    pub features_only: bool,
    #[arg(long)]
    pub no_features: bool,
    #[arg(long)]
    pub no_stories: bool,
    #[arg(long, value_parser = ["line", "density", "both"])]
    pub render_mode: Option<String>,
    #[arg(long, value_parser = ["linear", "log"])]
    pub density_scale: Option<String>,
    #[arg(long)]
    pub min_speed: Option<f64>,
    #[arg(long, conflicts_with = "no_velocity_field")]
    pub velocity_field: bool,
    #[arg(long)]
    pub no_velocity_field: bool,
    #[arg(short = 'N', long, value_parser = clap::value_parser!(u32).range(1..))]
    pub small_multiples: Option<u32>,
    /// Colour mode: dark (night mode, the default) or light.
    #[arg(long, value_parser = ["dark", "light"])]
    pub theme: Option<String>,
    /// One heading-quiver arrow per N selected frames.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub quiver_stride: Option<u32>,
    /// Time colormap of single-case quiver figures.
    #[arg(long)]
    pub quiver_single_cmap: Option<String>,
    /// Comma-separated time colormaps of the two quiver classes.
    #[arg(long)]
    pub quiver_binary_cmaps: Option<String>,
    #[arg(long)]
    pub dry_run: bool,
}

pub fn run(args: InspectArgs) -> anyhow::Result<()> {
    match args.command {
        InspectCommand::Export(args) => export::run(args),
        InspectCommand::VerifyExport { input } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&export::verify(&input)?)?
            );
            Ok(())
        }
        InspectCommand::Plot(args) => crate::python::run_inspect_plot(&args),
        #[cfg(feature = "full")]
        InspectCommand::Video {
            dir,
            output,
            output_dir,
            codec,
            heuristic,
            feature,
            all_features,
            config,
            ffmpeg_preset,
            source_video,
            crf,
            scored,
            qualia,
            upscale,
        } => {
            let crf = match crf {
                Some(crf) => Some(crf),
                None => bekiper_ingress::inspection_video::crf_from_env(&codec, |key| {
                    std::env::var(key).ok()
                })?,
            };
            let outputs = bekiper_ingress::inspection_video::render_from_directory(
                &dir,
                &bekiper_ingress::inspection_video::RenderOptions {
                    output,
                    output_dir,
                    codec,
                    heuristic,
                    feature,
                    all_features,
                    config_path: config,
                    ffmpeg_preset,
                    source_video,
                    crf,
                    scored,
                    qualia,
                    upscale,
                },
            )?;
            for path in outputs {
                println!("Saved: {}", path.display());
            }
            Ok(())
        }
        InspectCommand::Info { dir } => crate::python::run_inspect_info(&dir),
        #[cfg(feature = "full")]
        InspectCommand::Overlay(args) => super::overlay::run(args),
        InspectCommand::CheckAnnotation { path } => {
            tracing::info!(?path, "checking annotations");
            let dirs = resolve_annotation_directories(&path)?;
            for dir in dirs {
                let summary = validate_annotation_dir(&dir)?;
                println!(
                    "Annotations OK at {} ({}, {} perimeter ids, {} perimeter specs, {} re-referenced rows)",
                    summary.dir.display(),
                    if summary.mapping_rows == 1 {
                        "1 mapping row".to_string()
                    } else {
                        format!("{} mapping rows", summary.mapping_rows)
                    },
                    summary.perimeter_ids,
                    summary.loaded_specs,
                    summary.rereferenced_rows
                );
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bekiper_cli_inspect_{label}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn resolve_annotation_directories_accepts_directory() {
        let dir = temp_path("dir");
        let resolved = resolve_annotation_directories(&dir).unwrap();
        assert_eq!(resolved, vec![dir.clone()]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resolve_annotation_directories_accepts_metadata_file() {
        let dir = temp_path("metadata");
        let metadata = dir.join("metadata.csv");
        std::fs::write(&metadata, "").unwrap();

        let resolved = resolve_annotation_directories(&metadata).unwrap();
        assert_eq!(resolved, vec![dir.clone()]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resolve_annotation_directories_reads_scan_annotations_from_config() {
        let dir = temp_path("config");
        let config_path = dir.join("bekiper_config.toml");
        std::fs::write(
            &config_path,
            r#"
                [project]
                project_name = "test"
                data_directory = "data"
                output_dir = "output"

                [[scan]]
                video_dir = "videos"
                task_type = "object_recognition"
                fps = 30.0
                meters_per_pixel = 0.001
                annotations = { dir = "annotations/nort" }
            "#,
        )
        .unwrap();

        let resolved = resolve_annotation_directories(&config_path).unwrap();
        assert_eq!(resolved, vec![dir.join("annotations/nort")]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
