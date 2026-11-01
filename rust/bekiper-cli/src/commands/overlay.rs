use std::path::PathBuf;

use clap::Args;

use bekiper_ingress::workflow::overlay_annotation;

/// Default output directory for `bekiper inspect overlay`. Sits under the top-level
/// `output/` scaffold created by `bekiper init`. Users can override with
/// `-o/--output-dir` if their project layout differs.
const DEFAULT_OVERLAY_OUTPUT_DIR: &str = "output/annotation-inspects";

#[derive(Args)]
pub struct OverlayArgs {
    /// Video file(s) to extract frames from.
    videos: Vec<PathBuf>,

    /// Annotation directory (containing metadata.csv + plugin_files/).
    #[arg(short, long)]
    annotations: PathBuf,

    /// Output directory. Defaults to output/annotation-inspects/.
    #[arg(short, long, default_value = DEFAULT_OVERLAY_OUTPUT_DIR)]
    output_dir: PathBuf,
}

pub fn run(args: OverlayArgs) -> anyhow::Result<()> {
    std::fs::create_dir_all(&args.output_dir)?;

    for video in &args.videos {
        let stem = video.file_stem().unwrap_or_default().to_string_lossy();
        let output = args.output_dir.join(format!("{stem}_overlay.png"));

        match overlay_annotation(video, &args.annotations, &output) {
            Ok(()) => println!("{stem}: saved {}", output.display()),
            Err(e) => eprintln!("{stem}: FAILED ({e})"),
        }
    }

    Ok(())
}
