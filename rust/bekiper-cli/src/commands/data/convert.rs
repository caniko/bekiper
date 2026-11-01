use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Args;

use bekiper_ingress::pose_import::{
    PoseImportOptions, collect_convert_inputs, import_pose_file_to_parquet,
};

/// Batch-convert DLC .h5 pose files to normalized parquet.
///
/// The command prefers native Rust readers and only falls back to
/// Python/pandas for explicitly requested HDF5 escape-hatch cases.
#[derive(Args)]
pub struct ConvertArgs {
    /// Pose file or directory to scan recursively.
    pub path: PathBuf,

    /// Overwrite existing .parquet files.
    #[arg(long, default_value_t = false)]
    pub force: bool,

    /// Permit a temporary Python fallback for HDF5 inputs when Rust parsing fails.
    #[arg(long, default_value_t = false)]
    pub python_fallback: bool,

    /// Invert Y coordinates in pixel space before writing parquet.
    #[arg(long, default_value_t = false)]
    pub invert_y: bool,

    /// Source frame height in pixels, required when using `--invert-y`.
    #[arg(long)]
    pub frame_height_px: Option<u32>,
}

pub fn run(args: ConvertArgs) -> anyhow::Result<()> {
    let inputs = collect_convert_inputs(&args.path)?;
    if inputs.is_empty() {
        tracing::info!(path = %args.path.display(), "no supported pose files found");
        return Ok(());
    }
    if args.invert_y && args.frame_height_px.is_none() {
        anyhow::bail!("--invert-y requires --frame-height-px");
    }

    tracing::info!(count = inputs.len(), "found pose files to convert");

    let python = resolve_python();
    let mut converted = 0usize;
    let mut skipped = 0usize;

    for input_path in &inputs {
        let parquet_path = input_path.with_extension("parquet");
        if parquet_path.exists() && !args.force {
            skipped += 1;
            continue;
        }

        tracing::info!(file = %input_path.display(), "converting");

        let report = match import_pose_file_to_parquet(
            input_path,
            &parquet_path,
            PoseImportOptions {
                invert_y: args.invert_y,
                frame_height_px: args.frame_height_px,
            },
        ) {
            Ok(report) => report,
            Err(err) if args.python_fallback && is_hdf5(input_path) && !args.invert_y => {
                tracing::warn!(
                    file = %input_path.display(),
                    error = %err,
                    "rust conversion failed, using explicit python fallback"
                );
                convert_h5_with_python(&python, input_path, &parquet_path)?;
                converted += 1;
                continue;
            }
            Err(err) if args.python_fallback && is_hdf5(input_path) && args.invert_y => {
                anyhow::bail!(
                    "rust conversion failed for {}: {err}. Python fallback does not support --invert-y",
                    input_path.display()
                );
            }
            Err(err) if is_hdf5(input_path) => {
                anyhow::bail!(
                    "rust conversion failed for {}: {err}. Re-run with --python-fallback for temporary HDF5 fallback support",
                    input_path.display()
                );
            }
            Err(err) => anyhow::bail!("rust conversion failed for {}: {err}", input_path.display()),
        };

        tracing::info!(
            file = %input_path.display(),
            output = %parquet_path.display(),
            format = ?report.format,
            backend = ?report.backend,
            labels = ?report.labels,
            "converted pose file"
        );

        if !parquet_path.exists() {
            anyhow::bail!(
                "conversion reported success but did not produce {}",
                parquet_path.display()
            );
        }

        converted += 1;
    }

    tracing::info!(converted, skipped, "conversion complete");
    Ok(())
}

fn convert_h5_with_python(
    python: &Path,
    input_path: &Path,
    output_path: &Path,
) -> anyhow::Result<()> {
    let script = format!(
        "import pandas as pd; \
         df = pd.read_hdf(r'{}'); \
         df.to_parquet(r'{}')",
        input_path.display(),
        output_path.display(),
    );

    let status = Command::new(python)
        .args(["-c", &script])
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to run python ({}): {e}. \
                 Ensure pandas and pytables/tables are installed.",
                python.display()
            )
        })?;

    if !status.success() {
        anyhow::bail!(
            "python conversion failed for {} (exit {})",
            input_path.display(),
            status
        );
    }

    Ok(())
}

/// Resolve the Python interpreter.
///
/// Uses `python3` from PATH — in the nix devShell this is the nix-provided
/// Python with `bekiper-cli` and `bekiper-inspect` available as nix packages.
fn resolve_python() -> PathBuf {
    PathBuf::from("python3")
}

fn is_hdf5(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("h5") || ext.eq_ignore_ascii_case("hdf5"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_hdf5_matches_h5_and_hdf5() {
        assert!(is_hdf5(Path::new("a.h5")));
        assert!(is_hdf5(Path::new("a.HDF5")));
        assert!(!is_hdf5(Path::new("a.csv")));
    }
}
