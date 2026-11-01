use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::Command;

/// Invoke the `bekiper-inspect` Python CLI as a subprocess.
///
/// This bridges Rust analysis output to Python matplotlib visualization
/// in a single seamless CLI flow. The user never needs to know two
/// languages are involved.
pub fn invoke_bekiper_inspect(args: &[impl AsRef<OsStr>]) -> anyhow::Result<()> {
    let status = Command::new("bekiper-inspect").args(args).status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => {
            anyhow::bail!(
                "bekiper-inspect exited with status {}. \
                 Ensure bekiper-inspect is installed: pip install bekiper-inspect",
                s
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!(
                "bekiper-inspect not found on PATH. \
                 Install it with: pip install bekiper-inspect"
            );
        }
        Err(e) => Err(e.into()),
    }
}

/// Run `bekiper-inspect plot` on an inspection output directory.
pub fn run_inspect_plot(options: &crate::commands::inspect::PlotArgs) -> anyhow::Result<()> {
    invoke_bekiper_inspect(&plot_arguments(options))
}

fn plot_arguments(options: &crate::commands::inspect::PlotArgs) -> Vec<OsString> {
    let mut args = vec![OsString::from("plot"), options.dir.as_os_str().to_owned()];
    for (flag, path) in [("--output", &options.output), ("--config", &options.config)] {
        if let Some(path) = path {
            args.extend([flag.into(), path.as_os_str().to_owned()]);
        }
    }
    for (flag, value) in [
        ("--format", options.format.clone()),
        ("--dpi", options.dpi.map(|v| v.to_string())),
        ("--profile", options.profile.clone()),
        ("--only", options.only.clone()),
        ("--skip", options.skip.clone()),
        ("--render-mode", options.render_mode.clone()),
        ("--density-scale", options.density_scale.clone()),
        ("--min-speed", options.min_speed.map(|v| v.to_string())),
        (
            "--small-multiples",
            options.small_multiples.map(|v| v.to_string()),
        ),
        ("--theme", options.theme.clone()),
        (
            "--quiver-stride",
            options.quiver_stride.map(|v| v.to_string()),
        ),
        ("--quiver-single-cmap", options.quiver_single_cmap.clone()),
        ("--quiver-binary-cmaps", options.quiver_binary_cmaps.clone()),
    ] {
        if let Some(value) = value {
            args.extend([flag.into(), value.into()]);
        }
    }
    for (flag, enabled) in [
        ("--features-only", options.features_only),
        ("--no-features", options.no_features),
        ("--no-stories", options.no_stories),
        ("--velocity-field", options.velocity_field),
        ("--no-velocity-field", options.no_velocity_field),
        ("--dry-run", options.dry_run),
    ] {
        if enabled {
            args.push(flag.into());
        }
    }
    args
}

/// Run `bekiper-inspect info` on an inspection output directory.
pub fn run_inspect_info(inspection_dir: &Path) -> anyhow::Result<()> {
    let dir_str = inspection_dir.display().to_string();
    invoke_bekiper_inspect(&["info", &dir_str])
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        plot: crate::commands::inspect::PlotArgs,
    }

    #[test]
    fn plot_preserves_file_defaults_and_forwards_visual_options() {
        let defaults = Cli::try_parse_from(["test"]).unwrap();
        assert_eq!(
            plot_arguments(&defaults.plot),
            vec![OsString::from("plot"), "inspection".into()]
        );
        let cli = Cli::try_parse_from([
            "test",
            "trial",
            "--config",
            "theme.toml",
            "--format",
            ".png",
            "--dpi",
            "90",
            "--profile",
            "temporal",
            "--only",
            "story_*.png",
            "--skip",
            "*speed*",
            "--features-only",
            "--no-stories",
            "--render-mode",
            "both",
            "--density-scale",
            "log",
            "--min-speed",
            "0.02",
            "--no-velocity-field",
            "-N",
            "3",
            "--theme",
            "light",
            "--quiver-stride",
            "5",
            "--quiver-single-cmap",
            "RdPu",
            "--quiver-binary-cmaps",
            "Reds,Blues",
            "--dry-run",
        ])
        .unwrap();
        let args = plot_arguments(&cli.plot);
        for flag in [
            "--config",
            "--format",
            "--dpi",
            "--profile",
            "--only",
            "--skip",
            "--features-only",
            "--no-stories",
            "--render-mode",
            "--density-scale",
            "--min-speed",
            "--no-velocity-field",
            "--small-multiples",
            "--theme",
            "--quiver-stride",
            "--quiver-single-cmap",
            "--quiver-binary-cmaps",
            "--dry-run",
        ] {
            assert!(args.contains(&OsString::from(flag)), "missing {flag}");
        }
        assert!(Cli::try_parse_from(["test", "--dpi", "0"]).is_err());
        assert!(Cli::try_parse_from(["test", "--features-only", "--no-features"]).is_err());
        assert!(Cli::try_parse_from(["test", "--theme", "sepia"]).is_err());
        assert!(Cli::try_parse_from(["test", "--quiver-stride", "0"]).is_err());
    }
}
