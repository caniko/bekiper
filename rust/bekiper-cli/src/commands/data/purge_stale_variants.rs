use std::collections::HashSet;
use std::path::{Path, PathBuf};

use clap::Args;

use bekiper_core::identity::DEFAULT_CONFIG_FILENAME;
use bekiper_core::types::TaskType;
use bekiper_ingress::aggregate::TrialResult;
use bekiper_ingress::config::IngressConfig;
use bekiper_ingress::stage::Stage;

/// Delete per-trial output directories that pre-date a variant rollout.
///
/// After variants were added to a task in `bekiper_config.toml`, the old
/// non-suffixed `output/data/<base>/result.toml` directories are stale —
/// they were produced by a previous single-output run and would land in the
/// legacy `ymaze` / `oft` / `nort_*` aggregate sheets next to the new
/// variant sheets, mixing schemas. This command removes them.
///
/// A directory is considered stale when its `result.toml` exists, its
/// declared `task_type` is one for which the config now declares at least
/// one variant, and its directory name does **not** contain the `__`
/// variant separator. Newly-fanned-out variant directories (e.g.
/// `..._video_14__proximal25`) are kept untouched.
#[derive(Args)]
pub struct PurgeStaleVariantsArgs {
    /// Path to the project configuration file.
    #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
    config: PathBuf,

    /// Print what would be removed without deleting anything.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

pub fn run(args: PurgeStaleVariantsArgs) -> anyhow::Result<()> {
    let mut config = IngressConfig::from_file(&args.config)?;
    let base_dir = args.config.parent().unwrap_or(Path::new("."));
    config.expand_for_validation(base_dir, Stage::all())?;

    let variant_task_types = collect_variant_task_types(&config);
    if variant_task_types.is_empty() {
        println!("No tasks declare variants; nothing to do.");
        return Ok(());
    }

    let run_root = base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(Path::new("output")),
    );
    let data_root = run_root.join("data");
    if !data_root.exists() {
        println!("{} does not exist; nothing to purge.", data_root.display());
        return Ok(());
    }

    let mut stale: Vec<PathBuf> = Vec::new();
    collect_stale_dirs(&data_root, &variant_task_types, &mut stale, 0, 2)?;
    stale.sort();

    if stale.is_empty() {
        println!("No stale non-variant trial outputs found.");
        return Ok(());
    }

    for path in &stale {
        println!("{}", path.display());
    }
    if args.dry_run {
        println!("Would remove {} directory(ies).", stale.len());
        return Ok(());
    }
    for path in &stale {
        std::fs::remove_dir_all(path)?;
    }
    println!("Removed {} directory(ies).", stale.len());
    Ok(())
}

fn collect_variant_task_types(config: &IngressConfig) -> HashSet<TaskType> {
    let mut task_types = HashSet::new();
    for scan in &config.scan {
        if !scan.variants.is_empty() {
            task_types.insert(scan.task_type);
        }
    }
    for profile in &config.ingress_profile {
        for task in &profile.tasks {
            if !task.variants.is_empty() {
                task_types.insert(task.task_type);
            }
        }
    }
    task_types
}

fn collect_stale_dirs(
    dir: &Path,
    variant_task_types: &HashSet<TaskType>,
    out: &mut Vec<PathBuf>,
    depth: usize,
    max_depth: usize,
) -> anyhow::Result<()> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::warn!(path = %dir.display(), error = %e, "skipping unreadable directory");
            return Ok(());
        }
    };
    for entry in read_dir {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let result_path = path.join("result.toml");
        if result_path.exists() {
            if is_stale(&path, &result_path, variant_task_types)? {
                out.push(path);
            }
        } else if depth < max_depth {
            collect_stale_dirs(&path, variant_task_types, out, depth + 1, max_depth)?;
        }
    }
    Ok(())
}

fn is_stale(
    trial_dir: &Path,
    result_path: &Path,
    variant_task_types: &HashSet<TaskType>,
) -> anyhow::Result<bool> {
    let name = match trial_dir.file_name().and_then(|s| s.to_str()) {
        Some(name) => name,
        None => return Ok(false),
    };
    if name.contains("__") {
        // Already a variant-suffixed directory.
        return Ok(false);
    }
    let trial = match TrialResult::load(result_path) {
        Ok(trial) => trial,
        Err(err) => {
            tracing::warn!(path = %result_path.display(), error = %err, "skipping unparseable result.toml");
            return Ok(false);
        }
    };
    Ok(variant_task_types.contains(&trial.task_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    fn project_config(dir: &Path) -> PathBuf {
        let config_path = dir.join("bekiper_config.toml");
        write(
            &config_path,
            r#"
            [project]
            project_name = "purge_stale"
            data_directory = "."
            output_dir = "output"

            [[scan]]
            video_dir = "videos"
            task_type = "y_maze"
            meters_per_pixel = { method = "arm_length", meters = 0.35 }

            [[scan.variants]]
            name = "proximal65"
            [scan.variants.ymaze]
            proximal_threshold = 0.65
            "#,
        );
        std::fs::create_dir_all(dir.join("videos")).unwrap();
        config_path
    }

    fn write_result(path: &Path, task_type: &str) {
        write(
            path,
            &format!(
                r#"
                experiment = "doesnt_matter"
                fps = 30.0
                meters_per_pixel = 0.001
                task_type = "{task_type}"

                [detailed]
                "#
            ),
        );
    }

    #[test]
    fn dry_run_lists_but_keeps_legacy_ymaze_dir() {
        let dir = std::env::temp_dir().join(format!(
            "bekiper_purge_stale_variants_dry_run_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = project_config(&dir);

        let stale = dir.join("output/data/legacy_ymaze_video_1");
        write_result(&stale.join("result.toml"), "y_maze");
        let kept_variant = dir.join("output/data/ymaze_video_1__proximal65");
        write_result(&kept_variant.join("result.toml"), "y_maze");

        run(PurgeStaleVariantsArgs {
            config,
            dry_run: true,
        })
        .unwrap();

        assert!(stale.exists(), "dry-run must not delete");
        assert!(kept_variant.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deletes_legacy_ymaze_dir_keeps_variants_and_other_tasks() {
        let dir = std::env::temp_dir().join(format!(
            "bekiper_purge_stale_variants_real_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = project_config(&dir);

        let stale = dir.join("output/data/legacy_ymaze_video_1");
        write_result(&stale.join("result.toml"), "y_maze");
        let kept_variant = dir.join("output/data/ymaze_video_1__proximal65");
        write_result(&kept_variant.join("result.toml"), "y_maze");
        let kept_oft = dir.join("output/data/oft_video_1");
        write_result(&kept_oft.join("result.toml"), "open_field");

        run(PurgeStaleVariantsArgs {
            config,
            dry_run: false,
        })
        .unwrap();

        assert!(!stale.exists(), "legacy y_maze dir must be removed");
        assert!(kept_variant.exists(), "variant-suffixed dir must survive");
        assert!(
            kept_oft.exists(),
            "task without variants must not be touched"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
