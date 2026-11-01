fn collect_result_paths(dir: &Path, depth: usize, max_depth: usize) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    collect_result_paths_recursive(dir, &mut paths, depth, max_depth);
    paths
}

/// Recursively collect `result.toml` files up to *max_depth* levels.
fn collect_trial_results(
    dir: &Path,
    trials: &mut Vec<TrialResult>,
    depth: usize,
    max_depth: usize,
) {
    visit_result_paths(
        dir,
        depth,
        max_depth,
        &mut |result_path| match TrialResult::load(result_path) {
            Ok(trial) => trials.push(trial),
            Err(error) => tracing::warn!(
                path = %result_path.display(), error = %error,
                "skipping unreadable result.toml"
            ),
        },
    );
}

fn visit_result_paths(dir: &Path, depth: usize, max_depth: usize, visit: &mut impl FnMut(&Path)) {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::warn!(path = %dir.display(), error = %e, "skipping unreadable directory");
            return;
        }
    };
    for entry in read_dir {
        let Ok(entry) =
            entry.inspect_err(|e| tracing::warn!(error = %e, "skipping unreadable entry"))
        else {
            continue; // coverage: unreachable — readdir entry I/O errors cannot be provoked portably
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let result_path = path.join("result.toml");
        if result_path.exists() {
            visit(&result_path);
        } else if depth < max_depth {
            visit_result_paths(&path, depth + 1, max_depth, visit);
        }
    }
}

fn collect_result_paths_recursive(
    dir: &Path,
    result_paths: &mut Vec<std::path::PathBuf>,
    depth: usize,
    max_depth: usize,
) {
    visit_result_paths(dir, depth, max_depth, &mut |path| {
        result_paths.push(path.to_path_buf())
    });
}
