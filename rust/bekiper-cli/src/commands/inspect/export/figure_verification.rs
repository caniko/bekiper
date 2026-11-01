fn planned_stems(text: &str) -> BTreeSet<String> {
    text.lines()
        .filter(|line| {
            ["  story_", "  feature_", "  summary"]
                .iter()
                .any(|prefix| line.starts_with(prefix))
        })
        .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
        .collect()
}

fn verify_figures(
    expected: &BTreeSet<String>,
    stdout: &str,
    output: &Path,
    dpi: u64,
) -> Result<Vec<PathBuf>> {
    let output = output.canonicalize()?;
    let mut paths = BTreeSet::new();
    for line in stdout
        .lines()
        .filter_map(|line| line.strip_prefix("Saved: "))
    {
        let path = Path::new(line);
        ensure!(
            !fs::symlink_metadata(path)?.file_type().is_symlink(),
            "symlink figure output"
        );
        let path = path.canonicalize()?;
        ensure!(
            path.parent() == Some(output.as_path()) && fs::metadata(&path)?.len() > 0,
            "missing, empty or escaped figure output: {}",
            path.display()
        );
        ensure!(
            path.extension().is_some_and(|e| e == "png"),
            "trial export requires PNG output"
        );
        bekiper_post_analysis::verification::verify_png(&path, dpi)?;
        ensure!(paths.insert(path), "duplicate Saved figure output");
    }
    let actual: BTreeSet<_> = paths
        .iter()
        .filter_map(|p| p.file_stem().and_then(|n| n.to_str()).map(str::to_owned))
        .collect();
    ensure!(
        !expected.is_empty() && &actual == expected,
        "declared plots were not rendered exactly: expected {expected:?}, got {actual:?}"
    );
    Ok(paths.into_iter().collect())
}
