#[test]
fn ymaze_valid_exceeds_raw_keeps_row() {
    let root = unique_pipeline_dir("ymaze_valid_exceeds_raw_keeps_row");
    let result_path = root.join("data").join("trial").join("result.toml");
    write_ymaze_invariant_result(
        &result_path,
        "ymaze_trial_invalid",
        "ymaze__proximal25",
        &["A", "B", "C", "A"],
        &[("A", 1), ("B", 1), ("C", 0)],
        1,
        1,
        0,
        50.0,
    );

    let trial = bekiper_ingress::aggregate::TrialResult::load(&result_path).unwrap();
    let logs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_writer({
            let logs = logs.clone();
            move || SharedLogWriter(logs.clone())
        })
        .finish();

    let sheets = tracing::subscriber::with_default(subscriber, || {
        bekiper_ingress::aggregate::build_sheet_dataframes(vec![trial], &Default::default())
            .unwrap()
    });
    let (_, df, _) = sheets
        .iter()
        .find(|(name, _, _)| name == "ymaze__proximal25")
        .expect("ymaze proximal25 sheet");

    assert_eq!(df.height(), 1);

    let log_output = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
    assert!(
        !log_output.contains("dropping ymaze aggregate row"),
        "Y-maze rows must no longer be dropped when valid_total exceeds raw_total"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[derive(Clone)]
struct SharedLogWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for SharedLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
