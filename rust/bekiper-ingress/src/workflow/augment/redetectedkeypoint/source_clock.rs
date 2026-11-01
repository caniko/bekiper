/// Bind a source clock before cropping/trimming, using original frame keys.
/// An explicitly recorded clock cannot silently degrade into CFR timing.
pub(crate) fn attach_source_clock(
    raw: polars::prelude::LazyFrame,
    experiment: &ExperimentConfig,
) -> Result<polars::prelude::LazyFrame> {
    use polars::prelude::*;
    let timestamps = if let Some(path) = &experiment.timestamps_file {
        Some(bekiper_pose::steps::timestamp::load_elapsed_timestamps(
            path,
        )?)
    } else if raw
        .clone()
        .collect_schema()?
        .iter_names()
        .any(|name| name == "elapsed_s")
    {
        return Ok(raw);
    } else {
        experiment
            .frame_timing
            .as_ref()
            .filter(|t| !t.timestamps.is_empty())
            .map(|t| t.timestamps.clone())
    };
    let Some(timestamps) = timestamps else {
        return Ok(raw);
    };
    if timestamps.is_empty()
        || timestamps.iter().any(|t| !t.is_finite())
        || timestamps.windows(2).any(|t| t[0] >= t[1])
    {
        return Err(BekiperError::Data(
            "source clock must be finite and strictly increasing".into(),
        ));
    }
    let mut df = raw.collect()?;
    let times = crate::analysis::frame_keys::source_frame_indices(&df)?
        .into_iter()
        .map(|key| {
            usize::try_from(key)
                .ok()
                .and_then(|i| timestamps.get(i))
                .copied()
                .ok_or_else(|| {
                    BekiperError::Data(format!("source clock has no timestamp for frame {key}"))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    df.with_column(Column::new("elapsed_s".into(), times))?;
    Ok(df.lazy())
}
