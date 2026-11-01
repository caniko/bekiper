use std::path::{Path, PathBuf};

use polars::prelude::*;

use bekiper_core::error::{BekiperError, Result as CoreResult};
use bekiper_core::video::FrameTiming;

use crate::pipeline::PipelineStep;

/// Join per-frame wall-clock timestamps from a CSV file.
///
/// The CSV is headerless with the ISO 8601 timestamp in the last column
/// (mextractor format: `...,2023-08-19T12:07:03.9757952+02:00`).
/// One row per frame — the row index maps directly to the frame index
/// in the pose dataframe.
///
/// Adds one column:
/// - `elapsed_s`: seconds since the first frame (float64)
pub struct TimestampJoiner {
    pub timestamps_file: PathBuf,
}

impl PipelineStep for TimestampJoiner {
    fn name(&self) -> &str {
        "timestamp_joiner"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let timestamps = match parse_timestamps(self.timestamps_file.as_path()) {
            Ok(ts) => ts,
            Err(e) => {
                tracing::warn!(
                    path = %self.timestamps_file.display(),
                    error = %e,
                    "failed to parse timestamps CSV, skipping"
                );
                return lf;
            }
        };

        let n_frames = lf
            .clone()
            .select([col("*")])
            .collect()
            .map(|df| df.height())
            .unwrap_or(0);

        let n_timestamps = timestamps.len();
        if n_timestamps == 0 {
            tracing::warn!("timestamps CSV is empty, skipping");
            return lf;
        }

        // Truncate or pad to match frame count.
        let ts_vec: Vec<Option<f64>> = if n_timestamps >= n_frames {
            timestamps[..n_frames].iter().map(|t| Some(*t)).collect()
        } else {
            tracing::warn!(
                n_frames,
                n_timestamps,
                "fewer timestamps than frames, padding with null"
            );
            let mut v: Vec<Option<f64>> = timestamps.iter().map(|t| Some(*t)).collect();
            v.resize(n_frames, None);
            v
        };

        let elapsed_series = Series::new("elapsed_s".into(), &ts_vec);

        // Add elapsed_s as a new column by horizontal concatenation.
        // Both the dataframe and the series have the same length (we
        // truncated/padded ts_vec above).
        lf.with_column(elapsed_series.lit())
    }
}

/// Build frame timing from an `elapsed_s` column or other elapsed-seconds source.
///
/// Nulls are ignored until the first timestamp and then treated as the end of
/// the timed prefix. This matches the `TimestampJoiner` behavior, which pads
/// missing tail rows with nulls when the timestamps CSV is shorter than the
/// pose dataframe.
pub fn frame_timing_from_elapsed_seconds<I>(
    elapsed: I,
    fallback_fps: Option<f64>,
) -> Option<FrameTiming>
where
    I: IntoIterator<Item = Option<f64>>,
{
    let mut timestamps = Vec::new();
    let mut saw_timestamp = false;

    for maybe_elapsed in elapsed {
        match maybe_elapsed {
            Some(value) if value.is_finite() => {
                saw_timestamp = true;
                timestamps.push(value);
            }
            Some(_) => continue,
            None if saw_timestamp => break,
            None => continue,
        }
    }

    if timestamps.is_empty() {
        return None;
    }

    if timestamps.windows(2).any(|w| w[1] < w[0]) {
        tracing::warn!("elapsed_s column is not monotonic, ignoring augmented timing");
        return None;
    }

    let mut timing = FrameTiming::from_timestamps(timestamps);
    if timing.fps <= 0.0 {
        timing.fps = fallback_fps.unwrap_or(0.0);
    }
    Some(timing)
}

/// Load mextractor/Bonsai timestamps as elapsed seconds from the first frame.
pub fn load_elapsed_timestamps(path: &Path) -> CoreResult<Vec<f64>> {
    parse_timestamps(path).map_err(BekiperError::Data)
}

/// Parse the mextractor timestamps CSV, returning elapsed seconds from
/// the first frame for each row.
fn parse_timestamps(path: &Path) -> std::result::Result<Vec<f64>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut timestamps_ns: Vec<i64> = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let ts_str = fields
            .last()
            .ok_or_else(|| "empty line".to_string())?
            .trim();

        // Parse ISO 8601 datetime with timezone offset.
        // Format: 2023-08-19T12:07:03.9757952+02:00
        let nanos = parse_iso8601_to_nanos(ts_str)
            .map_err(|e| format!("invalid timestamp '{ts_str}': {e}"))?;
        timestamps_ns.push(nanos);
    }

    if timestamps_ns.is_empty() {
        return Ok(vec![]);
    }

    let t0 = timestamps_ns[0];
    Ok(timestamps_ns
        .iter()
        .map(|t| (*t - t0) as f64 / 1_000_000_000.0)
        .collect())
}

/// Parse an ISO 8601 datetime string to nanoseconds since Unix epoch.
///
/// Handles the mextractor format: `2023-08-19T12:07:03.9757952+02:00`
/// (fractional seconds up to 7 digits, timezone offset).
fn parse_iso8601_to_nanos(s: &str) -> std::result::Result<i64, String> {
    // Split at 'T' → date and time+tz
    let (date_part, time_tz) = s.split_once('T').ok_or("missing 'T' separator")?;

    // Parse date: YYYY-MM-DD
    let date_fields: Vec<&str> = date_part.split('-').collect();
    if date_fields.len() != 3 {
        return Err("invalid date".into());
    }
    let year: i64 = date_fields[0].parse().map_err(|_| "bad year")?;
    let month: u32 = date_fields[1].parse().map_err(|_| "bad month")?;
    let day: u32 = date_fields[2].parse().map_err(|_| "bad day")?;

    // Split time from timezone offset (+ or - after the time portion).
    // The offset is the last 6 chars: +HH:MM or -HH:MM
    let (time_part, tz_offset_secs) = if time_tz.len() >= 6 {
        let tz_sign_pos = time_tz.len() - 6;
        let tz_char = time_tz.as_bytes()[tz_sign_pos];
        if tz_char == b'+' || tz_char == b'-' {
            let time = &time_tz[..tz_sign_pos];
            let tz = &time_tz[tz_sign_pos..];
            let sign: i64 = if tz_char == b'+' { 1 } else { -1 };
            let tz_h: i64 = tz[1..3].parse().map_err(|_| "bad tz hours")?;
            let tz_m: i64 = tz[4..6].parse().map_err(|_| "bad tz minutes")?;
            (time, sign * (tz_h * 3600 + tz_m * 60))
        } else {
            (time_tz, 0i64)
        }
    } else {
        (time_tz, 0i64)
    };

    // Parse time: HH:MM:SS.fractional
    let time_fields: Vec<&str> = time_part.split(':').collect();
    if time_fields.len() != 3 {
        return Err("invalid time".into());
    }
    let hour: u32 = time_fields[0].parse().map_err(|_| "bad hour")?;
    let minute: u32 = time_fields[1].parse().map_err(|_| "bad minute")?;

    let (whole_secs, frac_nanos) = if let Some((sec_str, frac_str)) = time_fields[2].split_once('.')
    {
        let secs: u32 = sec_str.parse().map_err(|_| "bad seconds")?;
        // Pad/truncate fractional part to 9 digits (nanoseconds).
        let padded = format!("{:0<9}", frac_str);
        let nanos: i64 = padded[..9].parse().map_err(|_| "bad fractional seconds")?;
        (secs, nanos)
    } else {
        let secs: u32 = time_fields[2].parse().map_err(|_| "bad seconds")?;
        (secs, 0i64)
    };

    // Days from Unix epoch (1970-01-01) to the given date.
    // Simple calculation, accurate for dates 1970–2099.
    let days = days_from_epoch(year, month, day);
    let total_secs =
        days * 86400 + hour as i64 * 3600 + minute as i64 * 60 + whole_secs as i64 - tz_offset_secs;

    Ok(total_secs * 1_000_000_000 + frac_nanos)
}

/// Days from 1970-01-01 to the given date. Handles leap years.
fn days_from_epoch(year: i64, month: u32, day: u32) -> i64 {
    let mut y = year;
    let mut m = month as i64;
    // Adjust for months Jan/Feb (treat as month 13/14 of previous year)
    if m <= 2 {
        y -= 1;
        m += 12;
    }
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iso8601_basic() {
        // 2023-08-19T12:07:03.9757952+02:00 → UTC = 10:07:03.9757952
        let nanos = super::parse_iso8601_to_nanos("2023-08-19T12:07:03.9757952+02:00").unwrap();
        // Sanity: should be somewhere in 2023
        assert!(nanos > 1_690_000_000_000_000_000); // > 2023-07-22
        assert!(nanos < 1_700_000_000_000_000_000); // < 2023-11-14
    }

    #[test]
    fn parse_mextractor_timestamps() {
        let dir = std::env::temp_dir().join("bekiper_test_timestamps");
        std::fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("timestamps.csv");
        std::fs::write(
            &csv_path,
            "False,16,1920,1080,U8,3,1920,1080,5760,865960016,0,0,0,1920,1080,False,2023-08-19T12:07:03.9757952+02:00\n\
             False,16,1920,1080,U8,3,1920,1080,5760,911589456,0,0,0,1920,1080,False,2023-08-19T12:07:03.9976064+02:00\n\
             False,16,1920,1080,U8,3,1920,1080,5760,917831760,0,0,0,1920,1080,False,2023-08-19T12:07:04.0335488+02:00\n",
        ).unwrap();

        let elapsed = parse_timestamps(&csv_path).unwrap();
        assert_eq!(elapsed.len(), 3);
        assert!((elapsed[0] - 0.0).abs() < 1e-9);
        assert!((elapsed[1] - 0.0218112).abs() < 0.001); // ~22ms
        assert!((elapsed[2] - 0.0577536).abs() < 0.001); // ~58ms
    }

    #[test]
    fn frame_timing_from_elapsed_seconds_uses_average_fps() {
        let timing =
            frame_timing_from_elapsed_seconds(vec![Some(0.0), Some(0.10), Some(0.24)], None)
                .unwrap();

        assert!((timing.fps - (2.0 / 0.24)).abs() < 1e-9);
        assert!(timing.is_vfr);
        assert_eq!(timing.timestamps, vec![0.0, 0.10, 0.24]);
    }

    #[test]
    fn frame_timing_from_elapsed_seconds_uses_fallback_for_single_timestamp() {
        let timing = frame_timing_from_elapsed_seconds(vec![Some(1.5)], Some(20.0)).unwrap();

        assert!((timing.fps - 20.0).abs() < 1e-9);
        assert!(!timing.is_vfr);
        assert_eq!(timing.timestamps, vec![1.5]);
    }

    #[test]
    fn frame_timing_from_elapsed_seconds_stops_at_padded_nulls() {
        let timing = frame_timing_from_elapsed_seconds(
            vec![Some(0.0), Some(0.05), None, Some(0.10)],
            Some(30.0),
        )
        .unwrap();

        assert_eq!(timing.timestamps, vec![0.0, 0.05]);
        assert!((timing.fps - 20.0).abs() < 1e-9);
    }

    #[test]
    fn frame_timing_from_elapsed_seconds_rejects_non_monotonic_input() {
        let timing =
            frame_timing_from_elapsed_seconds(vec![Some(0.0), Some(0.10), Some(0.09)], Some(30.0));

        assert!(timing.is_none());
    }

    fn write_timestamps(name: &str, content: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bekiper_ts_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("timestamps.csv");
        std::fs::write(&path, content).unwrap();
        path
    }

    fn joined(path: PathBuf, frames: usize) -> DataFrame {
        let joiner = TimestampJoiner {
            timestamps_file: path,
        };
        assert_eq!(joiner.name(), "timestamp_joiner");
        let df = df! { "frame" => (0..frames as i64).collect::<Vec<_>>() }.unwrap();
        joiner.apply(df.lazy()).collect().unwrap()
    }

    const VFR: &str = "x,2023-08-19T12:07:03.0+02:00\n\n\
                       x,2023-08-19T12:07:03.04+02:00\n\
                       x,2023-08-19T12:07:03.1+02:00\n";

    #[test]
    fn joiner_truncates_extra_timestamps_and_keeps_vfr_spacing() {
        let out = joined(write_timestamps("truncate", VFR), 2);
        let elapsed: Vec<_> = out
            .column("elapsed_s")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(elapsed.len(), 2);
        assert_eq!(elapsed[0], Some(0.0));
        assert!((elapsed[1].unwrap() - 0.04).abs() < 1e-9);
    }

    #[test]
    fn joiner_pads_missing_timestamps_with_null() {
        let out = joined(write_timestamps("pad", VFR), 5);
        let elapsed: Vec<_> = out
            .column("elapsed_s")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect();
        assert!((elapsed[2].unwrap() - 0.1).abs() < 1e-9);
        assert_eq!(elapsed[3..], [None, None]);
        let timing = frame_timing_from_elapsed_seconds(elapsed, Some(30.0)).unwrap();
        assert_eq!(timing.timestamps.len(), 3);
        assert!(timing.is_vfr);
    }

    #[test]
    fn joiner_skips_unreadable_invalid_or_empty_files() {
        for path in [
            PathBuf::from("/nonexistent/bekiper/timestamps.csv"),
            write_timestamps("invalid", "x,not-a-timestamp\n"),
            write_timestamps("blank", "\n  \n"),
        ] {
            let out = joined(path, 3);
            assert!(out.column("elapsed_s").is_err());
            assert_eq!(out.height(), 3);
        }
        let error = load_elapsed_timestamps(&write_timestamps("load", "x,bad\n")).unwrap_err();
        assert!(
            error.to_string().contains("invalid timestamp 'bad'"),
            "{error}"
        );
        assert_eq!(
            load_elapsed_timestamps(&write_timestamps("load_ok", VFR))
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn elapsed_seconds_skip_non_finite_and_leading_nulls() {
        let timing = frame_timing_from_elapsed_seconds(
            vec![
                None,
                Some(f64::NAN),
                Some(0.0),
                Some(f64::INFINITY),
                Some(0.5),
            ],
            None,
        )
        .unwrap();
        assert_eq!(timing.timestamps, vec![0.0, 0.5]);
        assert!(frame_timing_from_elapsed_seconds(vec![None, Some(f64::NAN)], None).is_none());
        // A single timestamp without a fallback leaves the rate at zero.
        let single = frame_timing_from_elapsed_seconds(vec![Some(2.0)], None).unwrap();
        assert_eq!(single.fps, 0.0);
    }

    #[test]
    fn iso8601_accepts_utc_negative_offsets_and_whole_seconds() {
        let utc = parse_iso8601_to_nanos("1970-01-02T00:00:01").unwrap();
        assert_eq!(utc, (86_400 + 1) * 1_000_000_000);
        let short = parse_iso8601_to_nanos("1970-01-01T0:0:1").unwrap();
        assert_eq!(short, 1_000_000_000);
        // 10 characters ending without an offset sign stay local time.
        assert_eq!(
            parse_iso8601_to_nanos("1970-01-01T00:00:00.5").unwrap(),
            500_000_000
        );
        let west = parse_iso8601_to_nanos("1970-01-01T00:00:00-01:30").unwrap();
        assert_eq!(west, 5_400 * 1_000_000_000);
        // Leap day in March-based day counting.
        assert_eq!(
            parse_iso8601_to_nanos("2024-03-01T00:00:00+00:00").unwrap()
                - parse_iso8601_to_nanos("2024-02-28T00:00:00+00:00").unwrap(),
            2 * 86_400 * 1_000_000_000
        );
    }

    #[test]
    fn iso8601_rejects_each_malformed_field() {
        for (input, expected) in [
            ("2023-08-19 12:00:00", "missing 'T' separator"),
            ("2023-08T12:00:00", "invalid date"),
            ("yyyy-08-19T12:00:00", "bad year"),
            ("2023-mm-19T12:00:00", "bad month"),
            ("2023-08-ddT12:00:00", "bad day"),
            ("2023-08-19T12:00:00+xx:00", "bad tz hours"),
            ("2023-08-19T12:00:00+02:xx", "bad tz minutes"),
            ("2023-08-19T12:00", "invalid time"),
            ("2023-08-19Thh:00:00", "bad hour"),
            ("2023-08-19T12:mm:00", "bad minute"),
            ("2023-08-19T12:00:ss.5", "bad seconds"),
            ("2023-08-19T12:00:00.5x", "bad fractional seconds"),
            ("2023-08-19T12:00:ss", "bad seconds"),
        ] {
            assert_eq!(
                parse_iso8601_to_nanos(input).unwrap_err(),
                expected,
                "{input}"
            );
        }
    }
}
