use bekiper_core::error::{BekiperError, Result};
use polars::prelude::*;

/// Source-video keys in evaluated-row order. Old tables without explicit keys
/// use their row positions; cropped/sparse tables retain the recorded frame IDs.
pub(crate) fn source_frame_indices(df: &DataFrame) -> Result<Vec<u64>> {
    let Ok(column) = df.column("frame") else {
        return Ok((0..df.height() as u64).collect());
    };
    if !column.dtype().is_integer() {
        return Err(BekiperError::Data(
            "Inspection frame keys must be integers".into(),
        ));
    }
    let keys = column.cast(&DataType::UInt64)?;
    if keys.null_count() != 0 {
        return Err(BekiperError::Data(
            "Inspection frame keys must be nonnegative and non-null".into(),
        ));
    }
    let keys = keys.u64()?.into_no_null_iter().collect::<Vec<_>>();
    if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(BekiperError::Data(
            "Inspection frame keys must be unique and increasing".into(),
        ));
    }
    Ok(keys)
}

#[cfg(test)]
mod frame_key_tests {
    use super::*;

    #[test]
    fn frame_keys_preserve_crops_gaps_and_legacy_row_positions() {
        let df = df! { "frame" => [76u64, 78, 79] }.unwrap();
        assert_eq!(source_frame_indices(&df).unwrap(), [76, 78, 79]);
        assert_eq!(
            source_frame_indices(&df.drop("frame").unwrap()).unwrap(),
            [0, 1, 2]
        );
    }

    #[test]
    fn ambiguous_or_invalid_video_frame_keys_are_rejected() {
        for keys in [
            Series::new("frame".into(), [Some(0i64), None]),
            Series::new("frame".into(), [-1i64, 0]),
            Series::new("frame".into(), [1u64, 1]),
            Series::new("frame".into(), [2u64, 1]),
            Series::new("frame".into(), [0.5f64, 1.5]),
        ] {
            let df = DataFrame::new(keys.len(), vec![keys.into_column()]).unwrap();
            assert!(source_frame_indices(&df).is_err(), "{df:?}");
        }
    }
}
