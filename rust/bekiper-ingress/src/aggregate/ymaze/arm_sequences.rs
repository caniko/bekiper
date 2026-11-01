struct ArmSequenceCounts {
    a: u32,
    b: u32,
    c: u32,
}

fn arm_sequence_counts(value: Option<&Value>) -> Option<ArmSequenceCounts> {
    let Value::Array(values) = value? else {
        return None;
    };
    let mut counts = ArmSequenceCounts { a: 0, b: 0, c: 0 };
    for entry in values {
        let Some(label) = entry.as_str() else {
            continue;
        };
        match label {
            "A" => counts.a += 1,
            "B" => counts.b += 1,
            "C" => counts.c += 1,
            _ => continue,
        }
    }
    Some(counts)
}

fn sequence_cell(value: Option<&Value>) -> CellValue {
    let Some(Value::Array(values)) = value else {
        return CellValue::Null;
    };
    let labels: Vec<&str> = values.iter().filter_map(Value::as_str).collect();
    if labels.is_empty() {
        CellValue::Null
    } else {
        CellValue::String(labels.join(" "))
    }
}

fn sum_distribution(distribution: &Value, keys: &[&str]) -> CellValue {
    let mut sum = 0i64;
    let mut seen = false;
    for key in keys {
        if let Some(value) = distribution.get(key).and_then(Value::as_i64) {
            sum = sum.saturating_add(value);
            seen = true;
        } else if distribution.get(key).is_some_and(Value::is_u64) {
            // Counts beyond i64 saturate instead of wrapping negative.
            sum = i64::MAX;
            seen = true;
        }
    }
    if seen {
        CellValue::Int(sum)
    } else {
        CellValue::Null
    }
}
