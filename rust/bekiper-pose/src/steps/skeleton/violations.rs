impl SkeletonViolationEvaluation {
    fn new(label_order: Vec<String>, flags: Vec<Vec<bool>>) -> Self {
        let label_to_index = label_order
            .iter()
            .enumerate()
            .map(|(idx, label)| (label.clone(), idx))
            .collect();

        let row_count = flags.first().map(Vec::len).unwrap_or(0);
        let mut counts = vec![0u32; row_count];
        for row_idx in 0..row_count {
            counts[row_idx] = flags
                .iter()
                .filter(|label_flags| label_flags[row_idx])
                .count() as u32;
        }

        Self {
            label_order,
            label_to_index,
            flags,
            counts,
        }
    }

    pub fn count(&self, row_index: usize) -> u32 {
        self.counts[row_index]
    }

    pub fn is_violated(&self, row_index: usize, label: &str) -> bool {
        self.label_to_index
            .get(label)
            .is_some_and(|idx| self.flags[*idx][row_index])
    }

    pub fn clear_label(&mut self, row_index: usize, label: &str) {
        if let Some(idx) = self.label_to_index.get(label).copied()
            && self.flags[idx][row_index]
        {
            self.flags[idx][row_index] = false;
            self.counts[row_index] = self.counts[row_index].saturating_sub(1);
        }
    }

    pub fn label_flags(&self, label: &str) -> Option<&[bool]> {
        self.label_to_index
            .get(label)
            .map(|idx| self.flags[*idx].as_slice())
    }

    pub fn flagged_rows(&self, df: &DataFrame) -> Result<Vec<FrameSkeletonViolations>> {
        let frames = df
            .column("frame")?
            .i64()
            .map_err(|_| BekiperError::Data("frame column must be Int64".into()))?;

        let mut flagged = Vec::new();
        for row_index in 0..df.height() {
            let labels: Vec<String> = self
                .label_order
                .iter()
                .enumerate()
                .filter(|(label_idx, _)| self.flags[*label_idx][row_index])
                .map(|(_, label)| label.clone())
                .collect();

            if labels.is_empty() {
                continue;
            }

            let frame = frames.get(row_index).ok_or_else(|| {
                BekiperError::Data(format!("frame column has null at row {row_index}"))
            })?;

            flagged.push(FrameSkeletonViolations {
                row_index,
                frame,
                labels,
            });
        }

        Ok(flagged)
    }
}

#[derive(Debug, Clone)]
pub struct FittedSkeletonRepair {
    label_order: Vec<String>,
    label_to_index: HashMap<String, usize>,
    node_degrees: Vec<usize>,
    baselines: Vec<FittedEdgeBaseline>,
    relative_tolerance: f64,
    mad_multiplier: f64,
}
