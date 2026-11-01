impl PipelineStep for SkeletonRepairStep {
    fn name(&self) -> &str {
        "skeleton_repair"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let df = lf.collect().expect("skeleton_repair: collect failed");
        let fitted = self.fit(&df).expect("skeleton_repair: fit failed");
        let evaluation = fitted
            .evaluate(&df)
            .expect("skeleton_repair: evaluation failed");
        fitted
            .apply(df, &evaluation, &BTreeSet::new())
            .expect("skeleton_repair: apply failed")
            .lazy()
    }
}
