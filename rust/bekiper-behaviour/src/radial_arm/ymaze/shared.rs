// Per-recording zone geometry shared by Y-maze variants that differ only in the
// proximal fraction `p` (design 4.3): polygon membership is `p`-independent and
// the cross-line / radial predicates are monotone in `p`, so each head/tail
// point carries one critical `p*` and every variant reads `p <= p*` outside the
// float band, falling back to the exact scalar mirror of the polars predicate
// inside it (and for non-convex arms, where cross-line is not monotone).
use bekiper_math::sweep::band_eps;
use bekiper_perimeter::radial_maze::ArmFrame;

fn bool_lit(name: &str, values: &[bool]) -> Expr {
    lit(BooleanChunked::from_slice(name.into(), values).into_series())
}

type Pts = Vec<Option<(f64, f64)>>;

fn pts(df: &DataFrame, c: &CoordinateColumns) -> Pts {
    match (
        df.column(&c.x).and_then(|s| s.f64()),
        df.column(&c.y).and_then(|s| s.f64()),
    ) {
        (Ok(x), Ok(y)) => x.into_iter().zip(y).map(|(x, y)| x.zip(y)).collect(),
        _ => vec![None; df.height()],
    }
}

#[derive(Debug)]
pub(crate) struct ArmShared {
    pub head_in: Vec<bool>,
    pub tail_in: Vec<bool>,
    head: Pts,
    tail: Pts,
    /// `p*` per point (NaN = no kernel: legacy scalar decides).
    head_crit: Vec<f64>,
    tail_crit: Vec<f64>,
}

impl ArmShared {
    fn values(
        pts: &Pts,
        crit: &[f64],
        p: f64,
        kernel: impl Fn(f64, f64) -> bool,
        scalar: impl Fn(f64, f64) -> bool,
    ) -> Vec<bool> {
        pts.iter()
            .zip(crit)
            .map(|(pt, &c)| match pt {
                None => false,
                Some((x, y)) => {
                    if p > 0.0 && c.is_finite() && (c - p).abs() > band_eps(c, p) {
                        kernel(p, c)
                    } else {
                        scalar(*x, *y)
                    }
                }
            })
            .collect()
    }

    /// `threshold.past(point)` per frame (missing pose = false).
    pub fn past_values(&self, head: bool, threshold: &ArmThreshold, p: f64) -> Vec<bool> {
        let (pts, crit) = if head {
            (&self.head, &self.head_crit)
        } else {
            (&self.tail, &self.tail_crit)
        };
        Self::values(
            pts,
            crit,
            p,
            |p, c| p <= c,
            |x, y| threshold.past_scalar(x, y),
        )
    }

    /// `threshold.before(head)` per frame (missing pose = false).
    pub fn before_values(&self, threshold: &ArmThreshold, p: f64) -> Vec<bool> {
        Self::values(
            &self.head,
            &self.head_crit,
            p,
            |p, c| !(p <= c),
            |x, y| threshold.before_scalar(x, y),
        )
    }
}

#[derive(Debug)]
pub(crate) struct YMazeShared {
    pub arms: Vec<ArmShared>,
    pub center_polygon: Vec<bool>,
}

pub(crate) struct YMazeSharedPair {
    pub raw: std::sync::Arc<YMazeShared>,
    pub mitigated: std::sync::Arc<YMazeShared>,
}

impl YMazeShared {
    /// Geometry of `df` for `task`'s maze and landmarks. Arms whose frame cannot
    /// be resolved get NaN critical values and take the scalar predicate.
    pub fn build(task: &YMaze, df: &DataFrame) -> Self {
        use bekiper_core::shape::Shape;
        let (head, tail) = (
            pts(df, &task.params.head_coords),
            pts(df, &task.params.tail_coords),
        );
        let maze = &task.maze.shape;
        let center = maze.center.centroid();
        let arms = (0..task.n_arms())
            .map(|i| {
                let polygon = &maze.arms[i];
                let inside = |pts: &Pts| -> Vec<bool> {
                    pts.iter()
                        .map(|p| p.is_some_and(|(x, y)| polygon.contains(x, y)))
                        .collect()
                };
                let (head_in, tail_in) = (inside(&head), inside(&tail));
                let crit = |pts: &Pts, inside: &[bool]| -> Vec<f64> {
                    match task.threshold_geometry {
                        YMazeThresholdGeometry::ImenRadialVertexSpanV1 => {
                            let d: Vec<f64> = polygon
                                .vertices
                                .iter()
                                .map(|&(x, y)| {
                                    ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt()
                                })
                                .collect();
                            let near = d.iter().cloned().fold(f64::INFINITY, f64::min);
                            let far = d.iter().cloned().fold(0.0_f64, f64::max);
                            pts.iter()
                                .map(|p| match p {
                                    Some((x, y)) if far - near > 0.0 => {
                                        bekiper_perimeter::critical::radial_critical(
                                            center, near, far, *x, *y,
                                        )
                                    }
                                    _ => f64::NAN,
                                })
                                .collect()
                        }
                        YMazeThresholdGeometry::ArmStartCrosslineV1 => {
                            let frame: Option<ArmFrame> = maze.arm_frame(i).ok();
                            pts.iter()
                                .zip(inside)
                                .map(|(p, inside)| match (p, &frame) {
                                    (Some((x, y)), Some(f))
                                        if *inside && f.is_strictly_convex() =>
                                    {
                                        f.crossline_critical(*x, *y).unwrap_or(f64::NAN)
                                    }
                                    _ => f64::NAN,
                                })
                                .collect()
                        }
                    }
                };
                ArmShared {
                    head_crit: crit(&head, &head_in),
                    tail_crit: crit(&tail, &tail_in),
                    head_in,
                    tail_in,
                    head: head.clone(),
                    tail: tail.clone(),
                }
            })
            .collect();
        Self {
            arms,
            center_polygon: head
                .iter()
                .map(|p| p.is_some_and(|(x, y)| maze.center.contains(x, y)))
                .collect(),
        }
    }
}
