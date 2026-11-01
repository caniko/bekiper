// Variant-invariant geometry of one object, shared by every variant of a
// recording (design: docs/src/pipeline/multi-parameter-runs.md, 4.1/4.3).
//
// Critical values (`d*`, `e*`, the stored fan at `L_max`) turn the swept
// distance/erosion/length parameters into comparisons. Inside the float band
// (`bekiper_math::sweep::band_eps`) the legacy scalar predicate decides, so the
// columns are bit-equal to the polars geometry they replace.

type Point = Option<(f64, f64)>;

fn points(df: &DataFrame, coords: &CoordinateColumns) -> PolarsResult<Vec<Point>> {
    let (xs, ys) = (df.column(&coords.x)?.f64()?, df.column(&coords.y)?.f64()?);
    Ok(xs.into_iter().zip(ys).map(|(x, y)| x.zip(y)).collect())
}

#[derive(Debug, Clone)]
struct SharedFan {
    apex_fraction: f64,
    half_angle_deg: f64,
    l_max: f64,
    ear: Vec<Point>,
    hits: Vec<Option<bekiper_perimeter::fan::FanHit>>,
    outline: Vec<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub struct SharedObjectGeometry {
    shape: bekiper_perimeter::object_shape::ObjectShape,
    kernel: Option<bekiper_perimeter::critical::OffsetKernel>,
    centroid: Vec<Point>,
    nose: Vec<Point>,
    /// `shape.contains(centroid)`, missing = false.
    centroid_inside: Vec<bool>,
    /// Largest erosion keeping the centroid inside (NaN = missing).
    centroid_erode: Vec<f64>,
    /// Smallest nose offset containing the nose (NaN = missing).
    nose_expand: Vec<f64>,
    angle: [Series; 3],
    fan: Option<SharedFan>,
}

fn bool_expr(name: &str, values: Vec<bool>) -> Expr {
    lit(BooleanChunked::from_slice(name.into(), &values).into_series())
}

impl SharedObjectGeometry {
    /// `fan` carries the apex fraction, half angle and the *longest* length of
    /// the variant group.
    pub fn build(
        shape: &bekiper_perimeter::object_shape::ObjectShape,
        df: &DataFrame,
        centroid: &CoordinateColumns,
        nose: &CoordinateColumns,
        head_base: &CoordinateColumns,
        fan: Option<crate::heuristic::solo::whiskers::WhiskerFanParams>,
    ) -> PolarsResult<Self> {
        use bekiper_core::shape::Shape;
        let kernel = bekiper_perimeter::critical::OffsetKernel::new(shape);
        let centroid_pts = points(df, centroid)?;
        let nose_pts = points(df, nose)?;
        let crit =
            |pts: &[Point],
             f: &dyn Fn(&bekiper_perimeter::critical::OffsetKernel, f64, f64) -> f64| {
                pts.iter()
                    .map(|p| match (p, &kernel) {
                        (Some((x, y)), Some(k)) => f(k, *x, *y),
                        _ => f64::NAN,
                    })
                    .collect::<Vec<_>>()
            };
        let angle_exprs =
            shape.nearest_point_heading_exprs(&nose.x, &nose.y, "snout_dx", "snout_dy");
        let [a, tx, ty] = angle_exprs;
        let angle = df
            .clone()
            .lazy()
            .select([a.alias("angle"), tx.alias("target_x"), ty.alias("target_y")])
            .collect()?;
        let angle_series = |i: usize| angle.columns()[i].as_materialized_series().clone();
        let fan = fan
            .map(|params| -> PolarsResult<SharedFan> {
                let spec = crate::heuristic::solo::whiskers::WhiskerFanSpec::new(params, shape);
                let ear = points(df, head_base)?;
                let hits = ear
                    .iter()
                    .zip(&nose_pts)
                    .map(|(e, n)| match (e, n) {
                        (Some(e), Some(n)) => spec.hit_at(*e, *n),
                        _ => None,
                    })
                    .collect();
                Ok(SharedFan {
                    apex_fraction: params.apex_fraction,
                    half_angle_deg: params.half_angle_deg,
                    l_max: params.length_m,
                    ear,
                    hits,
                    outline: spec.outline,
                })
            })
            .transpose()?;
        Ok(Self {
            centroid_inside: centroid_pts
                .iter()
                .map(|p| p.is_some_and(|(x, y)| shape.contains(x, y)))
                .collect(),
            centroid_erode: crit(&centroid_pts, &|k, x, y| k.erode_critical(x, y)),
            nose_expand: crit(&nose_pts, &|k, x, y| k.expand_critical(x, y)),
            shape: shape.clone(),
            kernel,
            centroid: centroid_pts,
            nose: nose_pts,
            angle: [angle_series(0), angle_series(1), angle_series(2)],
            fan,
        })
    }

    /// Whether `nose_confinement_expr(distance)` reproduces the legacy mask.
    pub fn supports_expand(&self, distance: f64) -> bool {
        self.kernel.is_some() && distance > 0.0 && distance.is_finite()
    }

    /// `object.expanded(distance).confinement_mask(nose)` with missing = false.
    pub fn nose_confinement_expr(&self, distance: f64) -> Expr {
        use bekiper_core::shape::Shape;
        let mut expanded: Option<bekiper_perimeter::object_shape::ObjectShape> = None;
        let values = self
            .nose
            .iter()
            .zip(&self.nose_expand)
            .map(|(p, &c)| match p {
                None => false,
                Some((x, y)) => {
                    if c.is_finite()
                        && (c - distance).abs() > bekiper_math::sweep::band_eps(c, distance)
                    {
                        c <= distance
                    } else {
                        expanded
                            .get_or_insert_with(|| self.shape.expand(distance))
                            .contains(*x, *y)
                    }
                }
            })
            .collect();
        bool_expr("nose_confinement", values)
    }

    /// The `outside` mask of the legacy exploration heuristic, or `None` when
    /// the erosion must take the legacy path (collapsed or non-positive-kernel).
    pub fn outside_expr(&self, erosion: f64) -> Option<Expr> {
        use bekiper_core::shape::Shape;
        if !(erosion > 0.0) {
            let values = self.centroid_inside.iter().map(|inside| !inside).collect();
            return Some(bool_expr("outside", values));
        }
        let kernel = self.kernel.as_ref()?;
        if !kernel.erosion_applies(erosion) {
            return None;
        }
        let mut eroded: Option<bekiper_perimeter::object_shape::ObjectShape> = None;
        let values = self
            .centroid
            .iter()
            .zip(&self.centroid_erode)
            .map(|(p, &c)| match p {
                None => true,
                Some((x, y)) => {
                    let inside = if c.is_finite()
                        && (c - erosion).abs() > bekiper_math::sweep::band_eps(c, erosion)
                    {
                        erosion <= c
                    } else {
                        eroded
                            .get_or_insert_with(|| self.shape.expand(-erosion))
                            .contains(*x, *y)
                    };
                    !inside
                }
            })
            .collect();
        Some(bool_expr("outside", values))
    }

    /// `[angle_deg, target_x, target_y]` of the nearest-point heading.
    pub fn angle_exprs(&self) -> [Expr; 3] {
        [
            lit(self.angle[0].clone()),
            lit(self.angle[1].clone()),
            lit(self.angle[2].clone()),
        ]
    }

    /// `[hit, side, distance_m]` for a fan of `params`, derived from the fan
    /// computed once at `L_max`. `None` when the shared fan does not apply.
    pub fn fan_exprs(
        &self,
        params: &crate::heuristic::solo::whiskers::WhiskerFanParams,
    ) -> Option<[Expr; 3]> {
        let fan = self.fan.as_ref()?;
        if fan.apex_fraction != params.apex_fraction
            || fan.half_angle_deg != params.half_angle_deg
            || !(params.length_m <= fan.l_max)
        {
            return None;
        }
        let spec = crate::heuristic::solo::whiskers::WhiskerFanSpec {
            params: *params,
            outline: fan.outline.clone(),
        };
        let hits: Vec<Option<bekiper_perimeter::fan::FanHit>> = fan
            .hits
            .iter()
            .enumerate()
            .map(
                |(i, hit)| match bekiper_perimeter::fan::narrow_hit(*hit, params.length_m) {
                    Some(narrowed) => narrowed,
                    // A stored hit implies both points exist.
                    None => fan.ear[i]
                        .zip(self.nose[i])
                        .and_then(|(e, n)| spec.hit_at(e, n)),
                },
            )
            .collect();
        let hit = BooleanChunked::from_iter_values(
            "whisker_fan_hit".into(),
            hits.iter().map(Option::is_some),
        );
        let side = StringChunked::from_iter_options(
            "whisker_fan_side".into(),
            hits.iter().map(|h| h.map(|h| h.side.as_str())),
        );
        let distance = Float64Chunked::from_iter_options(
            "whisker_fan_distance_m".into(),
            hits.iter().map(|h| h.map(|h| h.distance)),
        );
        Some([
            lit(hit.into_series()),
            lit(side.into_series()),
            lit(distance.into_series()),
        ])
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use crate::heuristic::solo::whiskers::WhiskerFanParams;
    use bekiper_perimeter::object_shape::ObjectShape;
    use bekiper_perimeter::rectangle::Rectangle;

    const EAR: (f64, f64) = (-0.05, -0.03);
    const NOSE: (f64, f64) = (0.0, -0.03);

    fn shape() -> ObjectShape {
        ObjectShape::Rectangle(Rectangle::new(0.0, 0.0, 0.04, 0.04))
    }

    fn params(length_m: f64) -> WhiskerFanParams {
        WhiskerFanParams {
            length_m,
            ..Default::default()
        }
    }

    fn geometry() -> SharedObjectGeometry {
        let df = df!("cx" => [0.0], "cy" => [0.0], "nx" => [NOSE.0], "ny" => [NOSE.1],
            "ex" => [EAR.0], "ey" => [EAR.1], "snout_dx" => [0.05], "snout_dy" => [0.0])
        .unwrap();
        let c = |x: &str, y: &str| CoordinateColumns::new(x, y);
        SharedObjectGeometry::build(
            &shape(),
            &df,
            &c("cx", "cy"),
            &c("nx", "ny"),
            &c("ex", "ey"),
            Some(params(0.02)),
        )
        .unwrap()
    }

    #[test]
    fn collapsed_erosion_takes_the_legacy_path() {
        let g = geometry();
        assert!(g.outside_expr(0.05).is_none());
        assert!(g.outside_expr(0.005).is_some());
    }

    #[test]
    fn fan_exprs_require_a_compatible_stored_fan() {
        let g = geometry();
        assert!(g.fan_exprs(&params(0.03)).is_none());
        let mut other = params(0.01);
        other.half_angle_deg += 1.0;
        assert!(g.fan_exprs(&other).is_none());
        assert!(g.fan_exprs(&params(0.01)).is_some());
        // A length equal to the stored hit distance falls in the float band and
        // is re-evaluated by the scalar fan.
        let spec = WhiskerFanSpec::new(params(0.02), &shape());
        let d = spec.hit_at(EAR, NOSE).unwrap().distance;
        assert!(g.fan_exprs(&params(d)).is_some());
    }
}
