//! PyO3 wrappers for the simple (non-enum) inspection types.

use bekiper_core::inspection::{
    CoordinateColumnSpec, HeuristicMeta, InspectionSettings, QuadrantSpec,
};
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_core::video::VideoMetadata;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// VideoMeta
// ---------------------------------------------------------------------------

#[pyclass(name = "VideoMeta", module = "bekiper_rs", get_all, set_all)]
#[derive(Clone)]
pub struct PyVideoMeta {
    pub fps: f64,
    pub total_frames: u64,
    pub resolution: (u32, u32),
    pub meters_per_pixel: f64,
    pub path: Option<String>,
}

#[pymethods]
impl PyVideoMeta {
    #[new]
    #[pyo3(signature = (fps, total_frames, resolution, meters_per_pixel, path=None))]
    fn new(
        fps: f64,
        total_frames: u64,
        resolution: (u32, u32),
        meters_per_pixel: f64,
        path: Option<String>,
    ) -> Self {
        Self {
            fps,
            total_frames,
            resolution,
            meters_per_pixel,
            path,
        }
    }

    #[getter]
    fn duration_seconds(&self) -> f64 {
        if self.fps > 0.0 {
            self.total_frames as f64 / self.fps
        } else {
            0.0
        }
    }

    fn meters_to_pixels(&self, value: f64) -> f64 {
        if self.meters_per_pixel > 0.0 {
            value / self.meters_per_pixel
        } else {
            value
        }
    }

    /// Convert a (N, 2) numpy array of metric coords to pixel coords by
    /// dividing by `meters_per_pixel`. The division is performed via
    /// Python's `__truediv__` so this works for any numeric array-like
    /// (numpy, polars Series, plain float) without dragging numpy into
    /// the Rust crate.
    fn pixel_coords_from_metric<'py>(
        &self,
        coords: Bound<'py, pyo3::PyAny>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        if self.meters_per_pixel > 0.0 {
            coords.call_method1("__truediv__", (self.meters_per_pixel,))
        } else {
            Ok(coords)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoMeta(fps={}, total_frames={}, resolution={:?}, meters_per_pixel={}, path={:?})",
            self.fps, self.total_frames, self.resolution, self.meters_per_pixel, self.path
        )
    }
}

impl PyVideoMeta {
    pub fn from_rust(v: &VideoMetadata, path: Option<String>) -> Self {
        Self {
            fps: v.fps,
            total_frames: v.total_frames,
            resolution: v.resolution,
            meters_per_pixel: *v.meters_per_pixel,
            path,
        }
    }
}

// ---------------------------------------------------------------------------
// CoordinateColumnSpec
// ---------------------------------------------------------------------------

#[pyclass(name = "CoordinateColumnSpec", module = "bekiper_rs", get_all)]
#[derive(Clone)]
pub struct PyCoordinateColumnSpec {
    pub x: String,
    pub y: String,
}

#[pymethods]
impl PyCoordinateColumnSpec {
    #[new]
    fn new(x: String, y: String) -> Self {
        Self { x, y }
    }

    fn __repr__(&self) -> String {
        format!("CoordinateColumnSpec(x={:?}, y={:?})", self.x, self.y)
    }
}

impl PyCoordinateColumnSpec {
    pub fn from_rust(s: &CoordinateColumnSpec) -> Self {
        Self {
            x: s.x.clone(),
            y: s.y.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// HeuristicMeta
// ---------------------------------------------------------------------------

#[pyclass(name = "HeuristicMeta", module = "bekiper_rs", get_all)]
#[derive(Clone)]
pub struct PyHeuristicMeta {
    pub name: String,
    pub result_column: String,
    pub true_frames: u64,
    pub total_frames: u64,
    pub seconds: f64,
    pub truth_spans: Option<Vec<(u64, u64)>>,
}

#[pymethods]
impl PyHeuristicMeta {
    #[new]
    #[pyo3(signature = (name, result_column, true_frames, total_frames, seconds, truth_spans=None))]
    fn new(
        name: String,
        result_column: String,
        true_frames: u64,
        total_frames: u64,
        seconds: f64,
        truth_spans: Option<Vec<(u64, u64)>>,
    ) -> Self {
        Self {
            name,
            result_column,
            true_frames,
            total_frames,
            seconds,
            truth_spans,
        }
    }

    #[getter]
    fn percentage(&self) -> f64 {
        if self.total_frames > 0 {
            self.true_frames as f64 / self.total_frames as f64 * 100.0
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HeuristicMeta(name={:?}, result_column={:?}, true_frames={}, total_frames={}, seconds={})",
            self.name, self.result_column, self.true_frames, self.total_frames, self.seconds
        )
    }
}

impl PyHeuristicMeta {
    pub fn from_rust(h: &HeuristicMeta) -> Self {
        let truth_spans = if h.truth_spans.is_empty() {
            None
        } else {
            Some(h.truth_spans.clone())
        };
        Self {
            name: h.name.clone(),
            result_column: h.result_column.clone(),
            true_frames: h.true_frames,
            total_frames: h.total_frames,
            seconds: h.seconds,
            truth_spans,
        }
    }
}

// ---------------------------------------------------------------------------
// QuadrantSpec
// ---------------------------------------------------------------------------

#[pyclass(name = "QuadrantSpec", module = "bekiper_rs", get_all)]
#[derive(Clone)]
pub struct PyQuadrantSpec {
    pub label: String,
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
    pub rows: u32,
    pub cols: u32,
}

#[pymethods]
impl PyQuadrantSpec {
    #[new]
    fn new(
        label: String,
        x_min: f64,
        y_min: f64,
        x_max: f64,
        y_max: f64,
        rows: u32,
        cols: u32,
    ) -> Self {
        Self {
            label,
            x_min,
            y_min,
            x_max,
            y_max,
            rows,
            cols,
        }
    }
}

impl PyQuadrantSpec {
    pub fn from_rust(q: &QuadrantSpec) -> Self {
        Self {
            label: q.label.clone(),
            x_min: q.x_min,
            y_min: q.y_min,
            x_max: q.x_max,
            y_max: q.y_max,
            rows: q.rows,
            cols: q.cols,
        }
    }
}

// ---------------------------------------------------------------------------
// InspectionSettings
// ---------------------------------------------------------------------------

#[pyclass(name = "InspectionSettings", module = "bekiper_rs", get_all)]
#[derive(Clone, Copy)]
pub struct PyInspectionSettings {
    pub invert_y: bool,
    pub minimum_seconds_tolerance: f64,
    pub maximum_seconds_distraction: f64,
    pub meters_per_pixel: f64,
}

#[pymethods]
impl PyInspectionSettings {
    #[new]
    #[pyo3(signature = (minimum_seconds_tolerance, maximum_seconds_distraction, meters_per_pixel, invert_y=false))]
    fn new(
        minimum_seconds_tolerance: f64,
        maximum_seconds_distraction: f64,
        meters_per_pixel: f64,
        invert_y: bool,
    ) -> Self {
        Self {
            minimum_seconds_tolerance,
            invert_y,
            maximum_seconds_distraction,
            meters_per_pixel,
        }
    }
}

impl PyInspectionSettings {
    pub fn from_rust(s: &InspectionSettings) -> Self {
        Self {
            minimum_seconds_tolerance: s.minimum_seconds_tolerance,
            invert_y: s.invert_y,
            maximum_seconds_distraction: s.maximum_seconds_distraction,
            meters_per_pixel: s.meters_per_pixel,
        }
    }
}

// ---------------------------------------------------------------------------
// ScaleDerivation (mirrors ScaleDerivationInfo)
// ---------------------------------------------------------------------------

#[pyclass(name = "ScaleDerivation", module = "bekiper_rs", get_all)]
#[derive(Clone)]
pub struct PyScaleDerivation {
    pub method: String,
    pub physical_meters: Option<f64>,
    pub pixel_measurement: Option<f64>,
    pub arm_pixel_spans: Vec<f64>,
    pub reference_label: Option<String>,
    pub meters_per_pixel: f64,
    pub endpoints: Option<((f64, f64), (f64, f64))>,
    /// Vertical metres/pixel for non-square source pixels; `None` = isotropic.
    pub meters_per_pixel_y: Option<f64>,
}

#[pymethods]
impl PyScaleDerivation {
    #[new]
    #[pyo3(signature = (method, physical_meters, pixel_measurement, arm_pixel_spans, reference_label, meters_per_pixel, endpoints=None, meters_per_pixel_y=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        method: String,
        physical_meters: Option<f64>,
        pixel_measurement: Option<f64>,
        arm_pixel_spans: Vec<f64>,
        reference_label: Option<String>,
        meters_per_pixel: f64,
        endpoints: Option<((f64, f64), (f64, f64))>,
        meters_per_pixel_y: Option<f64>,
    ) -> Self {
        Self {
            method,
            physical_meters,
            pixel_measurement,
            arm_pixel_spans,
            reference_label,
            meters_per_pixel,
            endpoints,
            meters_per_pixel_y,
        }
    }
}

impl PyScaleDerivation {
    pub fn from_rust(s: &ScaleDerivationInfo) -> Self {
        let endpoints = s
            .endpoints
            .map(|e| ((e[0][0], e[0][1]), (e[1][0], e[1][1])));
        Self {
            method: s.method.clone(),
            physical_meters: s.physical_meters,
            pixel_measurement: s.pixel_measurement,
            arm_pixel_spans: s.arm_pixel_spans.clone(),
            reference_label: s.reference_label.clone(),
            meters_per_pixel: s.meters_per_pixel,
            endpoints,
            meters_per_pixel_y: s.meters_per_pixel_y,
        }
    }
}

// ---------------------------------------------------------------------------
// IngressSessionContext
// ---------------------------------------------------------------------------

#[pyclass(name = "IngressSessionContext", module = "bekiper_rs", get_all)]
#[derive(Clone, Default)]
pub struct PyIngressSessionContext {
    pub profile: String,
    pub source_file: Option<String>,
    pub source_row: Option<usize>,
    pub animal: Option<String>,
    pub trial: Option<String>,
    pub sequence: Option<String>,
    pub phase: Option<String>,
    pub phase_part: Option<String>,
    pub stage: Option<String>,
    pub apparatus: Option<String>,
    pub treatment: Option<String>,
    pub group: Option<String>,
    pub sex: Option<String>,
    pub age: Option<String>,
    pub genotype: Option<String>,
    pub object_a: Option<String>,
    pub object_b: Option<String>,
    pub fecal_boli_count: Option<u32>,
    pub comments: Option<String>,
}

#[pymethods]
impl PyIngressSessionContext {
    #[new]
    #[pyo3(signature = (
        profile,
        source_file=None, source_row=None,
        animal=None, trial=None, sequence=None, phase=None, phase_part=None,
        stage=None, apparatus=None, treatment=None, group=None,
        sex=None, age=None, genotype=None, object_a=None, object_b=None,
        fecal_boli_count=None, comments=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        profile: String,
        source_file: Option<String>,
        source_row: Option<usize>,
        animal: Option<String>,
        trial: Option<String>,
        sequence: Option<String>,
        phase: Option<String>,
        phase_part: Option<String>,
        stage: Option<String>,
        apparatus: Option<String>,
        treatment: Option<String>,
        group: Option<String>,
        sex: Option<String>,
        age: Option<String>,
        genotype: Option<String>,
        object_a: Option<String>,
        object_b: Option<String>,
        fecal_boli_count: Option<u32>,
        comments: Option<String>,
    ) -> Self {
        Self {
            profile,
            source_file,
            source_row,
            animal,
            trial,
            sequence,
            phase,
            phase_part,
            stage,
            apparatus,
            treatment,
            group,
            sex,
            age,
            genotype,
            object_a,
            object_b,
            fecal_boli_count,
            comments,
        }
    }
}

impl PyIngressSessionContext {
    pub fn from_rust(c: &bekiper_core::inspection::IngressSessionContext) -> Self {
        Self {
            profile: c.profile.clone(),
            source_file: c.source_file.clone(),
            source_row: c.source_row,
            animal: c.animal.clone(),
            trial: c.trial.clone(),
            sequence: c.sequence.clone(),
            phase: c.phase.clone(),
            phase_part: c.phase_part.clone(),
            stage: c.stage.clone(),
            apparatus: c.apparatus.clone(),
            treatment: c.treatment.clone(),
            group: c.group.clone(),
            sex: c.sex.clone(),
            age: c.age.clone(),
            genotype: c.genotype.clone(),
            object_a: c.object_a.clone(),
            object_b: c.object_b.clone(),
            fecal_boli_count: c.fecal_boli_count,
            comments: c.comments.clone(),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVideoMeta>()?;
    m.add_class::<PyCoordinateColumnSpec>()?;
    m.add_class::<PyHeuristicMeta>()?;
    m.add_class::<PyQuadrantSpec>()?;
    m.add_class::<PyInspectionSettings>()?;
    m.add_class::<PyScaleDerivation>()?;
    m.add_class::<PyIngressSessionContext>()?;
    Ok(())
}
