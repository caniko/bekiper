#[pyclass(
    name = "HeuristicDecompositionComponent",
    module = "bekiper_rs",
    get_all
)]
#[derive(Clone)]
pub struct PyHeuristicDecompositionComponent {
    pub label: String,
    pub result_column: String,
    pub coordinate_labels: Vec<String>,
    pub perimeter_label: Option<String>,
    pub buffer_meters: Option<f64>,
    pub ray_origin_label: Option<String>,
    pub ray_direction_x_column: Option<String>,
    pub ray_direction_y_column: Option<String>,
    pub ray_target_x_column: Option<String>,
    pub ray_target_y_column: Option<String>,
    pub value_column: Option<String>,
    pub value_unit: Option<String>,
    pub evaluated_perimeter: Option<PyPerimeterSpec>,
    pub buffer_perimeter: Option<PyPerimeterSpec>,
}

#[pymethods]
impl PyHeuristicDecompositionComponent {
    #[new]
    #[pyo3(signature = (label, result_column, coordinate_labels, perimeter_label=None, buffer_meters=None, ray_origin_label=None, ray_direction_x_column=None, ray_direction_y_column=None, ray_target_x_column=None, ray_target_y_column=None, value_column=None, value_unit=None, evaluated_perimeter=None, buffer_perimeter=None))]
    fn new(
        label: String,
        result_column: String,
        coordinate_labels: Vec<String>,
        perimeter_label: Option<String>,
        buffer_meters: Option<f64>,
        ray_origin_label: Option<String>,
        ray_direction_x_column: Option<String>,
        ray_direction_y_column: Option<String>,
        ray_target_x_column: Option<String>,
        ray_target_y_column: Option<String>,
        value_column: Option<String>,
        value_unit: Option<String>,
        evaluated_perimeter: Option<PyPerimeterSpec>,
        buffer_perimeter: Option<PyPerimeterSpec>,
    ) -> Self {
        Self {
            label,
            result_column,
            coordinate_labels,
            perimeter_label,
            buffer_meters,
            ray_origin_label,
            ray_direction_x_column,
            ray_direction_y_column,
            ray_target_x_column,
            ray_target_y_column,
            value_column,
            value_unit,
            evaluated_perimeter,
            buffer_perimeter,
        }
    }
}

impl PyHeuristicDecompositionComponent {
    fn from_rust(py: Python<'_>, c: &HeuristicDecompositionComponent) -> PyResult<Self> {
        Ok(Self {
            label: c.label.clone(),
            result_column: c.result_column.clone(),
            coordinate_labels: c.coordinate_labels.clone(),
            perimeter_label: c.perimeter_label.clone(),
            buffer_meters: c.buffer_meters,
            ray_origin_label: c.ray_origin_label.clone(),
            ray_direction_x_column: c.ray_direction_x_column.clone(),
            ray_direction_y_column: c.ray_direction_y_column.clone(),
            ray_target_x_column: c.ray_target_x_column.clone(),
            ray_target_y_column: c.ray_target_y_column.clone(),
            value_column: c.value_column.clone(),
            value_unit: c.value_unit.clone(),
            evaluated_perimeter: c
                .evaluated_perimeter
                .as_ref()
                .map(|p| PyPerimeterSpec::from_rust(py, p))
                .transpose()?,
            buffer_perimeter: c
                .buffer_perimeter
                .as_ref()
                .map(|p| PyPerimeterSpec::from_rust(py, p))
                .transpose()?,
        })
    }
}

#[pyclass(name = "QualiaChannelSpec", module = "bekiper_rs", get_all)]
#[derive(Clone)]
pub struct PyQualiaChannelSpec {
    pub heuristic_type: String,
    pub per_object_seconds: Vec<f64>,
    pub per_object_bouts: Vec<u32>,
    pub relative_bias: Vec<f64>,
}

#[pymethods]
impl PyQualiaChannelSpec {
    #[new]
    fn new(
        heuristic_type: String,
        per_object_seconds: Vec<f64>,
        per_object_bouts: Vec<u32>,
        relative_bias: Vec<f64>,
    ) -> Self {
        Self {
            heuristic_type,
            per_object_seconds,
            per_object_bouts,
            relative_bias,
        }
    }
}

impl PyQualiaChannelSpec {
    fn from_rust(q: &QualiaChannelSpec) -> Self {
        Self {
            heuristic_type: q.heuristic_type.clone(),
            per_object_seconds: q.per_object_seconds.clone(),
            per_object_bouts: q.per_object_bouts.clone(),
            relative_bias: q.relative_bias.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Common styling fields shared by every FeatureInspection variant.
// ---------------------------------------------------------------------------
