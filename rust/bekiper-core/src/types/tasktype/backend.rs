impl DetectionBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PreComputed => "pre_computed",
            Self::ManualAnnotation => "manual_annotation",
            Self::ClassicalCv => "classical_cv",
            Self::ArenaCalibrationCv => "arena_calibration_cv",
            Self::Sam2Auto => "sam2_auto",
            Self::GroundingDinoSam2 => "grounding_dino_sam2",
        }
    }

    /// Whether this backend needs a GPU (and therefore must be serialised).
    pub fn requires_gpu(&self) -> bool {
        match self {
            Self::PreComputed
            | Self::ManualAnnotation
            | Self::ClassicalCv
            | Self::ArenaCalibrationCv => false,
            Self::Sam2Auto | Self::GroundingDinoSam2 => true,
        }
    }
}

impl fmt::Display for DetectionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for DetectionBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pre_computed" | "precomputed" => Ok(Self::PreComputed),
            "manual_annotation" | "manual" | "annotation" => Ok(Self::ManualAnnotation),
            "classical_cv" | "cv" => Ok(Self::ClassicalCv),
            "arena_calibration_cv" | "arena_cv" | "arena_calibration" => {
                Ok(Self::ArenaCalibrationCv)
            }
            "sam2_auto" | "sam2" => Ok(Self::Sam2Auto),
            "grounding_dino_sam2" | "grounding_dino" | "gdino" => Ok(Self::GroundingDinoSam2),
            other => Err(format!(
                "unknown detection backend: '{other}'. Expected: pre_computed, manual_annotation, arena_calibration_cv, classical_cv, sam2_auto, grounding_dino_sam2"
            )),
        }
    }
}

impl Serialize for DetectionBackend {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DetectionBackend {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// PoseModelPreset — known model specifications
// ---------------------------------------------------------------------------
