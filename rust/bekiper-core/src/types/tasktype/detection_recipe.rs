impl DetectionRecipe {
    /// Object roles required by the annotation and analysis contract.
    pub fn required_annotation_labels(self) -> &'static [&'static str] {
        match self {
            Self::NortTraining => &["variable", "familiar"],
            Self::NortNovelty => &["novel", "familiar"],
            _ => &[],
        }
    }

    pub fn forbidden_annotation_labels(self) -> &'static [&'static str] {
        match self {
            Self::NortTraining => &["novel"],
            Self::NortNovelty => &["variable"],
            _ => &[],
        }
    }

    pub fn annotation_label_errors<'a>(
        self,
        labels: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let labels: Vec<_> = labels.into_iter().collect();
        let mut errors = Vec::new();
        for required in self.required_annotation_labels() {
            match labels
                .iter()
                .filter(|label| label.trim().eq_ignore_ascii_case(required))
                .count()
            {
                0 => errors.push(format!("Missing annotation: {required}")),
                1 => {}
                count => errors.push(format!("Duplicate annotation: {required} ({count})")),
            }
        }
        for forbidden in self.forbidden_annotation_labels() {
            if labels
                .iter()
                .any(|label| label.trim().eq_ignore_ascii_case(forbidden))
            {
                errors.push(format!("{forbidden} is not valid for {}", self.as_str()));
            }
        }
        errors
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NortTraining => "nort_training",
            Self::NortNovelty => "nort_novelty",
            Self::OulTraining => "oul_training",
            Self::OulUpdate => "oul_update",
            Self::OulTest => "oul_test",
            Self::Oft => "oft",
            Self::YMaze => "ymaze",
        }
    }

    /// If this detection recipe corresponds to an object-recognition protocol,
    /// return the matching behavioural protocol variant.
    pub fn object_recognition_protocol(&self) -> Option<ObjectRecognitionProtocol> {
        match self {
            Self::NortTraining => Some(ObjectRecognitionProtocol::NortTraining),
            Self::NortNovelty => Some(ObjectRecognitionProtocol::NortNovelty),
            Self::OulTraining => Some(ObjectRecognitionProtocol::OulTraining),
            Self::OulUpdate => Some(ObjectRecognitionProtocol::OulUpdate),
            Self::OulTest => Some(ObjectRecognitionProtocol::OulTest),
            Self::Oft | Self::YMaze => None,
        }
    }
}

impl fmt::Display for DetectionRecipe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for DetectionRecipe {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nort_training" | "nort" => Ok(Self::NortTraining),
            "nort_novelty" => Ok(Self::NortNovelty),
            "oul_training" => Ok(Self::OulTraining),
            "oul_update" => Ok(Self::OulUpdate),
            "oul_test" => Ok(Self::OulTest),
            "oft" | "open_field" => Ok(Self::Oft),
            "ymaze" | "y_maze" => Ok(Self::YMaze),
            other => Err(format!(
                "unknown detection recipe: '{other}'. Expected: nort_training, nort_novelty, oul_training, oul_update, oul_test, oft, ymaze"
            )),
        }
    }
}

impl Serialize for DetectionRecipe {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DetectionRecipe {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// ObjectRecognitionProtocol — stage-aware object-recognition semantics
// ---------------------------------------------------------------------------
