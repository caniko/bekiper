impl ObjectRecognitionProtocol {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NortTraining => "nort_training",
            Self::NortNovelty => "nort_novelty",
            Self::OulTraining => "oul_training",
            Self::OulUpdate => "oul_update",
            Self::OulTest => "oul_test",
        }
    }

    /// Protocols where the first two objects have familiar-vs-novel semantics
    /// and therefore support NORT discrimination/recognition indices.
    pub fn computes_novelty_indices(&self) -> bool {
        matches!(self, Self::NortNovelty)
    }

    /// Protocols where left/right or object A/B bias remains meaningful even
    /// though the objects are not novel-vs-familiar.
    pub fn is_training_like(&self) -> bool {
        matches!(
            self,
            Self::NortTraining | Self::OulTraining | Self::OulUpdate
        )
    }
}

impl fmt::Display for ObjectRecognitionProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ObjectRecognitionProtocol {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nort_training" | "nort_t1" => Ok(Self::NortTraining),
            "nort_novelty" | "nort_test" | "nort_t2" => Ok(Self::NortNovelty),
            "oul_training" => Ok(Self::OulTraining),
            "oul_update" => Ok(Self::OulUpdate),
            "oul_test" => Ok(Self::OulTest),
            other => Err(format!(
                "unknown object-recognition protocol: '{other}'. Expected: nort_training, nort_novelty, oul_training, oul_update, oul_test"
            )),
        }
    }
}

impl Serialize for ObjectRecognitionProtocol {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ObjectRecognitionProtocol {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// DetectionBackend — which detection method to use
// ---------------------------------------------------------------------------
