impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ObjectRecognition => "object_recognition",
            Self::YMaze => "y_maze",
            Self::OpenField => "open_field",
            Self::Cheeseboard => "cheeseboard",
        }
    }
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TaskType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "object_recognition"
            | "novel_object_recognition"
            | "objects_in_updating_locations"
            | "nort"
            | "oul" => Ok(Self::ObjectRecognition),
            "y_maze" | "ymaze" | "radial_arm" => Ok(Self::YMaze),
            "open_field" | "open_field_test" | "oft" => Ok(Self::OpenField),
            "cheeseboard" | "reward_tracing" => Ok(Self::Cheeseboard),
            other => Err(format!(
                "unknown task type: '{other}'. Expected: object_recognition, y_maze, open_field, cheeseboard (or aliases)"
            )),
        }
    }
}

impl Serialize for TaskType {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TaskType {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// DetectionRecipe — finite set of auto-detection protocols
// ---------------------------------------------------------------------------
