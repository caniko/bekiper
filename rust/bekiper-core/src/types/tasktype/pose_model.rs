impl PoseModelPreset {
    pub fn as_str(&self) -> &str {
        match self {
            Self::DlcSuperAnimal => "dlc-super-animal",
            Self::SleapSingleInstance => "sleap-single-instance",
            Self::LightningPose => "lightning-pose",
            Self::Custom(p) => p.to_str().unwrap_or("custom"),
        }
    }
}

impl fmt::Display for PoseModelPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for PoseModelPreset {
    fn from(s: String) -> Self {
        match s.as_str() {
            "dlc-super-animal" | "deeplabcut" => Self::DlcSuperAnimal,
            "sleap-single-instance" | "sleap" => Self::SleapSingleInstance,
            "lightning-pose" | "lightning" => Self::LightningPose,
            _ => Self::Custom(PathBuf::from(s)),
        }
    }
}

impl From<PoseModelPreset> for String {
    fn from(p: PoseModelPreset) -> Self {
        p.as_str().to_string()
    }
}

// ---------------------------------------------------------------------------
// YMazeArm — the three arms of a Y-maze
// ---------------------------------------------------------------------------

/// Y-maze arm identifier. Always exactly three arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum YMazeArm {
    A,
    B,
    C,
}
