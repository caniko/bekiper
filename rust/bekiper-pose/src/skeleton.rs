use serde::{Deserialize, Serialize};

use bekiper_core::error::{BekiperError, Result};

/// Built-in skeleton presets supported by topology-aware pose repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkeletonPreset {
    MouseTopView,
    MouseFull,
    ImenDlcTail,
    ImenDlcBaseTail,
}

impl SkeletonPreset {
    pub fn resolve(self) -> Skeleton {
        match self {
            Self::MouseTopView => Skeleton::mouse_top_view(),
            Self::MouseFull => Skeleton::mouse_full(),
            Self::ImenDlcTail => Skeleton::imen_dlc("tail"),
            Self::ImenDlcBaseTail => Skeleton::imen_dlc("base_tail"),
        }
    }
}

/// A named body part (keypoint) in a skeleton.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
}

impl Node {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// A directed edge between two skeleton nodes (parent → child).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
}

impl Edge {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

/// Skeleton topology: nodes (body parts) and edges (bones).
///
/// Used for validation, visualization, and computing derived features
/// (e.g. body length = sum of bone lengths along spine).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skeleton {
    pub name: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Skeleton {
    pub fn new(name: impl Into<String>, nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
        Self {
            name: name.into(),
            nodes,
            edges,
        }
    }

    /// All body part names in this skeleton.
    pub fn labels(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.name.clone()).collect()
    }

    /// Labels present in the skeleton but absent from an available set.
    pub fn missing_labels<'a, I>(&self, available: I) -> Vec<String>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let available: std::collections::HashSet<&str> = available.into_iter().collect();
        self.nodes
            .iter()
            .filter(|node| !available.contains(node.name.as_str()))
            .map(|node| node.name.clone())
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(BekiperError::Config(format!(
                "skeleton '{}' must define at least one node",
                self.name
            )));
        }

        let mut nodes = std::collections::HashSet::new();
        for node in &self.nodes {
            if node.name.is_empty() {
                return Err(BekiperError::Config(format!(
                    "skeleton '{}' contains an empty node label",
                    self.name
                )));
            }
            if !nodes.insert(node.name.as_str()) {
                return Err(BekiperError::Config(format!(
                    "skeleton '{}' contains duplicate node '{}'",
                    self.name, node.name
                )));
            }
        }

        let mut edges = std::collections::HashSet::new();
        for edge in &self.edges {
            if edge.source == edge.target {
                return Err(BekiperError::Config(format!(
                    "skeleton '{}' contains self-edge '{} -> {}'",
                    self.name, edge.source, edge.target
                )));
            }
            if !nodes.contains(edge.source.as_str()) {
                return Err(BekiperError::Config(format!(
                    "skeleton '{}' edge source '{}' is not a node",
                    self.name, edge.source
                )));
            }
            if !nodes.contains(edge.target.as_str()) {
                return Err(BekiperError::Config(format!(
                    "skeleton '{}' edge target '{}' is not a node",
                    self.name, edge.target
                )));
            }
            let key = (edge.source.as_str(), edge.target.as_str());
            if !edges.insert(key) {
                return Err(BekiperError::Config(format!(
                    "skeleton '{}' contains duplicate edge '{} -> {}'",
                    self.name, edge.source, edge.target
                )));
            }
        }

        Ok(())
    }

    pub fn validate_against_labels(&self, labels: &[String]) -> Result<()> {
        self.validate()?;

        let mut seen = std::collections::HashSet::new();
        for label in labels {
            if label.is_empty() {
                return Err(BekiperError::Config(
                    "model spec labels cannot contain empty strings".into(),
                ));
            }
            if !seen.insert(label.as_str()) {
                return Err(BekiperError::Config(format!(
                    "model spec labels contain duplicate label '{label}'"
                )));
            }
        }

        let missing = self.missing_labels(labels.iter().map(String::as_str));
        if !missing.is_empty() {
            return Err(BekiperError::Config(format!(
                "model spec labels do not cover skeleton '{}': missing {}",
                self.name,
                missing.join(", ")
            )));
        }

        Ok(())
    }

    /// Standard mouse skeleton for top-view tracking (DLC SuperAnimal compatible).
    ///
    /// 27 keypoints from DLC SuperAnimal-TopViewMouse:
    ///   Head: nose, ears, eyes, neck, head_midpoint
    ///   Spine: mid_back, mouse_center, mid_backend{1-3}, tail_base
    ///   Tail: tail1-5, tail_end
    ///   Limbs: shoulders, midsides, hips
    pub fn mouse_top_view() -> Self {
        let nodes = [
            "nose",
            "left_ear",
            "right_ear",
            "left_ear_tip",
            "right_ear_tip",
            "left_eye",
            "right_eye",
            "neck",
            "mid_back",
            "mouse_center",
            "mid_backend",
            "mid_backend2",
            "mid_backend3",
            "tail_base",
            "tail1",
            "tail2",
            "tail3",
            "tail4",
            "tail5",
            "left_shoulder",
            "left_midside",
            "left_hip",
            "right_shoulder",
            "right_midside",
            "right_hip",
            "tail_end",
            "head_midpoint",
        ]
        .into_iter()
        .map(Node::new)
        .collect();

        let edges = [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "head_midpoint"),
            ("left_ear", "left_ear_tip"),
            ("right_ear", "right_ear_tip"),
            ("left_eye", "head_midpoint"),
            ("right_eye", "head_midpoint"),
            ("head_midpoint", "neck"),
            ("neck", "mid_back"),
            ("mid_back", "mouse_center"),
            ("mouse_center", "mid_backend"),
            ("mid_backend", "mid_backend2"),
            ("mid_backend2", "mid_backend3"),
            ("mid_backend3", "tail_base"),
            ("tail_base", "tail1"),
            ("tail1", "tail2"),
            ("tail2", "tail3"),
            ("tail3", "tail4"),
            ("tail4", "tail5"),
            ("tail5", "tail_end"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("left_shoulder", "left_midside"),
            ("left_midside", "left_hip"),
            ("right_shoulder", "right_midside"),
            ("right_midside", "right_hip"),
        ]
        .into_iter()
        .map(|(s, t)| Edge::new(s, t))
        .collect();

        Self::new("mouse_top_view", nodes, edges)
    }

    /// Extended mouse skeleton with limbs (SLEAP-style, 13 keypoints).
    pub fn mouse_full() -> Self {
        let nodes = [
            "nose",
            "left_ear",
            "right_ear",
            "head_midpoint",
            "left_forepaw",
            "right_forepaw",
            "spine_midpoint",
            "left_hindpaw",
            "right_hindpaw",
            "tail_base",
            "tail_midpoint",
            "tail_tip",
            "belly",
        ]
        .into_iter()
        .map(Node::new)
        .collect();

        let edges = [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "head_midpoint"),
            ("head_midpoint", "left_forepaw"),
            ("head_midpoint", "right_forepaw"),
            ("head_midpoint", "spine_midpoint"),
            ("spine_midpoint", "belly"),
            ("spine_midpoint", "left_hindpaw"),
            ("spine_midpoint", "right_hindpaw"),
            ("spine_midpoint", "tail_base"),
            ("tail_base", "tail_midpoint"),
            ("tail_midpoint", "tail_tip"),
        ]
        .into_iter()
        .map(|(s, t)| Edge::new(s, t))
        .collect();

        Self::new("mouse_full", nodes, edges)
    }

    /// Four-point Imen DLC topology plus the derived ear midpoint.
    ///
    /// OFT/NORT files use `tail`; Y-maze files use `base_tail` for the same
    /// posterior keypoint, so the caller supplies the tail label.
    pub fn imen_dlc(tail_label: &'static str) -> Self {
        let nodes = ["nose", "left_ear", "right_ear", "center_ear", tail_label]
            .into_iter()
            .map(Node::new)
            .collect();

        let edges = [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("left_ear", "center_ear"),
            ("right_ear", "center_ear"),
            ("center_ear", tail_label),
        ]
        .into_iter()
        .map(|(s, t)| Edge::new(s, t))
        .collect();

        Self::new(format!("imen_dlc_{tail_label}"), nodes, edges)
    }
}
