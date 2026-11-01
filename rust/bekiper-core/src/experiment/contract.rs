//! The reviewer-package design contract (`design/review-package/contract.toml`):
//! presentation blocks, their variant axes and the themekit tokens, shared by
//! the HTML renderer and the OpenPencil design system.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::ExperimentDescriptor;
use crate::error::{BekiperError, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contract {
    pub version: u32,
    /// Variant axis → values (`archetype`, `state`, `viewport`).
    pub axes: BTreeMap<String, Vec<String>>,
    pub tokens: Tokens,
    pub blocks: Vec<ContractBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokens {
    pub collection: String,
    pub modes: Vec<String>,
    pub colors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractBlock {
    pub id: String,
    /// OpenPencil ComponentSet name.
    pub name: String,
    pub axes: Vec<String>,
    /// Archetypes the block applies to; empty means all.
    #[serde(default)]
    pub archetypes: Vec<String>,
    pub properties: BTreeMap<String, PropertyKind>,
}

/// OpenPencil component property kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PropertyKind {
    Text,
    Visible,
    InstanceSwap,
}

fn invalid(message: String) -> BekiperError {
    BekiperError::Config(format!("review contract: {message}"))
}

impl Contract {
    /// Parses and validates a contract.
    pub fn parse(source: &str) -> Result<Self> {
        let contract: Self = toml::from_str(source).map_err(|e| invalid(e.to_string()))?;
        contract.validate()?;
        Ok(contract)
    }

    pub fn block(&self, id: &str) -> Option<&ContractBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Unique block ids; block axes are declared axes; archetypes are axis values.
    pub fn validate(&self) -> Result<()> {
        let archetypes = self.axes.get("archetype").cloned().unwrap_or_default();
        let mut ids = BTreeSet::new();
        for block in &self.blocks {
            if !ids.insert(&block.id) {
                return Err(invalid(format!("duplicate block {}", block.id)));
            }
            if let Some(axis) = block.axes.iter().find(|a| !self.axes.contains_key(*a)) {
                return Err(invalid(format!(
                    "block {} uses unknown axis {axis}",
                    block.id
                )));
            }
            if let Some(a) = block.archetypes.iter().find(|a| !archetypes.contains(a)) {
                return Err(invalid(format!(
                    "block {} names unknown archetype {a}",
                    block.id
                )));
            }
        }
        Ok(())
    }

    /// Every descriptor archetype and block id exists and the block applies to it.
    pub fn check_descriptors(&self, descriptors: &[ExperimentDescriptor]) -> Result<()> {
        let archetypes = self.axes.get("archetype").cloned().unwrap_or_default();
        for d in descriptors {
            let archetype = d.archetype.as_str();
            if !archetypes.iter().any(|a| a == archetype) {
                return Err(invalid(format!(
                    "{}: archetype {archetype} is not an axis value",
                    d.id
                )));
            }
            for rule in &d.blocks {
                let block = self
                    .block(&rule.id)
                    .ok_or_else(|| invalid(format!("{}: unknown block {}", d.id, rule.id)))?;
                if !block.archetypes.is_empty() && !block.archetypes.iter().any(|a| a == archetype)
                {
                    return Err(invalid(format!(
                        "{}: block {} does not apply to {archetype}",
                        d.id, rule.id
                    )));
                }
            }
        }
        Ok(())
    }
}
