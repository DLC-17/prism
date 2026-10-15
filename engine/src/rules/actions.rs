use super::types::MacroActionType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    // Macro actions (root turn decisions)
    PlayItem { hand_idx: usize },
    PlaySupporter { hand_idx: usize },
    AttachEnergy { hand_idx: usize, target_entity_idx: usize },
    UseAbility { entity_idx: usize, ability_id: u8 },
    Retreat { target_bench_idx: usize },
    Attack { attack_idx: u8 },
    Pass,

    // Micro actions (intra-card atomic resolutions)
    SelectEntity { entity_idx: usize },
    SelectDeckCard { card_id: u16 },
    ConfirmSelection,
    FailToFind,
}

impl Action {
    pub fn macro_type(&self) -> Option<MacroActionType> {
        match self {
            Action::Attack { .. } => Some(MacroActionType::Attack),
            Action::Retreat { .. } => Some(MacroActionType::Retreat),
            Action::PlayItem { .. } => Some(MacroActionType::PlayItem),
            Action::PlaySupporter { .. } => Some(MacroActionType::PlaySupporter),
            Action::AttachEnergy { .. } => Some(MacroActionType::AttachEnergy),
            Action::UseAbility { .. } => Some(MacroActionType::UseAbility),
            Action::Pass => Some(MacroActionType::Pass),
            _ => None,
        }
    }
}
