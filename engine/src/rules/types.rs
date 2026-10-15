use serde::{Deserialize, Serialize};

pub const D_MODEL: usize = 256;
pub const N_ENTITIES: usize = 64;
pub const CARD_POOL_SIZE: usize = 2048;

pub const E_STATIC_DIM: usize = 128;
pub const E_DYNAMIC_DIM: usize = 96;
pub const E_ZONE_DIM: usize = 32;
pub const RAW_FEATURE_DIM: usize = E_STATIC_DIM + E_DYNAMIC_DIM + E_ZONE_DIM; // 256

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZoneType {
    ActiveSelf = 0,
    BenchSelf = 1,
    HandSelf = 2,
    DiscardSelf = 3,
    LostSelf = 4,
    PrizeSelf = 5,
    ActiveOpp = 6,
    BenchOpp = 7,
    HandOppRevealed = 8,
    DiscardOpp = 9,
    LostOpp = 10,
    Stadium = 11,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Zone {
    ActiveSelf,
    BenchSelf(u8),
    HandSelf(u8),
    DiscardSelf,
    LostSelf,
    PrizeSelf(u8),
    ActiveOpp,
    BenchOpp(u8),
    HandOppRevealed(u8),
    DiscardOpp,
    LostOpp,
    Stadium,
    GlobalToken,
}

impl Zone {
    pub fn zone_type(&self) -> Option<ZoneType> {
        match self {
            Zone::ActiveSelf => Some(ZoneType::ActiveSelf),
            Zone::BenchSelf(_) => Some(ZoneType::BenchSelf),
            Zone::HandSelf(_) => Some(ZoneType::HandSelf),
            Zone::DiscardSelf => Some(ZoneType::DiscardSelf),
            Zone::LostSelf => Some(ZoneType::LostSelf),
            Zone::PrizeSelf(_) => Some(ZoneType::PrizeSelf),
            Zone::ActiveOpp => Some(ZoneType::ActiveOpp),
            Zone::BenchOpp(_) => Some(ZoneType::BenchOpp),
            Zone::HandOppRevealed(_) => Some(ZoneType::HandOppRevealed),
            Zone::DiscardOpp => Some(ZoneType::DiscardOpp),
            Zone::LostOpp => Some(ZoneType::LostOpp),
            Zone::Stadium => Some(ZoneType::Stadium),
            Zone::GlobalToken => None,
        }
    }

    pub fn controller(&self) -> f32 {
        match self {
            Zone::ActiveSelf | Zone::BenchSelf(_) | Zone::HandSelf(_) |
            Zone::DiscardSelf | Zone::LostSelf | Zone::PrizeSelf(_) => 1.0,
            Zone::ActiveOpp | Zone::BenchOpp(_) | Zone::HandOppRevealed(_) |
            Zone::DiscardOpp | Zone::LostOpp => -1.0,
            Zone::Stadium | Zone::GlobalToken => 0.0,
        }
    }

    pub fn slot_index(&self) -> u8 {
        match self {
            Zone::BenchSelf(idx) | Zone::BenchOpp(idx) => *idx,
            Zone::PrizeSelf(idx) => *idx,
            Zone::HandSelf(idx) | Zone::HandOppRevealed(idx) => *idx,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EnergyType {
    Grass = 0,
    Fire = 1,
    Water = 2,
    Lightning = 3,
    Psychic = 4,
    Fighting = 5,
    Darkness = 6,
    Metal = 7,
    Colorless = 8,
    Special = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CardCategory {
    BasicPokemon = 0,
    Stage1Pokemon = 1,
    Stage2Pokemon = 2,
    VStarPokemon = 3,
    ExPokemon = 4,
    Item = 5,
    Supporter = 6,
    Tool = 7,
    Stadium = 8,
    BasicEnergy = 9,
    SpecialEnergy = 10,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MacroActionType {
    Attack = 0,
    Retreat = 1,
    PlayItem = 2,
    PlaySupporter = 3,
    AttachEnergy = 4,
    UseAbility = 5,
    Pass = 6,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchFilter {
    AnyPokemon,
    BasicPokemon,
    EvolutionPokemon,
    TrainerCard,
    AnyCard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubState {
    IdleRoot,
    SelectDiscard1 { card_played_idx: usize },
    SelectDiscard2 { card_played_idx: usize, discard_1_idx: usize },
    SelectSearchTarget { card_played_idx: usize, filter: SearchFilter },
    SelectBenchTarget { source_idx: usize },
    SelectEnergyTarget { energy_card_idx: usize },
    SelectRetreatEnergyDiscard { active_idx: usize, remaining_cost: u8 },
}
