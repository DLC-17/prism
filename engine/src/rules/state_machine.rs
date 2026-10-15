use super::card_db::CardCatalog;
use super::types::{CardCategory, SubState, Zone, N_ENTITIES, RAW_FEATURE_DIM};
use super::zone_manager::{encode_zone, PokemonEntity, ThreeSegmentDeck};
use serde::{Deserialize, Serialize};

pub const ACTIVE_SELF_SLOT: usize = 0;
pub const BENCH_SELF_START: usize = 1;
pub const BENCH_SELF_END: usize = 5; // 1..=5 (5 bench slots)
pub const STADIUM_SLOT: usize = 6;
pub const HAND_SELF_START: usize = 7;
pub const HAND_SELF_END: usize = 26; // 7..=26 (20 hand slots)
pub const PRIZE_SELF_START: usize = 27;
pub const PRIZE_SELF_END: usize = 32; // 27..=32 (6 prize slots)
pub const DISCARD_SELF_SLOT: usize = 33;
pub const LOST_SELF_SLOT: usize = 34;

pub const ACTIVE_OPP_SLOT: usize = 35;
pub const BENCH_OPP_START: usize = 36;
pub const BENCH_OPP_END: usize = 40; // 36..=40 (5 bench slots)
pub const HAND_OPP_START: usize = 41;
pub const HAND_OPP_END: usize = 46; // 41..=46
pub const DISCARD_OPP_SLOT: usize = 47;
pub const LOST_OPP_SLOT: usize = 48;
pub const GLOBAL_TOKEN_SLOT: usize = 49;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySlot {
    pub card_id: u16,
    pub zone: Zone,
    pub pokemon: Option<PokemonEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub entities: Vec<Option<EntitySlot>>, // Fixed size N_ENTITIES (64)
    pub sub_state: SubState,
    pub self_deck: ThreeSegmentDeck,
    pub opp_deck: ThreeSegmentDeck,
    pub turn_count: u32,
    pub match_clock_remaining: f32, // 0.0 to 1.0
    pub series_score: [f32; 3],     // [self_wins, opp_wins, game_num]
    pub has_board_mutated: bool,
}

impl GameState {
    pub fn new_initial(catalog: &CardCatalog, self_deck_list: &[u16], opp_deck_list: &[u16]) -> Self {
        let mut entities = vec![None; N_ENTITIES];

        // Active Self: Charmander (id 10)
        let charmander_def = catalog.get(10).unwrap();
        entities[ACTIVE_SELF_SLOT] = Some(EntitySlot {
            card_id: 10,
            zone: Zone::ActiveSelf,
            pokemon: Some(PokemonEntity::new_basic(10, charmander_def.base_hp)),
        });

        // Hand Self: Ultra Ball (1), Basic Fire Energy (100), Charmeleon (11)
        entities[HAND_SELF_START] = Some(EntitySlot {
            card_id: 1, // Ultra Ball
            zone: Zone::HandSelf(0),
            pokemon: None,
        });
        entities[HAND_SELF_START + 1] = Some(EntitySlot {
            card_id: 100, // Fire energy
            zone: Zone::HandSelf(1),
            pokemon: None,
        });
        entities[HAND_SELF_START + 2] = Some(EntitySlot {
            card_id: 11, // Charmeleon
            zone: Zone::HandSelf(2),
            pokemon: None,
        });

        // Active Opp: Charmander (id 10)
        entities[ACTIVE_OPP_SLOT] = Some(EntitySlot {
            card_id: 10,
            zone: Zone::ActiveOpp,
            pokemon: Some(PokemonEntity::new_basic(10, charmander_def.base_hp)),
        });

        // Global Token
        entities[GLOBAL_TOKEN_SLOT] = Some(EntitySlot {
            card_id: 0,
            zone: Zone::GlobalToken,
            pokemon: None,
        });

        Self {
            entities,
            sub_state: SubState::IdleRoot,
            self_deck: ThreeSegmentDeck::from_deck_list(self_deck_list),
            opp_deck: ThreeSegmentDeck::from_deck_list(opp_deck_list),
            turn_count: 1,
            match_clock_remaining: 1.0,
            series_score: [0.0, 0.0, 1.0],
            has_board_mutated: true,
        }
    }

    /// Generates legality mask M_i in {0, -inf} of length N_ENTITIES (64)
    pub fn compute_legality_mask(&self, catalog: &CardCatalog) -> [f32; N_ENTITIES] {
        let mut mask = [-f32::INFINITY; N_ENTITIES];

        match &self.sub_state {
            SubState::IdleRoot => {
                // In root state, can interact with active, bench, or playable hand cards
                for i in 0..N_ENTITIES {
                    if let Some(slot) = &self.entities[i] {
                        match slot.zone {
                            Zone::ActiveSelf | Zone::BenchSelf(_) => {
                                mask[i] = 0.0;
                            }
                            Zone::HandSelf(_) => {
                                if let Some(card_def) = catalog.get(slot.card_id) {
                                    // Items, supporters, energy, basics are legal root actions
                                    if card_def.category == CardCategory::Item ||
                                       card_def.category == CardCategory::BasicEnergy ||
                                       card_def.category == CardCategory::BasicPokemon ||
                                       card_def.category == CardCategory::Supporter {
                                        mask[i] = 0.0;
                                    }
                                }
                            }
                            Zone::GlobalToken => {
                                // Pass action
                                mask[i] = 0.0;
                            }
                            _ => {}
                        }
                    }
                }
            }
            SubState::SelectDiscard1 { card_played_idx } => {
                // Hand cards excluding the card played get 0.0
                for i in HAND_SELF_START..=HAND_SELF_END {
                    if i != *card_played_idx && self.entities[i].is_some() {
                        mask[i] = 0.0;
                    }
                }
            }
            SubState::SelectDiscard2 { card_played_idx, discard_1_idx } => {
                // Hand cards excluding card played and discard 1 get 0.0
                for i in HAND_SELF_START..=HAND_SELF_END {
                    if i != *card_played_idx && i != *discard_1_idx && self.entities[i].is_some() {
                        mask[i] = 0.0;
                    }
                }
            }
            SubState::SelectSearchTarget { .. } => {
                // In search target selection, deck search slots or GlobalToken (Fail-to-find) get 0.0
                mask[GLOBAL_TOKEN_SLOT] = 0.0;
                // Hand slots that can receive card get 0.0
                for i in HAND_SELF_START..=HAND_SELF_END {
                    if self.entities[i].is_none() {
                        mask[i] = 0.0;
                        break;
                    }
                }
            }
            _ => {
                // Default: allow active & bench
                mask[ACTIVE_SELF_SLOT] = 0.0;
            }
        }

        mask
    }

    /// Executes an atomic micro-action step
    pub fn step_micro_action(
        &mut self,
        target_entity_idx: usize,
        catalog: &CardCatalog,
    ) -> bool {
        self.has_board_mutated = false;

        match self.sub_state.clone() {
            SubState::IdleRoot => {
                if let Some(slot) = &self.entities[target_entity_idx] {
                    if let Some(card_def) = catalog.get(slot.card_id) {
                        if card_def.name == "Ultra Ball" {
                            // Begin Ultra Ball micro-step sequence
                            self.sub_state = SubState::SelectDiscard1 {
                                card_played_idx: target_entity_idx,
                            };
                            return true;
                        }
                    }
                }
                false
            }
            SubState::SelectDiscard1 { card_played_idx } => {
                // Discard 1 selected
                self.sub_state = SubState::SelectDiscard2 {
                    card_played_idx,
                    discard_1_idx: target_entity_idx,
                };
                true
            }
            SubState::SelectDiscard2 { card_played_idx, discard_1_idx } => {
                // Discard 2 selected -> enter search phase
                self.sub_state = SubState::SelectSearchTarget {
                    card_played_idx,
                    filter: super::types::SearchFilter::AnyPokemon,
                };
                // Resolve discards
                let ub_card_id = self.entities[card_played_idx].take().map(|s| s.card_id).unwrap_or(0);
                let _d1_card_id = self.entities[discard_1_idx].take().map(|s| s.card_id).unwrap_or(0);
                let _d2_card_id = self.entities[target_entity_idx].take().map(|s| s.card_id).unwrap_or(0);

                // Add to discard slot
                self.entities[DISCARD_SELF_SLOT] = Some(EntitySlot {
                    card_id: ub_card_id,
                    zone: Zone::DiscardSelf,
                    pokemon: None,
                });
                self.has_board_mutated = true;
                true
            }
            SubState::SelectSearchTarget { .. } => {
                // Selected card from search (or fail-to-find)
                // Add Charizard ex (id 12) to hand if available in middle pool
                let target_card = 12; // Charizard ex
                for i in HAND_SELF_START..=HAND_SELF_END {
                    if self.entities[i].is_none() {
                        self.entities[i] = Some(EntitySlot {
                            card_id: target_card,
                            zone: Zone::HandSelf((i - HAND_SELF_START) as u8),
                            pokemon: None,
                        });
                        break;
                    }
                }

                // Shuffle deck
                self.self_deck.trigger_shuffle();
                self.sub_state = SubState::IdleRoot;
                self.has_board_mutated = true; // Invalidate cache!
                true
            }
            _ => false,
        }
    }

    /// Encodes all 64 entities into X in R^{64 x 256}
    pub fn encode_entities(&self, catalog: &CardCatalog) -> [[f32; RAW_FEATURE_DIM]; N_ENTITIES] {
        let mut x = [[0.0f32; RAW_FEATURE_DIM]; N_ENTITIES];

        for i in 0..N_ENTITIES {
            if let Some(slot) = &self.entities[i] {
                // e_static (128)
                if let Some(card_def) = catalog.get(slot.card_id) {
                    let e_static = card_def.encode_static();
                    x[i][0..128].copy_from_slice(&e_static);
                }

                // e_dynamic (96)
                if let Some(pkmn) = &slot.pokemon {
                    let e_dyn = pkmn.encode_dynamic();
                    x[i][128..224].copy_from_slice(&e_dyn);
                }

                // e_zone (32)
                let e_z = encode_zone(slot.zone);
                x[i][224..256].copy_from_slice(&e_z);
            }
        }

        x
    }
}
