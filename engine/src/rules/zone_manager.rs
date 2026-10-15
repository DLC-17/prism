use super::types::{CARD_POOL_SIZE, E_DYNAMIC_DIM, E_ZONE_DIM, Zone};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSlot {
    pub card_id: u16,
    pub base_hp: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PokemonEntity {
    pub card_id: u16,
    pub damage_counters: u8,
    pub attached_energies: [u8; 10], // Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Colorless, Special
    pub attached_tool_id: u16,
    pub status_conditions: u8, // bitfield: 0: Asleep, 1: Burned, 2: Confused, 3: Paralyzed, 4: Poisoned
    pub turn_effects: [bool; 6], // cannot_attack, damage_reduction, retreat_lock, ability_locked, effect_immunity, attacked_this_turn
    pub evolution_stack: Vec<EvolutionSlot>, // LIFO stack of cards underneath
    pub evolution_depth: u8, // 0: Basic, 1: Stage 1, 2: Stage 2 via Candy, 3: Stage 2 Natural
    pub evolved_this_turn: bool,
}

impl PokemonEntity {
    pub fn new_basic(card_id: u16, base_hp: u16) -> Self {
        Self {
            card_id,
            damage_counters: 0,
            attached_energies: [0; 10],
            attached_tool_id: 0,
            status_conditions: 0,
            turn_effects: [false; 6],
            evolution_stack: vec![EvolutionSlot { card_id, base_hp }],
            evolution_depth: 0,
            evolved_this_turn: false,
        }
    }

    pub fn evolve(&mut self, stage_card_id: u16, stage_base_hp: u16, is_rare_candy: bool) {
        self.evolution_stack.push(EvolutionSlot {
            card_id: self.card_id,
            base_hp: stage_base_hp, // previous card's HP is preserved in stack
        });
        self.card_id = stage_card_id;
        self.evolution_depth = if is_rare_candy {
            2
        } else {
            (self.evolution_depth + 1).min(3)
        };
        self.evolved_this_turn = true;
    }

    pub fn under_slot_1(&self) -> Option<&EvolutionSlot> {
        if self.evolution_stack.len() >= 2 {
            self.evolution_stack.get(self.evolution_stack.len() - 1)
        } else {
            None
        }
    }

    pub fn under_slot_2(&self) -> Option<&EvolutionSlot> {
        if self.evolution_stack.len() >= 3 {
            self.evolution_stack.get(0)
        } else {
            None
        }
    }

    /// Encodes e_dynamic into exactly 96 floats:
    /// damage_counters (1) + attached_energy_histogram (11) + evolution_stack_lifo (64) +
    /// attached_tool_id (16) + status_conditions (4) + turn_effects (6) = 1 + 11 + 64 + 16 + 4 = 96
    pub fn encode_dynamic(&self) -> [f32; E_DYNAMIC_DIM] {
        let mut feats = [0.0f32; E_DYNAMIC_DIM];
        let mut idx = 0;

        // damage_counters (normalized HP_dmg / 10) (1)
        feats[idx] = self.damage_counters as f32 / 10.0;
        idx += 1;

        // attached_energy_histogram (10 discrete types + 1 total) = 11
        let mut total_energy = 0;
        for k in 0..10 {
            feats[idx + k] = self.attached_energies[k] as f32;
            total_energy += self.attached_energies[k];
        }
        feats[idx + 10] = total_energy as f32;
        idx += 11;

        // evolution_stack_lifo (64)
        // under_slot_1_id (24)
        if let Some(s1) = self.under_slot_1() {
            for d in 0..24 {
                let freq = (s1.card_id as f32) / (1000.0f32.powf((2 * (d / 2)) as f32 / 24.0));
                feats[idx + d] = if d % 2 == 0 { freq.sin() } else { freq.cos() };
            }
        }
        idx += 24;

        // under_slot_1_hp (1)
        feats[idx] = if let Some(s1) = self.under_slot_1() {
            s1.base_hp as f32 / 340.0
        } else {
            0.0
        };
        idx += 1;

        // under_slot_2_id (24)
        if let Some(s2) = self.under_slot_2() {
            for d in 0..24 {
                let freq = (s2.card_id as f32) / (1000.0f32.powf((2 * (d / 2)) as f32 / 24.0));
                feats[idx + d] = if d % 2 == 0 { freq.sin() } else { freq.cos() };
            }
        }
        idx += 24;

        // evolution_depth (1)
        feats[idx] = self.evolution_depth as f32 / 3.0;
        idx += 1;

        // evolved_this_turn (1)
        feats[idx] = if self.evolved_this_turn { 1.0 } else { 0.0 };
        idx += 1;

        // reserved padding to complete 64-dim LIFO representation (13)
        idx += 13; // (24 + 1 + 24 + 1 + 1 + 13 = 64)

        // attached_tool_id (16)
        if self.attached_tool_id > 0 {
            for d in 0..16 {
                let freq = (self.attached_tool_id as f32) / (1000.0f32.powf((2 * (d / 2)) as f32 / 16.0));
                feats[idx + d] = if d % 2 == 0 { freq.sin() } else { freq.cos() };
            }
        }
        idx += 16;

        // status_conditions (4): Asleep, Burned, Confused, Paralyzed/Poisoned
        for b in 0..4 {
            feats[idx + b] = if (self.status_conditions & (1 << b)) != 0 { 1.0 } else { 0.0 };
        }

        feats
    }
}

/// Zone embedding encoder into exactly 32 floats:
/// zone_id one-hot (12) + controller (1) + slot_index one-hot/normalized (6) + auxiliary/padding (13) = 32
pub fn encode_zone(zone: Zone) -> [f32; E_ZONE_DIM] {
    let mut feats = [0.0f32; E_ZONE_DIM];
    let mut idx = 0;

    // zone_id one-hot (12)
    if let Some(zt) = zone.zone_type() {
        feats[idx + zt as usize] = 1.0;
    }
    idx += 12;

    // controller (1): +1 Self, -1 Opponent
    feats[idx] = zone.controller();
    idx += 1;

    // slot_index (6): one-hot for index 0..5
    let slot = zone.slot_index();
    if (slot as usize) < 6 {
        feats[idx + slot as usize] = 1.0;
    }

    // auxiliary / padding (13)
    // reserved for zone properties (e.g. stadium active, prize revealed)
    feats
}

/// 3-Segment Deck structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeSegmentDeck {
    pub top_stack: VecDeque<u16>,            // Segment 1: L_top <= 3 discrete card IDs
    pub middle_pool: HashMap<u16, u8>,       // Segment 2: Hypergeometric count multiset
    pub bottom_queue: VecDeque<u16>,         // Segment 3: L_bot <= 10 ordered cards
    pub total_count: usize,
}

impl ThreeSegmentDeck {
    pub fn from_deck_list(deck: &[u16]) -> Self {
        let mut pool = HashMap::new();
        for &c in deck {
            *pool.entry(c).or_insert(0) += 1;
        }
        Self {
            top_stack: VecDeque::new(),
            middle_pool: pool,
            bottom_queue: VecDeque::new(),
            total_count: deck.len(),
        }
    }

    /// EVENT_SHUFFLE: Merges Segment 1 and Segment 3 into Segment 2.
    /// Resets all positional certitude.
    pub fn trigger_shuffle(&mut self) {
        while let Some(c) = self.top_stack.pop_front() {
            *self.middle_pool.entry(c).or_insert(0) += 1;
        }
        while let Some(c) = self.bottom_queue.pop_front() {
            *self.middle_pool.entry(c).or_insert(0) += 1;
        }
    }

    /// EVENT_CIPHERMANIAC: Inserts 2 chosen card IDs into Segment 1 at index 0 and 1.
    pub fn trigger_ciphermaniac(&mut self, card_a: u16, card_b: u16) {
        // Remove from middle pool if present
        for &c in &[card_a, card_b] {
            if let Some(cnt) = self.middle_pool.get_mut(&c) {
                if *cnt > 0 {
                    *cnt -= 1;
                }
            }
        }
        self.top_stack.push_front(card_b);
        self.top_stack.push_front(card_a);
    }

    /// EVENT_IONO: Opponent hand cards moved to Segment 3 in randomized/ordered queue;
    /// k cards drawn from Segment 1 / Segment 2.
    pub fn trigger_iono_bottom_deck(&mut self, bottom_cards: &[u16]) {
        for &c in bottom_cards {
            if self.bottom_queue.len() >= 10 {
                let displaced = self.bottom_queue.pop_front().unwrap();
                *self.middle_pool.entry(displaced).or_insert(0) += 1;
            }
            self.bottom_queue.push_back(c);
        }
        self.total_count += bottom_cards.len();
    }

    /// Computes hypergeometric count probability distribution for Segment 2
    pub fn get_middle_pool_distribution(&self) -> [f32; CARD_POOL_SIZE] {
        let mut dist = [0.0f32; CARD_POOL_SIZE];
        let total_pool_cards: u32 = self.middle_pool.values().map(|&v| v as u32).sum();
        if total_pool_cards > 0 {
            for (&card_id, &count) in &self.middle_pool {
                if (card_id as usize) < CARD_POOL_SIZE {
                    dist[card_id as usize] = count as f32 / total_pool_cards as f32;
                }
            }
        }
        dist
    }

    /// Deterministic Deck-Search Resolution:
    /// Prize_exact = InitialDeck_60 - (VisibleBoard + Hand + Discard + RemainingDeck)
    pub fn solve_exact_prizes(
        initial_60: &[u16],
        visible_cards: &[u16],
    ) -> Vec<u16> {
        let mut counts = HashMap::new();
        for &c in initial_60 {
            *counts.entry(c).or_insert(0i16) += 1;
        }
        for &c in visible_cards {
            if let Some(cnt) = counts.get_mut(&c) {
                *cnt -= 1;
            }
        }

        let mut exact_prizes = Vec::new();
        for (&c, &cnt) in &counts {
            for _ in 0..cnt.max(0) {
                exact_prizes.push(c);
            }
        }
        exact_prizes
    }
}
