use super::types::{CardCategory, EnergyType, E_STATIC_DIM};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackDef {
    pub name: String,
    pub energy_cost: [u8; 10], // Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Colorless, Special
    pub base_damage: u16,
    pub effect_flag_id: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardDef {
    pub card_id: u16,
    pub name: String,
    pub category: CardCategory,
    pub stage: u8, // 0: Basic, 1: Stage 1, 2: Stage 2, 3: VSTAR, 4: ex
    pub base_hp: u16,
    pub primary_type: Option<EnergyType>,
    pub weakness: Option<EnergyType>,
    pub resistance: Option<EnergyType>,
    pub retreat_cost: u8,
    pub attacks: Vec<AttackDef>,
    pub ability_category: u8, // 0: None, 1: Draw, 2: EnergyAttach, 3: DamageMod, 4: Search, 5: Lock
    pub evolves_from_id: u16, // 0 if Basic
}

impl CardDef {
    /// Encodes rule_features into exactly 64 floats:
    /// Stage (5) + HP (1) + Type (10) + Weakness (11) + Resistance (11) + Retreat (1) +
    /// Attack1 (11) + Attack2 (11) + Ability (2) + Category (1) = 64 floats
    pub fn encode_rule_features(&self) -> [f32; 64] {
        let mut feats = [0.0f32; 64];
        let mut idx = 0;

        // Stage one-hot (5)
        if self.stage < 5 {
            feats[idx + self.stage as usize] = 1.0;
        }
        idx += 5;

        // Base HP normalized (1)
        feats[idx] = self.base_hp as f32 / 340.0;
        idx += 1;

        // Primary type one-hot (10)
        if let Some(t) = self.primary_type {
            feats[idx + t as usize] = 1.0;
        }
        idx += 10;

        // Weakness (10 one-hot + 1 modifier) (11)
        if let Some(w) = self.weakness {
            feats[idx + w as usize] = 1.0;
            feats[idx + 10] = 2.0 / 2.0; // 2x weakness
        }
        idx += 11;

        // Resistance (10 one-hot + 1 modifier) (11)
        if let Some(r) = self.resistance {
            feats[idx + r as usize] = 1.0;
            feats[idx + 10] = 30.0 / 30.0; // -30 resistance
        }
        idx += 11;

        // Retreat cost normalized (1)
        feats[idx] = self.retreat_cost as f32 / 4.0;
        idx += 1;

        // Attack 1: energy requirement vector (10) + base damage / 300 (1) = 11
        if let Some(atk1) = self.attacks.get(0) {
            for k in 0..10 {
                feats[idx + k] = atk1.energy_cost[k] as f32 / 5.0;
            }
            feats[idx + 10] = atk1.base_damage as f32 / 300.0;
        }
        idx += 11;

        // Attack 2: energy requirement vector (10) + base damage / 300 (1) = 11
        if let Some(atk2) = self.attacks.get(1) {
            for k in 0..10 {
                feats[idx + k] = atk2.energy_cost[k] as f32 / 5.0;
            }
            feats[idx + 10] = atk2.base_damage as f32 / 300.0;
        }
        idx += 11;

        // Ability category & flag (2)
        feats[idx] = if self.ability_category > 0 { 1.0 } else { 0.0 };
        feats[idx + 1] = self.ability_category as f32 / 5.0;
        idx += 2;

        // Card category normalized (1)
        feats[idx] = self.category as u8 as f32 / 10.0;

        feats
    }

    /// Full static embedding representation: 64 card ID embedding components + 64 rule features = 128
    pub fn encode_static(&self) -> [f32; E_STATIC_DIM] {
        let mut out = [0.0f32; E_STATIC_DIM];
        // Hash/sinusoidal projection for raw card_id into first 64 dimensions
        // (In Python PyTorch, this is further fine-tuned via nn.Embedding(CardPoolSize, 64))
        for d in 0..64 {
            let freq = (self.card_id as f32) / (10000.0f32.powf((2 * (d / 2)) as f32 / 64.0));
            out[d] = if d % 2 == 0 { freq.sin() } else { freq.cos() };
        }

        let rule_feats = self.encode_rule_features();
        out[64..128].copy_from_slice(&rule_feats);
        out
    }
}

pub struct CardCatalog {
    cards: HashMap<u16, CardDef>,
}

impl CardCatalog {
    pub fn new_standard_catalog() -> Self {
        let mut catalog = Self {
            cards: HashMap::new(),
        };
        catalog.register_reference_cards();
        catalog
    }

    pub fn get(&self, card_id: u16) -> Option<&CardDef> {
        self.cards.get(&card_id)
    }

    pub fn insert(&mut self, card: CardDef) {
        self.cards.insert(card.card_id, card);
    }

    fn register_reference_cards(&mut self) {
        // Ultra Ball
        self.insert(CardDef {
            card_id: 1,
            name: "Ultra Ball".to_string(),
            category: CardCategory::Item,
            stage: 0,
            base_hp: 0,
            primary_type: None,
            weakness: None,
            resistance: None,
            retreat_cost: 0,
            attacks: vec![],
            ability_category: 0,
            evolves_from_id: 0,
        });

        // Charmander (Root basic)
        self.insert(CardDef {
            card_id: 10,
            name: "Charmander".to_string(),
            category: CardCategory::BasicPokemon,
            stage: 0,
            base_hp: 70,
            primary_type: Some(EnergyType::Fire),
            weakness: Some(EnergyType::Water),
            resistance: None,
            retreat_cost: 1,
            attacks: vec![AttackDef {
                name: "Scratch".to_string(),
                energy_cost: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                base_damage: 10,
                effect_flag_id: 0,
            }],
            ability_category: 0,
            evolves_from_id: 0,
        });

        // Charmeleon (Stage 1)
        self.insert(CardDef {
            card_id: 11,
            name: "Charmeleon".to_string(),
            category: CardCategory::Stage1Pokemon,
            stage: 1,
            base_hp: 90,
            primary_type: Some(EnergyType::Fire),
            weakness: Some(EnergyType::Water),
            resistance: None,
            retreat_cost: 2,
            attacks: vec![AttackDef {
                name: "Combustion".to_string(),
                energy_cost: [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                base_damage: 50,
                effect_flag_id: 0,
            }],
            ability_category: 0,
            evolves_from_id: 10,
        });

        // Charizard ex (Stage 2 ex)
        self.insert(CardDef {
            card_id: 12,
            name: "Charizard ex".to_string(),
            category: CardCategory::ExPokemon,
            stage: 4,
            base_hp: 330,
            primary_type: Some(EnergyType::Darkness),
            weakness: Some(EnergyType::Grass),
            resistance: None,
            retreat_cost: 2,
            attacks: vec![AttackDef {
                name: "Burning Darkness".to_string(),
                energy_cost: [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                base_damage: 180,
                effect_flag_id: 1,
            }],
            ability_category: 2, // Infernal Reign: attaches basic Fire energies
            evolves_from_id: 11,
        });

        // Basic Fire Energy
        self.insert(CardDef {
            card_id: 100,
            name: "Basic Fire Energy".to_string(),
            category: CardCategory::BasicEnergy,
            stage: 0,
            base_hp: 0,
            primary_type: Some(EnergyType::Fire),
            weakness: None,
            resistance: None,
            retreat_cost: 0,
            attacks: vec![],
            ability_category: 0,
            evolves_from_id: 0,
        });
    }
}
