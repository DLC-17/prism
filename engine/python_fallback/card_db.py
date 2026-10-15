"""Card catalog and feature encoding for the pure-Python simulation mirror."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .types import CardCategory, EnergyType, E_STATIC_DIM


@dataclass
class AttackDef:
    name: str
    energy_cost: List[int]  # 10 discrete elements
    base_damage: int
    effect_flag_id: int = 0


@dataclass
class CardDef:
    card_id: int
    name: str
    category: CardCategory
    stage: int  # 0: Basic, 1: Stage 1, 2: Stage 2, 3: VSTAR, 4: ex
    base_hp: int
    primary_type: Optional[EnergyType] = None
    weakness: Optional[EnergyType] = None
    resistance: Optional[EnergyType] = None
    retreat_cost: int = 0
    attacks: List[AttackDef] = field(default_factory=list)
    ability_category: int = 0
    evolves_from_id: int = 0

    def encode_rule_features(self) -> np.ndarray:
        feats = np.zeros(64, dtype=np.float32)
        idx = 0

        # Stage one-hot (5)
        if self.stage < 5:
            feats[idx + self.stage] = 1.0
        idx += 5

        # Base HP normalized (1)
        feats[idx] = self.base_hp / 340.0
        idx += 1

        # Primary type one-hot (10)
        if self.primary_type is not None:
            feats[idx + int(self.primary_type)] = 1.0
        idx += 10

        # Weakness (10 one-hot + 1 modifier) (11)
        if self.weakness is not None:
            feats[idx + int(self.weakness)] = 1.0
            feats[idx + 10] = 1.0  # 2x weakness
        idx += 11

        # Resistance (10 one-hot + 1 modifier) (11)
        if self.resistance is not None:
            feats[idx + int(self.resistance)] = 1.0
            feats[idx + 10] = 1.0  # -30 resistance
        idx += 11

        # Retreat cost normalized (1)
        feats[idx] = self.retreat_cost / 4.0
        idx += 1

        # Attack 1 (11)
        if len(self.attacks) > 0:
            for k in range(min(10, len(self.attacks[0].energy_cost))):
                feats[idx + k] = self.attacks[0].energy_cost[k] / 5.0
            feats[idx + 10] = self.attacks[0].base_damage / 300.0
        idx += 11

        # Attack 2 (11)
        if len(self.attacks) > 1:
            for k in range(min(10, len(self.attacks[1].energy_cost))):
                feats[idx + k] = self.attacks[1].energy_cost[k] / 5.0
            feats[idx + 10] = self.attacks[1].base_damage / 300.0
        idx += 11

        # Ability category & flag (2)
        feats[idx] = 1.0 if self.ability_category > 0 else 0.0
        feats[idx + 1] = self.ability_category / 5.0
        idx += 2

        # Card category normalized (1)
        feats[idx] = int(self.category) / 10.0

        return feats

    def encode_static(self) -> np.ndarray:
        out = np.zeros(E_STATIC_DIM, dtype=np.float32)
        # Sinusoidal projection for raw card_id into 64 dimensions
        d_indices = np.arange(64, dtype=np.float32)
        freq = self.card_id / (10000.0 ** (2.0 * (d_indices // 2) / 64.0))
        out[:64] = np.where(d_indices % 2 == 0, np.sin(freq), np.cos(freq))
        out[64:128] = self.encode_rule_features()
        return out


class CardCatalog:
    def __init__(self):
        self.cards: Dict[int, CardDef] = {}
        self._register_reference_cards()

    def get(self, card_id: int) -> Optional[CardDef]:
        return self.cards.get(card_id)

    def insert(self, card: CardDef):
        self.cards[card.card_id] = card

    def _register_reference_cards(self):
        self.insert(
            CardDef(
                card_id=1,
                name="Ultra Ball",
                category=CardCategory.ITEM,
                stage=0,
                base_hp=0,
            )
        )
        self.insert(
            CardDef(
                card_id=10,
                name="Charmander",
                category=CardCategory.BASIC_POKEMON,
                stage=0,
                base_hp=70,
                primary_type=EnergyType.FIRE,
                weakness=EnergyType.WATER,
                retreat_cost=1,
                attacks=[
                    AttackDef(
                        name="Scratch",
                        energy_cost=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        base_damage=10,
                    )
                ],
            )
        )
        self.insert(
            CardDef(
                card_id=11,
                name="Charmeleon",
                category=CardCategory.STAGE_1_POKEMON,
                stage=1,
                base_hp=90,
                primary_type=EnergyType.FIRE,
                weakness=EnergyType.WATER,
                retreat_cost=2,
                attacks=[
                    AttackDef(
                        name="Combustion",
                        energy_cost=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                        base_damage=50,
                    )
                ],
                evolves_from_id=10,
            )
        )
        self.insert(
            CardDef(
                card_id=12,
                name="Charizard ex",
                category=CardCategory.EX_POKEMON,
                stage=4,
                base_hp=330,
                primary_type=EnergyType.DARKNESS,
                weakness=EnergyType.GRASS,
                retreat_cost=2,
                attacks=[
                    AttackDef(
                        name="Burning Darkness",
                        energy_cost=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                        base_damage=180,
                    )
                ],
                ability_category=2,
                evolves_from_id=11,
            )
        )
        self.insert(
            CardDef(
                card_id=100,
                name="Basic Fire Energy",
                category=CardCategory.BASIC_ENERGY,
                stage=0,
                base_hp=0,
                primary_type=EnergyType.FIRE,
            )
        )
