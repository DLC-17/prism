"""Synchronous micro-step atomic state machine and entity encoding in Python."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .types import (
    Zone,
    ZoneType,
    SubState,
    CardCategory,
    N_ENTITIES,
    RAW_FEATURE_DIM,
)
from .card_db import CardCatalog
from .zone_manager import PokemonEntity, ThreeSegmentDeck, encode_zone

ACTIVE_SELF_SLOT: int = 0
BENCH_SELF_START: int = 1
BENCH_SELF_END: int = 5
STADIUM_SLOT: int = 6
HAND_SELF_START: int = 7
HAND_SELF_END: int = 26
PRIZE_SELF_START: int = 27
PRIZE_SELF_END: int = 32
DISCARD_SELF_SLOT: int = 33
LOST_SELF_SLOT: int = 34

ACTIVE_OPP_SLOT: int = 35
BENCH_OPP_START: int = 36
BENCH_OPP_END: int = 40
HAND_OPP_START: int = 41
HAND_OPP_END: int = 46
DISCARD_OPP_SLOT: int = 47
LOST_OPP_SLOT: int = 48
GLOBAL_TOKEN_SLOT: int = 49


@dataclass
class EntitySlot:
    card_id: int
    zone: Zone
    pokemon: Optional[PokemonEntity] = None


class GameState:
    def __init__(
        self,
        catalog: CardCatalog,
        self_deck_list: List[int],
        opp_deck_list: List[int],
    ):
        self.catalog = catalog
        self.entities: List[Optional[EntitySlot]] = [None] * N_ENTITIES
        self.sub_state: SubState = SubState.IDLE_ROOT
        self.self_deck = ThreeSegmentDeck(self_deck_list)
        self.opp_deck = ThreeSegmentDeck(opp_deck_list)
        self.turn_count: int = 1
        self.match_clock_remaining: float = 1.0
        self.series_score: List[float] = [0.0, 0.0, 1.0]
        self.has_board_mutated: bool = True
        self._setup_initial_board()

    def _setup_initial_board(self):
        # Active Self: Charmander (id 10)
        charmander_def = self.catalog.get(10)
        assert charmander_def is not None
        self.entities[ACTIVE_SELF_SLOT] = EntitySlot(
            card_id=10,
            zone=Zone(ZoneType.ACTIVE_SELF),
            pokemon=PokemonEntity.new_basic(10, charmander_def.base_hp),
        )

        # Hand Self: Ultra Ball (1), Fire Energy (100), Charmeleon (11)
        self.entities[HAND_SELF_START] = EntitySlot(
            card_id=1,
            zone=Zone(ZoneType.HAND_SELF, slot_index=0),
        )
        self.entities[HAND_SELF_START + 1] = EntitySlot(
            card_id=100,
            zone=Zone(ZoneType.HAND_SELF, slot_index=1),
        )
        self.entities[HAND_SELF_START + 2] = EntitySlot(
            card_id=11,
            zone=Zone(ZoneType.HAND_SELF, slot_index=2),
        )

        # Active Opp: Charmander (id 10)
        self.entities[ACTIVE_OPP_SLOT] = EntitySlot(
            card_id=10,
            zone=Zone(ZoneType.ACTIVE_OPP, is_self=False),
            pokemon=PokemonEntity.new_basic(10, charmander_def.base_hp),
        )

        # Global Token
        self.entities[GLOBAL_TOKEN_SLOT] = EntitySlot(
            card_id=0,
            zone=Zone(None),
        )

    def compute_legality_mask(self) -> np.ndarray:
        mask = np.full(N_ENTITIES, -np.inf, dtype=np.float32)

        if self.sub_state == SubState.IDLE_ROOT:
            for i in range(N_ENTITIES):
                slot = self.entities[i]
                if slot is not None:
                    if slot.zone.zone_type in (
                        ZoneType.ACTIVE_SELF,
                        ZoneType.BENCH_SELF,
                    ):
                        mask[i] = 0.0
                    elif slot.zone.zone_type == ZoneType.HAND_SELF:
                        card_def = self.catalog.get(slot.card_id)
                        if card_def is not None and card_def.category in (
                            CardCategory.ITEM,
                            CardCategory.BASIC_ENERGY,
                            CardCategory.BASIC_POKEMON,
                            CardCategory.SUPPORTER,
                        ):
                            mask[i] = 0.0
                    elif slot.zone.zone_type is None and i == GLOBAL_TOKEN_SLOT:
                        mask[i] = 0.0

        elif self.sub_state == SubState.SELECT_DISCARD_1:
            # Hand cards excluding played card
            for i in range(HAND_SELF_START, HAND_SELF_END + 1):
                if i != self._card_played_idx and self.entities[i] is not None:
                    mask[i] = 0.0

        elif self.sub_state == SubState.SELECT_DISCARD_2:
            # Hand cards excluding played card and discard 1
            for i in range(HAND_SELF_START, HAND_SELF_END + 1):
                if (
                    i != self._card_played_idx
                    and i != self._discard_1_idx
                    and self.entities[i] is not None
                ):
                    mask[i] = 0.0

        elif self.sub_state == SubState.SELECT_SEARCH_TARGET:
            mask[GLOBAL_TOKEN_SLOT] = 0.0
            for i in range(HAND_SELF_START, HAND_SELF_END + 1):
                if self.entities[i] is None:
                    mask[i] = 0.0
                    break

        return mask

    def step_micro_action(self, target_entity_idx: int) -> bool:
        self.has_board_mutated = False

        if self.sub_state == SubState.IDLE_ROOT:
            slot = self.entities[target_entity_idx]
            if slot is not None:
                card_def = self.catalog.get(slot.card_id)
                if card_def is not None and card_def.name == "Ultra Ball":
                    self._card_played_idx = target_entity_idx
                    self.sub_state = SubState.SELECT_DISCARD_1
                    # Intra-card sub-step: cache is preserved!
                    return True

        elif self.sub_state == SubState.SELECT_DISCARD_1:
            self._discard_1_idx = target_entity_idx
            self.sub_state = SubState.SELECT_DISCARD_2
            # Intra-card sub-step: cache is preserved!
            return True

        elif self.sub_state == SubState.SELECT_DISCARD_2:
            self._discard_2_idx = target_entity_idx
            self.sub_state = SubState.SELECT_SEARCH_TARGET
            # Intra-card sub-step: cache is preserved!
            return True

        elif self.sub_state == SubState.SELECT_SEARCH_TARGET:
            # Resolve Ultra Ball effect: move discards and target card, trigger shuffle
            self.entities[self._card_played_idx] = None
            self.entities[self._discard_1_idx] = None
            self.entities[self._discard_2_idx] = None

            # Add searched Charizard ex (id 12) to hand
            for i in range(HAND_SELF_START, HAND_SELF_END + 1):
                if self.entities[i] is None:
                    self.entities[i] = EntitySlot(
                        card_id=12,
                        zone=Zone(ZoneType.HAND_SELF, slot_index=i - HAND_SELF_START),
                    )
                    break

            self.self_deck.trigger_shuffle()
            self.sub_state = SubState.IDLE_ROOT
            self.has_board_mutated = True  # Physical board mutated -> cache invalidation!
            return True

        return False

    def encode_entities(self) -> np.ndarray:
        x = np.zeros((N_ENTITIES, RAW_FEATURE_DIM), dtype=np.float32)
        for i in range(N_ENTITIES):
            slot = self.entities[i]
            if slot is not None:
                card_def = self.catalog.get(slot.card_id)
                if card_def is not None:
                    x[i, :128] = card_def.encode_static()
                if slot.pokemon is not None:
                    x[i, 128:224] = slot.pokemon.encode_dynamic()
                x[i, 224:256] = encode_zone(slot.zone)
        return x
