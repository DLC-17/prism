"""Zone management, Pokemon entity instances, and 3-segment deck model in Python."""

import numpy as np
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .types import Zone, CARD_POOL_SIZE, E_DYNAMIC_DIM, E_ZONE_DIM


@dataclass
class EvolutionSlot:
    card_id: int
    base_hp: int


@dataclass
class PokemonEntity:
    card_id: int
    damage_counters: int = 0
    attached_energies: List[int] = field(default_factory=lambda: [0] * 10)
    attached_tool_id: int = 0
    status_conditions: int = 0
    turn_effects: List[bool] = field(default_factory=lambda: [False] * 6)
    evolution_stack: List[EvolutionSlot] = field(default_factory=list)
    evolution_depth: int = 0
    evolved_this_turn: bool = False

    @classmethod
    def new_basic(cls, card_id: int, base_hp: int) -> "PokemonEntity":
        entity = cls(card_id=card_id)
        entity.evolution_stack = [EvolutionSlot(card_id=card_id, base_hp=base_hp)]
        return entity

    def evolve(self, stage_card_id: int, stage_base_hp: int, is_rare_candy: bool = False):
        self.evolution_stack.append(
            EvolutionSlot(card_id=self.card_id, base_hp=stage_base_hp)
        )
        self.card_id = stage_card_id
        if is_rare_candy:
            self.evolution_depth = 2
        else:
            self.evolution_depth = min(3, self.evolution_depth + 1)
        self.evolved_this_turn = True

    def under_slot_1(self) -> Optional[EvolutionSlot]:
        if len(self.evolution_stack) >= 2:
            return self.evolution_stack[-1]
        return None

    def under_slot_2(self) -> Optional[EvolutionSlot]:
        if len(self.evolution_stack) >= 3:
            return self.evolution_stack[0]
        return None

    def encode_dynamic(self) -> np.ndarray:
        feats = np.zeros(E_DYNAMIC_DIM, dtype=np.float32)
        idx = 0

        # damage_counters (1)
        feats[idx] = self.damage_counters / 10.0
        idx += 1

        # attached_energy_histogram (11)
        total_energy = sum(self.attached_energies[:10])
        for k in range(10):
            feats[idx + k] = float(self.attached_energies[k])
        feats[idx + 10] = float(total_energy)
        idx += 11

        # evolution_stack_lifo (64)
        s1 = self.under_slot_1()
        if s1 is not None:
            d_indices = np.arange(24, dtype=np.float32)
            freq = s1.card_id / (1000.0 ** (2.0 * (d_indices // 2) / 24.0))
            feats[idx : idx + 24] = np.where(d_indices % 2 == 0, np.sin(freq), np.cos(freq))
        idx += 24

        # under_slot_1_hp (1)
        feats[idx] = (s1.base_hp / 340.0) if s1 is not None else 0.0
        idx += 1

        s2 = self.under_slot_2()
        if s2 is not None:
            d_indices = np.arange(24, dtype=np.float32)
            freq = s2.card_id / (1000.0 ** (2.0 * (d_indices // 2) / 24.0))
            feats[idx : idx + 24] = np.where(d_indices % 2 == 0, np.sin(freq), np.cos(freq))
        idx += 24

        feats[idx] = self.evolution_depth / 3.0
        idx += 1
        feats[idx] = 1.0 if self.evolved_this_turn else 0.0
        idx += 1

        # reserved padding for 64-dim LIFO stack
        idx += 13

        # attached_tool_id (16)
        if self.attached_tool_id > 0:
            d_indices = np.arange(16, dtype=np.float32)
            freq = self.attached_tool_id / (1000.0 ** (2.0 * (d_indices // 2) / 16.0))
            feats[idx : idx + 16] = np.where(d_indices % 2 == 0, np.sin(freq), np.cos(freq))
        idx += 16

        # status_conditions (4)
        for b in range(4):
            feats[idx + b] = 1.0 if (self.status_conditions & (1 << b)) != 0 else 0.0
        idx += 4

        return feats


def encode_zone(zone: Zone) -> np.ndarray:
    feats = np.zeros(E_ZONE_DIM, dtype=np.float32)
    idx = 0

    if zone.zone_type is not None:
        feats[idx + int(zone.zone_type)] = 1.0
    idx += 12

    feats[idx] = zone.controller
    idx += 1

    if 0 <= zone.slot_index < 6:
        feats[idx + zone.slot_index] = 1.0
    idx += 6

    return feats


class ThreeSegmentDeck:
    def __init__(self, deck_list: List[int]):
        self.top_stack: deque[int] = deque()
        self.middle_pool: Counter[int] = Counter(deck_list)
        self.bottom_queue: deque[int] = deque()
        self.total_count: int = len(deck_list)

    def trigger_shuffle(self):
        while self.top_stack:
            c = self.top_stack.popleft()
            self.middle_pool[c] += 1
        while self.bottom_queue:
            c = self.bottom_queue.popleft()
            self.middle_pool[c] += 1

    def trigger_ciphermaniac(self, card_a: int, card_b: int):
        for c in (card_a, card_b):
            if self.middle_pool[c] > 0:
                self.middle_pool[c] -= 1
        self.top_stack.appendleft(card_b)
        self.top_stack.appendleft(card_a)

    def trigger_iono_bottom_deck(self, bottom_cards: List[int]):
        for c in bottom_cards:
            if len(self.bottom_queue) >= 10:
                displaced = self.bottom_queue.popleft()
                self.middle_pool[displaced] += 1
            self.bottom_queue.append(c)
        self.total_count += len(bottom_cards)

    def get_middle_pool_distribution(self) -> np.ndarray:
        dist = np.zeros(CARD_POOL_SIZE, dtype=np.float32)
        total_pool_cards = sum(self.middle_pool.values())
        if total_pool_cards > 0:
            for card_id, count in self.middle_pool.items():
                if card_id < CARD_POOL_SIZE and count > 0:
                    dist[card_id] = count / total_pool_cards
        return dist

    @staticmethod
    def solve_exact_prizes(
        initial_60: List[int],
        visible_cards: List[int],
    ) -> List[int]:
        counts = Counter(initial_60)
        for c in visible_cards:
            if counts[c] > 0:
                counts[c] -= 1

        exact_prizes = []
        for c, cnt in counts.items():
            exact_prizes.extend([c] * cnt)
        return exact_prizes
