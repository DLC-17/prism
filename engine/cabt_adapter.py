"""Adapter mapping Kaggle PTCG cabt simulator Observations to Prism tensors."""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from engine.python_fallback.types import (
    N_ENTITIES,
    RAW_FEATURE_DIM,
    ZoneType,
    Zone,
    EnergyType,
    CardCategory,
)
from engine.python_fallback.card_db import CardCatalog, CardDef, AttackDef
from engine.python_fallback.zone_manager import PokemonEntity, encode_zone
from engine.python_fallback.state_machine import (
    ACTIVE_SELF_SLOT,
    BENCH_SELF_START,
    HAND_SELF_START,
    ACTIVE_OPP_SLOT,
    BENCH_OPP_START,
    DISCARD_SELF_SLOT,
    DISCARD_OPP_SLOT,
    STADIUM_SLOT,
    GLOBAL_TOKEN_SLOT,
)


class CabtObservationAdapter:
    """Translates PTCGABC cabt simulator observations into Prism's

    Composite Entity representation X in R^{1 x 64 x 256} and legality mask M_i in R^{1 x 64}.
    """

    def __init__(self, catalog: Optional[CardCatalog] = None):
        self.catalog = catalog or CardCatalog()

    def observation_to_tensors(
        self,
        obs_dict: Dict[str, Any],
        your_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maps a cabt observation dict into (raw_features, legality_mask, z_match).

        Args:
            obs_dict: Dict containing 'current', 'select', 'logs'
            your_index: Index of current agent (0 or 1)

        Returns:
            raw_features: Tensor of shape (1, 64, 256)
            legality_mask: Tensor of shape (1, 64) in {0, -inf}
            z_match: Tensor of shape (1, 64)
        """
        raw_x = np.zeros((N_ENTITIES, RAW_FEATURE_DIM), dtype=np.float32)
        mask = np.full(N_ENTITIES, -np.inf, dtype=np.float32)
        z_match = np.zeros(64, dtype=np.float32)

        current = obs_dict.get("current")
        if current is None:
            # Setup phase: pass through default
            mask[GLOBAL_TOKEN_SLOT] = 0.0
            return (
                torch.from_numpy(raw_x).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
                torch.from_numpy(z_match).unsqueeze(0),
            )

        players = current.get("players", [])
        self_player = players[your_index] if len(players) > your_index else {}
        opp_index = 1 - your_index
        opp_player = players[opp_index] if len(players) > opp_index else {}

        # 1. Encode Self Active
        self_active = self_player.get("active", [])
        if self_active and self_active[0] is not None:
            pkmn_data = self_active[0]
            card_id = pkmn_data.get("id", 0)
            self._encode_pokemon_slot(
                slot_idx=ACTIVE_SELF_SLOT,
                card_id=card_id,
                pkmn_data=pkmn_data,
                zone=Zone(ZoneType.ACTIVE_SELF),
                target_array=raw_x,
            )

        # 2. Encode Self Bench (0..4)
        self_bench = self_player.get("bench", [])
        for i, pkmn_data in enumerate(self_bench[:5]):
            if pkmn_data is not None:
                card_id = pkmn_data.get("id", 0)
                self._encode_pokemon_slot(
                    slot_idx=BENCH_SELF_START + i,
                    card_id=card_id,
                    pkmn_data=pkmn_data,
                    zone=Zone(ZoneType.BENCH_SELF, slot_index=i),
                    target_array=raw_x,
                )

        # 3. Encode Self Hand (up to 20)
        self_hand = self_player.get("hand", [])
        if self_hand:
            for i, card_data in enumerate(self_hand[:20]):
                card_id = card_data.get("id", 0)
                slot_idx = HAND_SELF_START + i
                self._encode_card_slot(
                    slot_idx=slot_idx,
                    card_id=card_id,
                    zone=Zone(ZoneType.HAND_SELF, slot_index=i),
                    target_array=raw_x,
                )

        # 4. Encode Opponent Active
        opp_active = opp_player.get("active", [])
        if opp_active and opp_active[0] is not None:
            pkmn_data = opp_active[0]
            card_id = pkmn_data.get("id", 0)
            self._encode_pokemon_slot(
                slot_idx=ACTIVE_OPP_SLOT,
                card_id=card_id,
                pkmn_data=pkmn_data,
                zone=Zone(ZoneType.ACTIVE_OPP, is_self=False),
                target_array=raw_x,
            )

        # 5. Encode Opponent Bench (0..4)
        opp_bench = opp_player.get("bench", [])
        for i, pkmn_data in enumerate(opp_bench[:5]):
            if pkmn_data is not None:
                card_id = pkmn_data.get("id", 0)
                self._encode_pokemon_slot(
                    slot_idx=BENCH_OPP_START + i,
                    card_id=card_id,
                    pkmn_data=pkmn_data,
                    zone=Zone(ZoneType.BENCH_OPP, slot_index=i, is_self=False),
                    target_array=raw_x,
                )

        # 6. Global Token
        raw_x[GLOBAL_TOKEN_SLOT, 224:256] = encode_zone(Zone(None))

        # 7. Map Legality Mask from cabt 'select.option'
        select_data = obs_dict.get("select")
        if select_data and "option" in select_data:
            options = select_data["option"]
            for opt in options:
                opt_type = opt.get("type")
                # Option maps to target entities
                area = opt.get("area")
                index = opt.get("index", 0)

                if area == "ACTIVE":
                    mask[ACTIVE_SELF_SLOT] = 0.0
                elif area == "BENCH" and index < 5:
                    mask[BENCH_SELF_START + index] = 0.0
                elif area == "HAND" and index < 20:
                    mask[HAND_SELF_START + index] = 0.0
                elif opt_type in ("END", "NO", "PASS"):
                    mask[GLOBAL_TOKEN_SLOT] = 0.0
        else:
            mask[GLOBAL_TOKEN_SLOT] = 0.0

        # Match Context z_match: turn count, prizes, first player
        turn_count = current.get("turn", 1)
        z_match[35] = 1.0  # clock placeholder
        z_match[36] = (6 - len(self_player.get("prize", []))) / 6.0
        z_match[37] = (6 - len(opp_player.get("prize", []))) / 6.0
        z_match[38] = turn_count / 30.0
        z_match[39] = 1.0 if current.get("firstPlayer") == your_index else 0.0

        return (
            torch.from_numpy(raw_x).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
            torch.from_numpy(z_match).unsqueeze(0),
        )

    def _encode_pokemon_slot(
        self,
        slot_idx: int,
        card_id: int,
        pkmn_data: Dict[str, Any],
        zone: Zone,
        target_array: np.ndarray,
    ):
        card_def = self.catalog.get(card_id)
        if card_def:
            target_array[slot_idx, :128] = card_def.encode_static()

        # Dynamic features: damage, energy, evolution stack, tools
        hp = pkmn_data.get("hp", 0)
        max_hp = pkmn_data.get("maxHp", hp)
        damage_counters = max(0, max_hp - hp) // 10

        energies = pkmn_data.get("energies", [])
        energy_counts = [0] * 10
        for e in energies:
            if isinstance(e, int) and 0 <= e < 10:
                energy_counts[e] += 1

        pkmn_entity = PokemonEntity(
            card_id=card_id,
            damage_counters=damage_counters,
            attached_energies=energy_counts,
            evolution_depth=len(pkmn_data.get("preEvolution", [])),
            evolved_this_turn=pkmn_data.get("appearThisTurn", False),
        )
        target_array[slot_idx, 128:224] = pkmn_entity.encode_dynamic()
        target_array[slot_idx, 224:256] = encode_zone(zone)

    def _encode_card_slot(
        self,
        slot_idx: int,
        card_id: int,
        zone: Zone,
        target_array: np.ndarray,
    ):
        card_def = self.catalog.get(card_id)
        if card_def:
            target_array[slot_idx, :128] = card_def.encode_static()
        target_array[slot_idx, 224:256] = encode_zone(zone)
