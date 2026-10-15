"""Tests for CabtObservationAdapter mapping cabt simulator observations to Prism tensors."""

import torch
import pytest
from engine.cabt_adapter import CabtObservationAdapter
from engine.python_fallback.card_db import CardCatalog


def test_cabt_observation_adapter_tensor_shapes():
    adapter = CabtObservationAdapter(CardCatalog())

    mock_obs = {
        "current": {
            "turn": 3,
            "turnActionCount": 1,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [
                {
                    "active": [{"id": 10, "hp": 50, "maxHp": 70, "energies": [1], "preEvolution": []}],
                    "bench": [{"id": 11, "hp": 90, "maxHp": 90, "energies": [1, 1], "preEvolution": [{"id": 10}]}],
                    "hand": [{"id": 1}, {"id": 100}],
                    "prize": [None] * 5,
                    "discard": [],
                },
                {
                    "active": [{"id": 10, "hp": 70, "maxHp": 70, "energies": [], "preEvolution": []}],
                    "bench": [],
                    "handCount": 4,
                    "prize": [None] * 6,
                    "discard": [],
                },
            ],
        },
        "select": {
            "type": "MAIN",
            "context": "MAIN",
            "option": [
                {"type": "ATTACK", "area": "ACTIVE", "index": 0},
                {"type": "PLAY", "area": "HAND", "index": 0},
                {"type": "END"},
            ],
        },
        "logs": [],
    }

    raw_features, mask, z_match = adapter.observation_to_tensors(mock_obs, your_index=0)

    # Verify tensor shapes
    assert raw_features.shape == (1, 64, 256)
    assert mask.shape == (1, 64)
    assert z_match.shape == (1, 64)

    # Verify active Pokémon damage counter: (70 - 50) / 10 = 2 -> 2.0 / 10.0 = 0.2
    assert abs(raw_features[0, 0, 128].item() - 0.2) < 1e-4

    # Verify bench Pokémon evolution depth: len(preEvolution) = 1 -> 1.0 / 3.0 = 0.333
    # 128 (static) + 1 (dmg) + 11 (energy) + 24 (slot1) + 1 (hp) + 24 (slot2) = 189
    assert abs(raw_features[0, 1, 189].item() - 0.3333) < 1e-3

    # Verify legality mask: Active, Hand[0], and GlobalToken (Pass) should be 0.0
    assert mask[0, 0].item() == 0.0  # Active
    assert mask[0, 7].item() == 0.0  # Hand[0]
    assert mask[0, 49].item() == 0.0  # GlobalToken (Pass)
