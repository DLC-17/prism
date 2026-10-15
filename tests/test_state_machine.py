"""Tests for state machine transitions, legality masks, and atomic action resolution."""

import numpy as np
import pytest
from engine.python_fallback.card_db import CardCatalog
from engine.python_fallback.state_machine import (
    GameState,
    SubState,
    HAND_SELF_START,
    GLOBAL_TOKEN_SLOT,
)


def test_initial_board_and_root_legality_mask():
    catalog = CardCatalog()
    self_deck = [10, 11, 12, 100] * 15
    opp_deck = [10, 11, 12, 100] * 15

    state = GameState(catalog, self_deck, opp_deck)
    assert state.sub_state == SubState.IDLE_ROOT
    assert state.has_board_mutated is True

    mask = state.compute_legality_mask()

    # Active Pokémon and GlobalToken (Pass) should be legal
    assert mask[0] == 0.0
    assert mask[GLOBAL_TOKEN_SLOT] == 0.0

    # Hand cards (Ultra Ball, Fire Energy, Charmeleon)
    # Ultra Ball and Fire Energy are playable items/energy at root
    assert mask[HAND_SELF_START] == 0.0  # Ultra Ball
    assert mask[HAND_SELF_START + 1] == 0.0  # Fire Energy


def test_ultra_ball_multi_step_resolution_and_invalidation():
    catalog = CardCatalog()
    self_deck = [10, 11, 12, 100] * 15
    opp_deck = [10, 11, 12, 100] * 15

    state = GameState(catalog, self_deck, opp_deck)

    # Step 1: Play Ultra Ball
    ub_idx = HAND_SELF_START
    res = state.step_micro_action(ub_idx)
    assert res is True
    assert state.sub_state == SubState.SELECT_DISCARD_1
    assert state.has_board_mutated is False  # Intra-card sub-selection: PRESERVE CACHE

    mask1 = state.compute_legality_mask()
    # Ultra Ball itself must be masked out (-inf)
    assert mask1[ub_idx] == -np.inf
    # Other hand cards must be legal (0.0)
    assert mask1[HAND_SELF_START + 1] == 0.0
    assert mask1[HAND_SELF_START + 2] == 0.0

    # Step 2: Select first discard (Fire Energy at HAND_SELF_START + 1)
    d1_idx = HAND_SELF_START + 1
    res = state.step_micro_action(d1_idx)
    assert res is True
    assert state.sub_state == SubState.SELECT_DISCARD_2
    assert state.has_board_mutated is False  # Intra-card sub-selection: PRESERVE CACHE

    mask2 = state.compute_legality_mask()
    # Both Ultra Ball and discard 1 must be masked out (-inf)
    assert mask2[ub_idx] == -np.inf
    assert mask2[d1_idx] == -np.inf
    # Discard 2 must be legal
    assert mask2[HAND_SELF_START + 2] == 0.0

    # Step 3: Select second discard (Charmeleon at HAND_SELF_START + 2)
    d2_idx = HAND_SELF_START + 2
    res = state.step_micro_action(d2_idx)
    assert res is True
    assert state.sub_state == SubState.SELECT_SEARCH_TARGET

    # Step 4: Select search target (or fail-to-find)
    res = state.step_micro_action(GLOBAL_TOKEN_SLOT)
    assert res is True
    assert state.sub_state == SubState.IDLE_ROOT
    assert state.has_board_mutated is True  # Board mutated! Cache INVALIDATED

    # Verify Charizard ex (id 12) added to hand
    hand_card_ids = [
        state.entities[i].card_id
        for i in range(HAND_SELF_START, HAND_SELF_START + 5)
        if state.entities[i] is not None
    ]
    assert 12 in hand_card_ids
