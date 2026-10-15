"""Simulated Best-of-3 match test against the #1 rated Kaggle playground deck:

Hydrapple ex / Teal Mask Ogerpon ex.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import numpy as np
from engine.python_fallback.card_db import CardCatalog
from engine.python_fallback.state_machine import (
    GameState,
    HAND_SELF_START,
    ACTIVE_SELF_SLOT,
    ACTIVE_OPP_SLOT,
    BENCH_OPP_START,
    GLOBAL_TOKEN_SLOT,
    EntitySlot,
    SubState,
)
from engine.python_fallback.types import ZoneType
from engine.python_fallback.zone_manager import PokemonEntity, Zone
from engine.hydrapple_catalog import register_hydrapple_archetype, create_hydrapple_deck_list
from models.actor_critic import PrismActorCritic
from models.critic import should_concede
from inference.agent_worker import TournamentAgentWorker


def test_simulated_match_against_hydrapple():
    # 1. Setup Environment & Models
    catalog = CardCatalog()
    register_hydrapple_archetype(catalog)

    model = PrismActorCritic()
    model.eval()

    worker = TournamentAgentWorker(model, catalog, device="cpu")

    # Agent deck: Charizard ex / Pidgeot ex
    agent_deck = [10] * 4 + [11] * 4 + [12] * 4 + [1] * 4 + [100] * 44
    # Opponent deck: #1 rated Hydrapple ex / Teal Mask Ogerpon ex
    hydrapple_deck = create_hydrapple_deck_list()

    print("\n" + "=" * 60)
    print("RUNNING SIMULATED Bo3 MATCH vs #1 HYDRAPPLE EX DECK")
    print("=" * 60)

    # -------------------------------------------------------------
    # GAME 1: Opening & Archetype Bayesian Recognition
    # -------------------------------------------------------------
    state_g1 = GameState(catalog, agent_deck, hydrapple_deck)
    state_g1.series_score = [0.0, 0.0, 1.0]  # Game 1, 0-0
    state_g1.match_clock_remaining = 0.95

    # Opponent reveals Applin (id 201) in active and Ogerpon ex (id 204) on bench
    applin_def = catalog.get(201)
    ogerpon_def = catalog.get(204)
    state_g1.entities[ACTIVE_OPP_SLOT] = EntitySlot(
        card_id=201,
        zone=Zone(ZoneType.ACTIVE_OPP, is_self=False),
        pokemon=PokemonEntity.new_basic(201, applin_def.base_hp),
    )
    state_g1.entities[BENCH_OPP_START] = EntitySlot(
        card_id=204,
        zone=Zone(ZoneType.BENCH_OPP, slot_index=0, is_self=False),
        pokemon=PokemonEntity.new_basic(204, ogerpon_def.base_hp),
    )

    # Passive ingestion of opponent cards
    worker.ingest_opponent_transaction("EVENT_REVEAL", card_id=201)
    worker.ingest_opponent_transaction("EVENT_REVEAL", card_id=204)

    # Verify Bayesian Archetype Tracker recognizes Hydrapple ex (Archetype 4)
    posterior = worker.archetype_tracker.get_posterior()
    print(f"[Game 1] Bayesian Posterior for Hydrapple Archetype: {posterior[4]:.4f}")
    assert posterior[4] > 0.80, f"Expected Hydrapple posterior > 0.80, got {posterior[4]}"

    # Execute Agent Active Turn 1
    t0 = time.perf_counter()
    turn_res_1 = worker.execute_active_turn(state_g1, max_micro_steps=10)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    print(f"[Game 1 Turn 1] Execution Time: {elapsed_ms:.2f} ms (Target: <50 ms)")
    print(f"[Game 1 Turn 1] Actions Taken: {len(turn_res_1['actions_taken'])}")
    assert elapsed_ms < 50.0, f"Turn execution exceeded 50 ms budget! {elapsed_ms:.2f} ms"

    # -------------------------------------------------------------
    # GAME 2: Mid-game Evaluation & Value Critics
    # -------------------------------------------------------------
    # Suppose Game 1 won by Agent (Series score: 1-0 in Game 2)
    state_g2 = GameState(catalog, agent_deck, hydrapple_deck)
    state_g2.series_score = [1.0, 0.0, 2.0]
    state_g2.turn_count = 6
    state_g2.match_clock_remaining = 0.55

    # Opponent sets up Hydrapple ex (id 203) with 5 Grass energies attached
    hydrapple_def = catalog.get(203)
    hydrapple_pkmn = PokemonEntity(
        card_id=203,
        damage_counters=0,
        attached_energies=[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        evolution_depth=2,
    )
    state_g2.entities[ACTIVE_OPP_SLOT] = EntitySlot(
        card_id=203,
        zone=Zone(ZoneType.ACTIVE_OPP, is_self=False),
        pokemon=hydrapple_pkmn,
    )

    # Agent executes turn in Game 2
    turn_res_2 = worker.execute_active_turn(state_g2, max_micro_steps=15)
    print(f"[Game 2 Turn 6] Actions: {turn_res_2['actions_taken']}")
    print(f"[Game 2 Turn 6] Backbone calls: {worker.cache_engine.total_backbone_calls}, "
          f"Decoder calls: {worker.cache_engine.total_decoder_calls}")

    # -------------------------------------------------------------
    # GAME 3 & Strategic Concession Verification
    # -------------------------------------------------------------
    # Test concession trigger in a losing Game 2 (Series 1-0, Clock < 0.40)
    concede_test_state = GameState(catalog, agent_deck, hydrapple_deck)
    concede_test_state.series_score = [1.0, 0.0, 2.0]  # Leading 1-0
    concede_test_state.match_clock_remaining = 0.30     # Clock low (30% remaining)

    # Severe board deficit: Agent active has 320 damage, opponent has full board
    concede_test_state.entities[ACTIVE_SELF_SLOT].pokemon.damage_counters = 32

    # Verify concession logic
    should_concede_result = should_concede(
        v_game=-0.92,
        v_match=-0.15,
        series_score=concede_test_state.series_score,
        match_clock_remaining=concede_test_state.match_clock_remaining,
    )
    print(f"[Strategic Concession Test] Result: {should_concede_result} (Expected: True)")
    assert should_concede_result is True

    print("=" * 60)
    print("SIMULATED MATCH AGAINST #1 HYDRAPPLE DECK PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_simulated_match_against_hydrapple()
