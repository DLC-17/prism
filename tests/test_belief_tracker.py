"""Tests for 3-segment deck belief tracker, deterministic deck search, and Bayes updates."""

import torch
import numpy as np
from models.belief_tracker import (
    DeckBeliefTracker,
    BayesianArchetypeTracker,
    MatchContextBuilder,
)


def test_deck_belief_operations():
    tracker = DeckBeliefTracker(card_pool_size=200)
    tracker.reset_with_deck([10] * 4 + [11] * 3 + [12] * 2 + [100] * 10)

    # Initially middle pool has cards
    t = tracker.to_tensor()
    assert t.shape == (3, 200)
    assert t[0].sum().item() == 0.0  # Top stack empty
    assert pytest_approx(t[1].sum().item(), 1.0)  # Middle pool sums to 1.0

    # Ciphermaniac: places cards 12 and 11 on top stack
    tracker.trigger_ciphermaniac(card_a=12, card_b=11)
    t = tracker.to_tensor()
    assert pytest_approx(t[0].sum().item(), 1.0)
    assert t[0, 12].item() > 0.0
    assert t[0, 11].item() > 0.0

    # Shuffle: merges back to middle pool
    tracker.trigger_shuffle()
    t = tracker.to_tensor()
    assert t[0].sum().item() == 0.0
    assert pytest_approx(t[1].sum().item(), 1.0)

    # Iono: places cards on bottom queue
    tracker.trigger_iono([100, 100])
    t = tracker.to_tensor()
    assert pytest_approx(t[2].sum().item(), 1.0)
    assert t[2, 100].item() > 0.0


def test_deterministic_deck_search_resolution():
    # Initial 60 card deck
    initial_60 = [10] * 4 + [11] * 4 + [12] * 4 + [1] * 4 + [100] * 44

    # Visible cards on board, hand, discard, and remaining deck
    # 6 prize cards are missing: say one 12, one 11, and four 100
    visible_cards = [10] * 4 + [11] * 3 + [12] * 3 + [1] * 4 + [100] * 40
    assert len(visible_cards) == 54

    prizes = DeckBeliefTracker.solve_exact_prizes(initial_60, visible_cards)
    assert len(prizes) == 6
    assert prizes.count(12) == 1
    assert prizes.count(11) == 1
    assert prizes.count(100) == 4


def test_bayesian_archetype_inference():
    tracker = BayesianArchetypeTracker(num_archetypes=32)

    # Initially uniform
    post = tracker.get_posterior()
    assert np.allclose(post, 1.0 / 32.0, atol=1e-3)

    # Observe Charmander (id 10) and Charizard ex (id 12)
    tracker.observe_card(10)
    tracker.observe_card(12)

    post = tracker.get_posterior()
    # Archetype 0 is Charizard ex, should have overwhelming posterior probability
    assert post[0] > 0.80
    assert post[0] > post[1]
    assert post[0] > post[2]


def pytest_approx(val, target, tol=1e-4):
    return abs(val - target) < tol
