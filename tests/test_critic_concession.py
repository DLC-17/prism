"""Tests for Dual-Horizon Critic and Strategic Concession Rule."""

import torch
from models.critic import should_concede, DualHorizonCritic


def test_strategic_concession_positive_cases():
    # Game 1 (Series 0-0), severely losing game (-0.90), good match equity (-0.20), low clock (0.30)
    assert should_concede(
        v_game=-0.90,
        v_match=-0.20,
        series_score=[0.0, 0.0, 1.0],
        match_clock_remaining=0.30,
    ) is True

    # Game 2 leading 1-0, severely losing game (-0.88), match equity (-0.10), low clock (0.35)
    assert should_concede(
        v_game=-0.88,
        v_match=-0.10,
        series_score=[1.0, 0.0, 2.0],
        match_clock_remaining=0.35,
    ) is True


def test_strategic_concession_negative_cases():
    # Case 1: V_game is not low enough (-0.80 >= -0.85)
    assert should_concede(
        v_game=-0.80,
        v_match=-0.20,
        series_score=[0.0, 0.0, 1.0],
        match_clock_remaining=0.30,
    ) is False

    # Case 2: V_match <= V_game (no match equity benefit)
    assert should_concede(
        v_game=-0.90,
        v_match=-0.95,
        series_score=[0.0, 0.0, 1.0],
        match_clock_remaining=0.30,
    ) is False

    # Case 3: Trailing 0-1 (SeriesScore not in {0-0, 1-0}) -> Conceding loses match immediately!
    assert should_concede(
        v_game=-0.90,
        v_match=-0.20,
        series_score=[0.0, 1.0, 2.0],
        match_clock_remaining=0.30,
    ) is False

    # Case 4: Tied 1-1 in Game 3 -> Conceding loses match immediately!
    assert should_concede(
        v_game=-0.90,
        v_match=-0.20,
        series_score=[1.0, 1.0, 3.0],
        match_clock_remaining=0.30,
    ) is False

    # Case 5: Clock is not critical (0.55 >= 0.40)
    assert should_concede(
        v_game=-0.90,
        v_match=-0.20,
        series_score=[0.0, 0.0, 1.0],
        match_clock_remaining=0.55,
    ) is False


def test_dual_horizon_critic_evaluation():
    torch.manual_seed(42)
    critic = DualHorizonCritic(d_model=256, match_dim=64)
    h_global = torch.randn(1, 256)
    z_match = torch.randn(1, 64)

    v_game, v_match = critic(h_global, z_match)
    assert -1.0 <= v_game.item() <= 1.0
    assert -1.0 <= v_match.item() <= 1.0
