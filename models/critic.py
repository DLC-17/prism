"""Dual-Horizon Value Critic & Strategic Concession Engine for prism."""

import torch
import torch.nn as nn
from typing import Tuple, List, Union


def should_concede(
    v_game: float,
    v_match: float,
    series_score: Union[List[float], Tuple[float, float, float]],
    match_clock_remaining: float,
) -> bool:
    """Evaluates the Strategic Concession Rule:

    Concession is triggered at root action nodes when:
    V_game < -0.85 AND V_match > V_game AND SeriesScore in {0-0, 1-0} AND t_clock < 0.40
    """
    # series_score format: [self_wins, opp_wins, game_num]
    # SeriesScore in {0-0, 1-0}:
    self_wins = int(series_score[0])
    opp_wins = int(series_score[1])
    is_valid_series_state = (self_wins == 0 and opp_wins == 0) or (self_wins == 1 and opp_wins == 0)

    is_losing_game = v_game < -0.85
    has_match_equity = v_match > v_game
    is_clock_critical = match_clock_remaining < 0.40

    return is_losing_game and has_match_equity and is_valid_series_state and is_clock_critical


class DualHorizonCritic(nn.Module):
    """Pulls the pooled global representation h_global in R^{d_model}

    and concatenates z_match in R^{64}:
    V_game = tanh(MLP_game([h_global || z_match])) in [-1, 1]
    V_match = tanh(MLP_match([h_global || z_match])) in [-1, 1]
    """

    def __init__(self, d_model: int = 256, match_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        in_dim = d_model + match_dim

        self.mlp_game = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.mlp_match = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        h_global: torch.Tensor,
        z_match: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            h_global: (B, d_model)
            z_match: (B, match_dim)

        Returns:
            V_game: (B, 1) in [-1, 1]
            V_match: (B, 1) in [-1, 1]
        """
        combined = torch.cat([h_global, z_match], dim=-1)
        v_game = self.mlp_game(combined)
        v_match = self.mlp_match(combined)
        return v_game, v_match

    def evaluate_concession(
        self,
        h_global: torch.Tensor,
        z_match: torch.Tensor,
        series_score: List[float],
        match_clock_remaining: float,
    ) -> bool:
        with torch.no_grad():
            v_game, v_match = self.forward(h_global, z_match)
            return should_concede(
                v_game.item(),
                v_match.item(),
                series_score,
                match_clock_remaining,
            )
