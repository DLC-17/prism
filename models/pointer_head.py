"""Autoregressive Atomic Action Decoder and Entity Pointer Head for prism."""

import torch
import torch.nn as nn
import math


class MacroActionHead(nn.Module):
    """Predicts root macro-action type distribution over 7 classes:

    (Attack, Retreat, Play Item, Play Supporter, Attach Energy, Use Ability, Pass)
    """

    def __init__(self, d_model: int = 256, micro_dim: int = 32, num_actions: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + micro_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, h_global: torch.Tensor, c_micro: torch.Tensor) -> torch.Tensor:
        """Args:

            h_global: (B, d_model)
            c_micro: (B, micro_dim)

        Returns:
            macro_logits: (B, 7)
        """
        combined = torch.cat([c_micro, h_global], dim=-1)
        return self.mlp(combined)


class EntityPointerHead(nn.Module):
    """Dot-product pointer attention over N entities in H_field:

    q_t = MLP_query([c_micro || Pooling(H_field)]) in R^{d_model}
    Logits_i = (q_t . (H_field[i] W_K)^T) / sqrt(d_k) + M_i
    """

    def __init__(self, d_model: int = 256, micro_dim: int = 32, d_k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k

        self.mlp_query = nn.Sequential(
            nn.Linear(d_model + micro_dim, 256),
            nn.GELU(),
            nn.Linear(256, d_k),
        )
        self.W_k = nn.Linear(d_model, d_k)

    def forward(
        self,
        H_field: torch.Tensor,
        h_global: torch.Tensor,
        c_micro: torch.Tensor,
        legality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

            H_field: (B, N, d_model)
            h_global: (B, d_model)
            c_micro: (B, micro_dim)
            legality_mask: (B, N) with values in {0, -inf}

        Returns:
            masked_logits: (B, N)
        """
        B, N, _ = H_field.shape

        combined = torch.cat([c_micro, h_global], dim=-1)
        q_t = self.mlp_query(combined)  # (B, d_k)
        keys = self.W_k(H_field)        # (B, N, d_k)

        # Dot-product attention: (B, 1, d_k) @ (B, d_k, N) -> (B, 1, N) -> (B, N)
        raw_logits = torch.bmm(keys, q_t.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_k)

        # Apply legality mask M_i in {0, -inf}
        masked_logits = raw_logits + legality_mask
        return masked_logits
