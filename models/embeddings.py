"""Composite Entity Tokenizer for prism.

Maps composite entity features into d_model (256):
x_i = LayerNorm(MLP([e_static(i) || e_dynamic(i) || e_zone(i)]))
"""

import torch
import torch.nn as nn
from typing import Optional


class CompositeEntityTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        card_pool_size: int = 2048,
        card_embed_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.card_pool_size = card_pool_size
        self.card_embed_dim = card_embed_dim

        # Trainable card ID embedding (64) for end-to-end learning
        self.card_embedding = nn.Embedding(card_pool_size, card_embed_dim)

        # Raw input dimension is 128 (static) + 96 (dynamic) + 32 (zone) = 256
        self.raw_dim = 256

        self.mlp = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        raw_features: torch.Tensor,
        card_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

            raw_features: Tensor of shape (B, N, 256) containing [e_static || e_dynamic || e_zone]
            card_ids: Optional Tensor of shape (B, N) with discrete card IDs to fuse
                      dense learned embeddings into the first 64 static slots.

        Returns:
            X: Tensor of shape (B, N, d_model)
        """
        B, N, D = raw_features.shape
        assert D == self.raw_dim, f"Expected raw feature dim {self.raw_dim}, got {D}"

        feats = raw_features
        if card_ids is not None:
            # Overwrite or add learned card embedding to the first 64 dims of e_static
            card_embed = self.card_embedding(card_ids)  # (B, N, 64)
            feats = torch.cat(
                [
                    card_embed,
                    raw_features[..., 64:],
                ],
                dim=-1,
            )

        projected = self.mlp(feats)
        return self.layer_norm(projected)
