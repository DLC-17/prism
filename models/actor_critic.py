"""Unified Prism Actor-Critic Architecture."""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from .embeddings import CompositeEntityTokenizer
from .set_transformer import RelationalSetTransformerBackbone, PMA1
from .pointer_head import MacroActionHead, EntityPointerHead
from .critic import DualHorizonCritic


class PrismActorCritic(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        card_pool_size: int = 2048,
        num_sab_layers: int = 4,
        num_heads: int = 8,
        d_ffn: int = 1024,
        micro_dim: int = 32,
        match_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = CompositeEntityTokenizer(
            d_model=d_model,
            card_pool_size=card_pool_size,
        )
        self.backbone = RelationalSetTransformerBackbone(
            d_model=d_model,
            num_layers=num_sab_layers,
            num_heads=num_heads,
            d_ffn=d_ffn,
        )
        self.pma1 = PMA1(d_model=d_model, num_heads=num_heads)
        self.macro_head = MacroActionHead(d_model=d_model, micro_dim=micro_dim)
        self.pointer_head = EntityPointerHead(d_model=d_model, micro_dim=micro_dim)
        self.critic = DualHorizonCritic(d_model=d_model, match_dim=match_dim)

    def compute_field_representation(
        self,
        raw_features: torch.Tensor,
        card_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs Tokenizer + Set Transformer Backbone + PMA1 global pooling.

        Triggered strictly on physical board mutations: O(L * N^2 * d).

        Returns:
            H_field: (B, N, d_model)
            h_global: (B, d_model)
        """
        X = self.tokenizer(raw_features, card_ids=card_ids)
        H_field = self.backbone(X)
        h_global = self.pma1(H_field)
        return H_field, h_global

    def decode_action_step(
        self,
        H_field: torch.Tensor,
        h_global: torch.Tensor,
        c_micro: torch.Tensor,
        legality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs lightweight sub-action execution using cached field representations: O(N * d).

        Returns:
            macro_logits: (B, 7)
            entity_logits: (B, N)
        """
        macro_logits = self.macro_head(h_global, c_micro)
        entity_logits = self.pointer_head(H_field, h_global, c_micro, legality_mask)
        return macro_logits, entity_logits

    def evaluate_values(
        self,
        h_global: torch.Tensor,
        z_match: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns V_game in [-1, 1] and V_match in [-1, 1]."""
        return self.critic(h_global, z_match)
