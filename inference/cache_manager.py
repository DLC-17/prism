"""Field-Inference Caching Engine.

Decouples the O(L * N^2 * d) Set Transformer backbone from
O(N * d) atomic pointer decoders during multi-step card resolution sequences.
"""

import torch
from typing import Tuple, Optional
from models.actor_critic import PrismActorCritic


class FieldInferenceEngine:
    """Field Inference Caching Engine.

    Cache Invalidation Contract:
    - INVALIDATE: Card enters play, card discarded, damage counters modified,
                  energy attached, attack resolved, shuffle executed.
    - PRESERVE: Intra-card sub-selections (selecting Ultra Ball discards,
                picking search targets, selecting retreat energy cost discards).
    """

    def __init__(self, model: PrismActorCritic):
        self.model = model
        self.cache_H: Optional[torch.Tensor] = None
        self.cache_h_global: Optional[torch.Tensor] = None
        self.cache_valid: bool = False
        self.total_backbone_calls: int = 0
        self.total_decoder_calls: int = 0

    def on_board_mutation(self):
        """Invalidates the cached field representations due to physical board mutation."""
        self.cache_valid = False

    def forward_step(
        self,
        raw_features: torch.Tensor,
        card_ids: Optional[torch.Tensor],
        micro_step_context: torch.Tensor,
        legality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a forward step.

        Only recomputes the backbone if the cache has been invalidated.
        Always evaluates the pointer decoder using cached H_field.
        """
        if not self.cache_valid or self.cache_H is None or self.cache_h_global is None:
            self.cache_H, self.cache_h_global = self.model.compute_field_representation(
                raw_features, card_ids
            )
            self.cache_valid = True
            self.total_backbone_calls += 1

        # Lightweight sub-action execution: O(N * d)
        macro_logits, entity_logits = self.model.decode_action_step(
            self.cache_H,
            self.cache_h_global,
            micro_step_context,
            legality_mask,
        )
        self.total_decoder_calls += 1

        return macro_logits, entity_logits
