"""Tests for Field-Inference Caching Engine and invalidation contracts."""

import torch
from models.actor_critic import PrismActorCritic
from inference.cache_manager import FieldInferenceEngine


def test_field_inference_caching_and_invalidation():
    torch.manual_seed(42)
    model = PrismActorCritic()
    model.eval()

    cache_engine = FieldInferenceEngine(model)
    assert cache_engine.cache_valid is False
    assert cache_engine.total_backbone_calls == 0
    assert cache_engine.total_decoder_calls == 0

    raw_features = torch.randn(1, 64, 256)
    micro_context = torch.zeros(1, 32)
    legality_mask = torch.zeros(1, 64)

    # Step 1: Initial call triggers backbone: O(L * N^2 * d)
    cache_engine.forward_step(raw_features, None, micro_context, legality_mask)
    assert cache_engine.cache_valid is True
    assert cache_engine.total_backbone_calls == 1
    assert cache_engine.total_decoder_calls == 1

    # Step 2: Intra-card sub-selection (e.g. Ultra Ball discard 1) -> Backbone NOT called!
    cache_engine.forward_step(raw_features, None, micro_context, legality_mask)
    assert cache_engine.cache_valid is True
    assert cache_engine.total_backbone_calls == 1
    assert cache_engine.total_decoder_calls == 2

    # Step 3: Intra-card sub-selection (Ultra Ball discard 2) -> Backbone NOT called!
    cache_engine.forward_step(raw_features, None, micro_context, legality_mask)
    assert cache_engine.cache_valid is True
    assert cache_engine.total_backbone_calls == 1
    assert cache_engine.total_decoder_calls == 3

    # Step 4: Physical board mutation occurs (card moved to discard, shuffle) -> Invalidate!
    cache_engine.on_board_mutation()
    assert cache_engine.cache_valid is False

    # Step 5: Next action forces backbone recalculation
    cache_engine.forward_step(raw_features, None, micro_context, legality_mask)
    assert cache_engine.cache_valid is True
    assert cache_engine.total_backbone_calls == 2
    assert cache_engine.total_decoder_calls == 4
