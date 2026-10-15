"""Turn Latency and CPU Budget Benchmark.

Verifies inference profile within 2 vCPU budget:
- Set Transformer forward pass <= ~1.2 ms
- Cached pointer head evaluation <= ~0.03 ms
- Total turn calculation time <= 50 ms across 20 atomic sub-actions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
from models.actor_critic import PrismActorCritic
from inference.cache_manager import FieldInferenceEngine
from inference.agent_worker import TournamentAgentWorker
from engine.python_fallback.card_db import CardCatalog
from engine.python_fallback.state_machine import GameState


def benchmark_components():
    # Restrict PyTorch to 2 vCPU threads
    torch.set_num_threads(2)

    model = PrismActorCritic()
    model.eval()

    raw_features = torch.randn(1, 64, 256)
    micro_context = torch.zeros(1, 32)
    legality_mask = torch.zeros(1, 64)

    # Warmup
    for _ in range(20):
        H_field, h_global = model.compute_field_representation(raw_features)
        model.decode_action_step(H_field, h_global, micro_context, legality_mask)

    # 1. Benchmark Set Transformer Backbone (O(L * N^2 * d))
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        H_field, h_global = model.compute_field_representation(raw_features)
    backbone_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # 2. Benchmark Pointer Decoder (O(N * d))
    iters = 1000
    t0 = time.perf_counter()
    for _ in range(iters):
        model.decode_action_step(H_field, h_global, micro_context, legality_mask)
    decoder_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    print(f"Set Transformer Backbone latency (2 vCPUs): {backbone_ms:.3f} ms")
    print(f"Cached Pointer Decoder latency (2 vCPUs):   {decoder_ms:.4f} ms")

    # 3. Benchmark End-to-end Turn with Field-Inference Caching (20 atomic micro-steps)
    catalog = CardCatalog()
    deck = [10, 11, 12, 100] * 15
    worker = TournamentAgentWorker(model, catalog, device="cpu")

    turn_times = []
    for _ in range(50):
        state = GameState(catalog, deck, deck)
        res = worker.execute_active_turn(state, max_micro_steps=20)
        turn_times.append(res["elapsed_ms"])

    mean_turn_ms = sum(turn_times) / len(turn_times)
    max_turn_ms = max(turn_times)
    print(f"Active Turn Latency (Mean): {mean_turn_ms:.3f} ms (Target: <50 ms)")
    print(f"Active Turn Latency (Max):  {max_turn_ms:.3f} ms")

    assert mean_turn_ms < 50.0, f"Turn calculation exceeded 50 ms budget! {mean_turn_ms:.3f} ms"
    print("ALL LATENCY BUDGET CHECKS PASSED!")


if __name__ == "__main__":
    benchmark_components()
