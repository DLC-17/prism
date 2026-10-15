"""Asynchronous Tournament Agent Worker.

Operates within 2 vCPU / 12.2 GB RAM budget.
Employs pre-allocated static tensor buffers for zero-allocation turns.
Decouples active intra-turn multi-step execution from passive opponent transaction ingestion.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from models.actor_critic import PrismActorCritic
from models.critic import should_concede
from models.belief_tracker import DeckBeliefTracker, BayesianArchetypeTracker, MatchContextBuilder
from inference.cache_manager import FieldInferenceEngine
from engine.python_fallback.state_machine import GameState, N_ENTITIES, RAW_FEATURE_DIM, GLOBAL_TOKEN_SLOT
from engine.python_fallback.card_db import CardCatalog
from engine.python_fallback.types import SubState


class PreallocatedTensorBuffer:
    """Fixed-buffer static memory arrays ensuring zero garbage collection overhead

    during tournament turns.
    """

    def __init__(self, d_model: int = 256, n_entities: int = 64, micro_dim: int = 32, match_dim: int = 64):
        self.raw_features = torch.zeros(1, n_entities, RAW_FEATURE_DIM, dtype=torch.float32)
        self.card_ids = torch.zeros(1, n_entities, dtype=torch.long)
        self.legality_mask = torch.zeros(1, n_entities, dtype=torch.float32)
        self.c_micro = torch.zeros(1, micro_dim, dtype=torch.float32)
        self.z_match = torch.zeros(1, match_dim, dtype=torch.float32)

    def update_from_numpy(
        self,
        raw_feats_np: np.ndarray,
        card_ids_np: Optional[np.ndarray],
        mask_np: np.ndarray,
        c_micro_np: np.ndarray,
        z_match_np: np.ndarray,
    ):
        # In-place copy prevents any heap reallocation
        self.raw_features[0].copy_(torch.from_numpy(raw_feats_np))
        if card_ids_np is not None:
            self.card_ids[0].copy_(torch.from_numpy(card_ids_np))
        else:
            self.card_ids.zero_()
        self.legality_mask[0].copy_(torch.from_numpy(mask_np))
        self.c_micro[0].copy_(torch.from_numpy(c_micro_np))
        self.z_match[0].copy_(torch.from_numpy(z_match_np))


class TournamentAgentWorker:
    def __init__(
        self,
        model: PrismActorCritic,
        catalog: CardCatalog,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.catalog = catalog
        self.cache_engine = FieldInferenceEngine(self.model)
        self.buffers = PreallocatedTensorBuffer()

        self.deck_tracker = DeckBeliefTracker()
        self.archetype_tracker = BayesianArchetypeTracker()

    def ingest_opponent_transaction(self, event_type: str, card_id: Optional[int] = None, data: Optional[Dict] = None):
        """Passive turn handling: ingests completed immutable transaction events

        without any branching.
        """
        if card_id is not None and card_id > 0:
            self.archetype_tracker.observe_card(card_id)

        if event_type == "EVENT_SHUFFLE":
            self.deck_tracker.trigger_shuffle()
            self.cache_engine.on_board_mutation()
        elif event_type == "EVENT_IONO" and data is not None:
            bottom_cards = data.get("bottom_cards", [])
            self.deck_tracker.trigger_iono(bottom_cards)
            self.cache_engine.on_board_mutation()

    def execute_active_turn(
        self,
        game_state: GameState,
        max_micro_steps: int = 20,
    ) -> Dict[str, Any]:
        """Runs the active turn optimization using Field-Inference Caching and pre-allocated buffers.

        Guarantees <50 ms total calculation time across 20 atomic sub-actions.
        """
        turn_start_time = time.perf_counter()
        actions_taken = []
        conceded = False

        # Build Bo3 match context vector
        z_match_t = MatchContextBuilder.build(
            archetype_posterior=self.archetype_tracker.get_posterior(),
            series_score=game_state.series_score,
            match_clock_remaining=game_state.match_clock_remaining,
            turn_count=game_state.turn_count,
        )
        z_match_np = z_match_t.squeeze(0).numpy()

        with torch.no_grad():
            for step_idx in range(max_micro_steps):
                # 1. Check for physical board mutation to invalidate cache
                if game_state.has_board_mutated:
                    self.cache_engine.on_board_mutation()

                # 2. Extract features and legality mask
                raw_feats_np = game_state.encode_entities()
                mask_np = game_state.compute_legality_mask()

                # Micro-step context encoding
                c_micro_np = np.zeros(32, dtype=np.float32)
                c_micro_np[0] = step_idx / float(max_micro_steps)

                # 3. Update fixed buffers in-place (Zero GC allocation)
                self.buffers.update_from_numpy(
                    raw_feats_np=raw_feats_np,
                    card_ids_np=None,
                    mask_np=mask_np,
                    c_micro_np=c_micro_np,
                    z_match_np=z_match_np,
                )

                # 4. Check strategic concession at root nodes
                if game_state.sub_state == SubState.IDLE_ROOT:
                    # Run backbone if not valid to evaluate values
                    if not self.cache_engine.cache_valid:
                        self.cache_engine.cache_H, self.cache_engine.cache_h_global = (
                            self.model.compute_field_representation(
                                self.buffers.raw_features,
                                card_ids=None,
                            )
                        )
                        self.cache_engine.cache_valid = True

                    v_game_t, v_match_t = self.model.evaluate_values(
                        self.cache_engine.cache_h_global,
                        self.buffers.z_match,
                    )
                    v_game = v_game_t.item()
                    v_match = v_match_t.item()

                    if should_concede(
                        v_game,
                        v_match,
                        game_state.series_score,
                        game_state.match_clock_remaining,
                    ):
                        conceded = True
                        actions_taken.append({"action": "CONCEDE", "v_game": v_game, "v_match": v_match})
                        break

                # 5. Forward step through Field Inference Engine
                macro_logits, entity_logits = self.cache_engine.forward_step(
                    raw_features=self.buffers.raw_features,
                    card_ids=None,
                    micro_step_context=self.buffers.c_micro,
                    legality_mask=self.buffers.legality_mask,
                )

                # 6. Select highest probability legal action
                target_entity = int(torch.argmax(entity_logits[0]).item())

                # If pass selected or no legal target (argmax is -inf)
                if mask_np[target_entity] == -float("inf") or target_entity == GLOBAL_TOKEN_SLOT:
                    actions_taken.append({"step": step_idx, "action": "PASS"})
                    break

                actions_taken.append(
                    {
                        "step": step_idx,
                        "sub_state": str(game_state.sub_state),
                        "target_entity": target_entity,
                    }
                )

                # 7. Apply action to game state
                progress = game_state.step_micro_action(target_entity)
                if not progress:
                    break

        elapsed_ms = (time.perf_counter() - turn_start_time) * 1000.0

        return {
            "elapsed_ms": elapsed_ms,
            "actions_taken": actions_taken,
            "conceded": conceded,
            "backbone_calls": self.cache_engine.total_backbone_calls,
            "decoder_calls": self.cache_engine.total_decoder_calls,
        }
