"""Kaggle Submission Agent Entrypoint for The Pokémon Company - PTCG AI Battle Challenge.

Runs inside /kaggle_simulations/agent/
"""

import sys
import os
import torch

# Ensure agent directory is in Python path
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from models.actor_critic import PrismActorCritic
from engine.cabt_adapter import CabtObservationAdapter
from engine.python_fallback.card_db import CardCatalog
from engine.hydrapple_catalog import register_hydrapple_archetype
from inference.cache_manager import FieldInferenceEngine

# Global instances initialized once
_CATALOG = CardCatalog()
register_hydrapple_archetype(_CATALOG)
_ADAPTER = CabtObservationAdapter(_CATALOG)

_DEVICE = "cpu"
_MODEL = PrismActorCritic()
_WEIGHTS_PATH = os.path.join(AGENT_DIR, "prism_model.pt")

if os.path.exists(_WEIGHTS_PATH):
    try:
        _MODEL.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=_DEVICE))
        print("Loaded trained Prism weights successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights from {_WEIGHTS_PATH}: {e}")

_MODEL.to(_DEVICE)
_MODEL.eval()
_CACHE_ENGINE = FieldInferenceEngine(_MODEL)


def agent(observation, configuration=None):
    """Kaggle-environments entrypoint for Pokémon TCG AI Battle Challenge.

    Args:
        observation: dict with keys 'current', 'select', 'logs'
        configuration: dict with game configuration

    Returns:
        List of selected option indices, e.g. [0]
    """
    select_data = observation.get("select")
    if not select_data or "option" not in select_data or not select_data["option"]:
        # Setup or default phase: return [0]
        return [0]

    options = select_data["option"]
    if len(options) == 1:
        return [0]

    # Check for logs indicating physical mutation to invalidate cache
    logs = observation.get("logs", [])
    if logs:
        # Check if last log is a mutation event
        last_log = logs[-1] if isinstance(logs, list) else {}
        log_type = last_log.get("type") if isinstance(last_log, dict) else None
        if log_type in ("MOVE_CARD", "ATTACH", "EVOLVE", "DEVOLVE", "HP_CHANGE", "SHUFFLE"):
            _CACHE_ENGINE.on_board_mutation()

    # Determine player index
    current = observation.get("current")
    your_index = current.get("yourIndex", 0) if current else 0

    with torch.no_grad():
        raw_x, mask, z_match = _ADAPTER.observation_to_tensors(observation, your_index=your_index)
        c_micro = torch.zeros(1, 32)

        # Forward step through Field-Inference Caching engine
        macro_logits, entity_logits = _CACHE_ENGINE.forward_step(
            raw_features=raw_x,
            card_ids=None,
            micro_step_context=c_micro,
            legality_mask=mask,
        )

        # Map entity logits to available option indices
        best_option_idx = 0
        best_score = -float("inf")

        for opt_idx, opt in enumerate(options):
            area = opt.get("area")
            index = opt.get("index", 0)
            opt_type = opt.get("type")

            # Entity slot mapping
            target_entity_slot = 49  # Default to GlobalToken (Pass/End)
            if area == "ACTIVE":
                target_entity_slot = 0
            elif area == "BENCH" and index < 5:
                target_entity_slot = 1 + index
            elif area == "HAND" and index < 20:
                target_entity_slot = 7 + index

            score = entity_logits[0, target_entity_slot].item()
            # If end or pass
            if opt_type in ("END", "NO", "PASS"):
                score = entity_logits[0, 49].item()

            if score > best_score:
                best_score = score
                best_option_idx = opt_idx

    return [best_option_idx]
