"""Training pipeline for Prism Actor-Critic on Kaggle Notebooks."""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.actor_critic import PrismActorCritic
from engine.cabt_adapter import CabtObservationAdapter
from engine.python_fallback.card_db import CardCatalog
from engine.hydrapple_catalog import register_hydrapple_archetype
from inference.cache_manager import FieldInferenceEngine


def generate_synthetic_training_batch(batch_size: int = 32, device: str = "cpu"):
    """Generates synthetic replay batch for training warm-up."""
    # (B, 64, 256)
    raw_features = torch.randn(batch_size, 64, 256, device=device)
    c_micro = torch.randn(batch_size, 32, device=device)
    z_match = torch.randn(batch_size, 64, device=device)

    # Legality mask with some legal slots (0.0) and others (-inf)
    legality_mask = torch.full((batch_size, 64), -float("inf"), device=device)
    target_entity = torch.empty(batch_size, dtype=torch.long, device=device)
    for b in range(batch_size):
        legal_indices = np.random.choice(64, size=8, replace=False)
        legality_mask[b, legal_indices] = 0.0
        target_entity[b] = int(np.random.choice(legal_indices))

    # Target labels
    target_macro = torch.randint(0, 7, (batch_size,), device=device)
    target_v_game = torch.tanh(torch.randn(batch_size, 1, device=device))
    target_v_match = torch.tanh(torch.randn(batch_size, 1, device=device))

    return {
        "raw_features": raw_features,
        "c_micro": c_micro,
        "z_match": z_match,
        "legality_mask": legality_mask,
        "target_macro": target_macro,
        "target_entity": target_entity,
        "target_v_game": target_v_game,
        "target_v_match": target_v_match,
    }


def train_prism(
    epochs: int = 5,
    steps_per_epoch: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cpu",
    output_dir: str = "./checkpoints",
):
    """Trains PrismActorCritic with multi-task loss (Macro + Pointer + Dual Critics)."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Training Prism Actor-Critic on device: {device}")

    model = PrismActorCritic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            batch = generate_synthetic_training_batch(batch_size=batch_size, device=device)

            optimizer.zero_grad()

            # 1. Compute field representations & global pooling
            H_field, h_global = model.compute_field_representation(batch["raw_features"])

            # 2. Decode micro-step actions
            macro_logits, entity_logits = model.decode_action_step(
                H_field=H_field,
                h_global=h_global,
                c_micro=batch["c_micro"],
                legality_mask=batch["legality_mask"],
            )

            # 3. Dual-Horizon Values
            v_game, v_match = model.evaluate_values(h_global, batch["z_match"])

            # 4. Compute Multi-Task Loss
            loss_macro = criterion_ce(macro_logits, batch["target_macro"])
            loss_entity = criterion_ce(entity_logits, batch["target_entity"])
            loss_v_game = criterion_mse(v_game, batch["target_v_game"])
            loss_v_match = criterion_mse(v_match, batch["target_v_match"])

            loss = loss_macro + loss_entity + 0.5 * (loss_v_game + loss_v_match)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    # Save trained checkpoint
    save_path = os.path.join(output_dir, "prism_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved successfully to {save_path}")

    # Also copy to submission directory
    sub_weights = os.path.join(BASE_DIR, "kaggle", "submission", "prism_model.pt")
    torch.save(model.state_dict(), sub_weights)
    print(f"Copied weights to submission package: {sub_weights}")


if __name__ == "__main__":
    train_prism(
        epochs=3,
        steps_per_epoch=20,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
