"""Packages Prism submission into submission.tar.gz for Kaggle competition."""

import os
import sys
import shutil
import tarfile

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

STAGING_DIR = os.path.join(BASE_DIR, "build_submission")
OUTPUT_TAR = os.path.join(BASE_DIR, "submission.tar.gz")


def package_submission():
    print(f"Preparing submission bundle from {BASE_DIR}...")
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)
    os.makedirs(STAGING_DIR)

    # 1. Copy main.py and deck.csv to top-level
    shutil.copy(
        os.path.join(BASE_DIR, "submission", "main.py"),
        os.path.join(STAGING_DIR, "main.py"),
    )
    shutil.copy(
        os.path.join(BASE_DIR, "submission", "deck.csv"),
        os.path.join(STAGING_DIR, "deck.csv"),
    )

    # Copy weights if available
    weights_src = os.path.join(BASE_DIR, "submission", "prism_model.pt")
    if os.path.exists(weights_src):
        shutil.copy(weights_src, os.path.join(STAGING_DIR, "prism_model.pt"))
    else:
        # Create initial weights if not trained yet
        import torch
        from models.actor_critic import PrismActorCritic
        dummy_model = PrismActorCritic()
        torch.save(dummy_model.state_dict(), os.path.join(STAGING_DIR, "prism_model.pt"))
        print("Generated baseline prism_model.pt weights.")

    # 2. Copy models/, engine/, and inference/ modules
    shutil.copytree(
        os.path.join(BASE_DIR, "models"),
        os.path.join(STAGING_DIR, "models"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        os.path.join(BASE_DIR, "engine"),
        os.path.join(STAGING_DIR, "engine"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "target"),
    )
    shutil.copytree(
        os.path.join(BASE_DIR, "inference"),
        os.path.join(STAGING_DIR, "inference"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    # 3. Create .tar.gz bundle
    print(f"Creating {OUTPUT_TAR}...")
    with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
        for item in os.listdir(STAGING_DIR):
            item_path = os.path.join(STAGING_DIR, item)
            tar.add(item_path, arcname=item)

    tar_size_mb = os.path.getsize(OUTPUT_TAR) / (1024 * 1024)
    print(f"Submission bundle created successfully: {OUTPUT_TAR} ({tar_size_mb:.2f} MB)")

    # Clean up staging dir
    shutil.rmtree(STAGING_DIR)


if __name__ == "__main__":
    package_submission()
