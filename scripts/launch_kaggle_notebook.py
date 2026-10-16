"""Helper script to trigger save_notebook via Kaggle MCP or Kaggle API.
Reads notebooks/kaggle_training_pipeline.ipynb and formats the payload.
"""

import json
from pathlib import Path

def get_notebook_payload():
    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_training_pipeline.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_text = f.read()

    return {
        "newTitle": "Prism PTCG AI Training Pipeline",
        "slug": "prism-ptcg-ai-training-pipeline",
        "text": nb_text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": False,
        "enableGpu": True,
        "enableInternet": True,
        "competitionDataSources": ["the-pokemon-company-ptcg-ai-battle-challenge-playground"],
        "kernelExecutionType": "SaveAndRunAll"
    }

if __name__ == "__main__":
    payload = get_notebook_payload()
    print(f"Prepared payload for '{payload['newTitle']}' (length {len(payload['text'])} chars)")
