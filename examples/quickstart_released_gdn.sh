#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python scripts/run_morphology.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_pas_layer12 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --output outputs/quickstart/released_gdn_summer

PYTHONPATH=src python scripts/run_lifecycle_atlas.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_pas_layer12 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --output outputs/quickstart/released_gdn_lifecycle
