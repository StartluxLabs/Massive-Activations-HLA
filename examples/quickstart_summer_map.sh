#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python scripts/run_morphology.py \
  --models configs/models/map_models.yaml \
  --model-names gdn_1p3b_12to1 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --output outputs/quickstart/map_summer
