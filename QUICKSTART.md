# Quickstart

## 1. Configure model paths

The released GDN quickstart loads directly from:

```text
startlux-models/Massive-Activations-HLA
```

For local M-A-P checkpoints, modern models, or a local mirror of the released
GDN checkpoints, set environment variables and edit/copy the model config:

```bash
export MAP_MODEL_ROOT=/path/to/map/checkpoints
export RELEASED_GDN_ROOT=/path/to/released-gdn-checkpoints
export MODERN_MODEL_ROOT=/path/to/modern/models
```

You may also create `configs/local/local_paths.yaml` from the example file.
Local config files are ignored by git.

## 2. Run morphology on the Summer prompt

```bash
PYTHONPATH=src \
python scripts/run_morphology.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_pas_layer12 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --output outputs/quickstart/morphology
```

For an offline five-domain smoke test, replace `--prompts configs/datasets/prompts.yaml`
with:

```bash
--datasets configs/datasets/five_domains_tiny.yaml --limit 1
```

## 3. Compute ISP retention from captured traces

First capture an ISP model:

```bash
PYTHONPATH=src \
python scripts/run_morphology.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_isp_3to1 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --output outputs/quickstart/isp_morphology
```

Then compute the metric:

```bash
PYTHONPATH=src \
python scripts/run_pas_isp_metrics.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_isp_3to1 \
  --trace-root outputs/quickstart/isp_morphology \
  --prompt-names summer \
  --trace-token-index 0 \
  --sink-token 0 \
  --output outputs/quickstart/metrics.csv
```

For strict sink-token extraction from full-attention maps, additionally pass
`--attention-root /path/to/attention_maps`. The expected layout is:

```text
<attention-root>/<model>/<prompt>/layer_<L>.npy
```

where `L` is the 1-based full-attention layer index and each array has shape
`[heads, query, source]` or `[query, source]`.

## 4. Plot lifecycle atlas

```bash
PYTHONPATH=src \
python scripts/run_lifecycle_atlas.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_pas_layer12 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --output outputs/quickstart/lifecycle_atlas
```
