# Quickstart

## 1. Configure model paths

The released GDN quickstart loads directly from:

```text
startlux-models/Massive-Activations-HLA
```

The public M-A-P registry also loads directly from Hugging Face. For an offline
M-A-P mirror or local modern models, copy the registry and replace `hf_id` with
local `path` entries. Modern-model roots can be provided with:

```bash
export MODERN_MODEL_ROOT=/path/to/modern/models
```

For a local mirror of released GDN checkpoints, copy
`configs/models/released_gdn_models.yaml` and replace `hf_id`/`subfolder` with
direct `path` entries. `configs/local/local_paths.example.yaml` is a template,
not an automatically loaded configuration file.

Run `python scripts/check_environment.py` before the first quickstart. The
public quickstart supports baseline and no-output-gate checkpoints. The
`gdn-gatedfa-*` weights require an undistributed full-attention-gate integration;
see README.md for scope and attribution.

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
