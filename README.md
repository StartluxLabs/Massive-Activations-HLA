# Massive-Activations-HLA

Official code and reproducibility toolkit for the paper:

> **Massive Activations in Hybrid Linear Attention Large Language Models:
> Pre-Attention Spikes and Inter-Spike Plateaus**

This project studies **massive activations (MAs)** in hybrid linear attention
(HLA) language models. We find that sparse full-attention layers induce
architecture-aligned activation events: sharp **pre-attention spikes (PAS)** and,
when full attention becomes denser, persistent **inter-spike plateaus (ISP)**.

[Paper](https://arxiv.org/abs/2608.12149) ·
[Models on Hugging Face](https://huggingface.co/collections/startlux-models/massive-activations-in-hybrid-linear-attention-models) ·
[Quickstart](QUICKSTART.md) ·
[Reproduction guide](docs/reproduction.md) ·
[Installation](INSTALL.md)

A synchronized Hugging Face model-card source is maintained at
[`docs/HUGGINGFACE_MODEL_CARD.md`](docs/HUGGINGFACE_MODEL_CARD.md) so that model
compatibility claims stay aligned with this README.

## What this repository provides

This repository contains clean, paper-facing code for reproducing the main
analysis artifacts:

| Artifact | Script | What it reproduces |
|---|---|---|
| MA morphology | `scripts/run_morphology.py` | Layerwise token activation curves across models, domains, and hybrid ratios |
| PAS/ISP metrics | `scripts/run_pas_isp_metrics.py` | Sink-spike alignment and inter-spike retention |
| Lifecycle atlas | `scripts/run_lifecycle_atlas.py` | Residual / attention / MLP / block-output decomposition of systematic outliers |

Released controlled-pretraining checkpoints are hosted separately on
Hugging Face:

```text
startlux-models/Massive-Activations-HLA
```

The repository does **not** include model weights, datasets, generated figures,
or private paths.

## Key findings

The paper asks how massive activations behave once full attention is no longer
present at every layer. The main findings are:

- **MAs are architecture-aligned in HLA LLMs.** Their largest spikes occur
  immediately before full-attention layers rather than uniformly across depth.
- **PAS are robust across models and inputs.** The pre-attention spike pattern
  appears in controlled GDN models, M-A-P checkpoints, and several modern hybrid
  model families.
- **ISP reveal a denser-attention transition.** As full attention becomes more
  frequent, isolated PAS are connected by sustained activation plateaus.
- **Lifecycle atlases expose where outliers are written and canceled.** The
  provided atlas visualizations decompose each layer into residual input,
  sequence-mixing update, MLP update, and block output.

## Overview

Hybrid linear attention models are often introduced as efficient alternatives to
full-attention Transformers, but their activation dynamics are not simply
Transformer-like behavior with cheaper sequence mixing. In HLA LLMs, full
attention layers act as structural anchor points around which large hidden-state
outliers emerge and evolve.

The paper focuses on two characteristic MA morphologies:

- **Pre-attention spikes (PAS):** sharp activation spikes immediately before
  full-attention layers. PAS show that the largest activation events are
  aligned with the placement of full attention rather than appearing at arbitrary
  depth.
- **Inter-spike plateaus (ISP):** sustained activation plateaus connecting
  adjacent PAS. ISP appear as full attention becomes denser, revealing a gradual
  transition from isolated spikes to persistent, full-attention-like MA
  dynamics.

The toolkit is designed for three use cases:

1. **Reproduce the released-checkpoint analysis pipeline** for the public
   prompts and checkpoints described below.
2. **Inspect new HLA models** by tracing first-token or arbitrary-token MA
   trajectories.
3. **Compare architectures** using the same PAS/ISP metrics and visualization
   style across controlled GDN models, M-A-P checkpoints, and optional modern
   hybrid models.

## Installation

Create the tested released-GDN environment:

```bash
conda create -n ma-hla python=3.12 -y
conda activate ma-hla
python -m pip install --upgrade pip
bash scripts/install_released_gdn_cu126.sh
```

For HuggingFace datasets:

```bash
pip install -r requirements/datasets.txt
```

The installation script uses the exact tested PyTorch wheel:

```bash
torch==2.7.1+cu126
```

It installs the official `causal-conv1d` and `flash-attn` PyTorch 2.7/CUDA 12
wheels, verifies their checksums, and then installs FLA.

The released baseline and no-output-gate GDN checkpoints are tested with
`flash-linear-attention==0.5.2` (upstream tag `v0.5.2`, commit
`9c8e42e762fce087c27b673af4922795d9edb85e`). For the exact A800/CUDA 12.6
environment, follow [INSTALL.md](INSTALL.md).

### Full-attention-gate ablation scope

The two `gdn-gatedfa-*` checkpoints use a post-SDPA, head-specific sigmoid
output gate inspired by the **G1** design in [Gated Attention for Large Language
Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
([official code](https://github.com/qiuzh20/gated_attention)). Our Gated
DeltaNet integration of that gate is not distributed in this repository. Those
two checkpoints are therefore released as **weights-only research artifacts**
and are not part of the public, from-scratch quickstart. The loader fails with
an explicit compatibility message instead of silently loading them incorrectly.

All baseline and `gdn-nooutgate-*` checkpoints are covered by the public FLA
environment and are reproducible from the published code and weights.

## Quickstart: plot a PAS morphology curve

The following command loads the released 340M GDN checkpoint from HuggingFace
and plots the first-token MA trajectory for the running example
`Summer is warm. Winter is cold.`

```bash
PYTHONPATH=src python scripts/run_morphology.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_pas_layer12 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --max-length 32 \
  --output outputs/quickstart/morphology
```

Outputs are written to:

```text
outputs/quickstart/morphology/gdn_340m_pas_layer12/summer/
```

For a smoke test that does not download datasets, use:

```bash
--datasets configs/datasets/five_domains_tiny.yaml --limit 1
```

This is an **offline dataset-input** test, not a fully offline model test: the
checkpoint is still downloaded from Hugging Face unless it is already cached or
you configure a local model path.

## Compute PAS/ISP metrics

First capture an ISP model:

```bash
PYTHONPATH=src python scripts/run_morphology.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_isp_3to1 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --max-length 32 \
  --output outputs/quickstart/isp_morphology
```

Then compute the metrics:

```bash
PYTHONPATH=src python scripts/run_pas_isp_metrics.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_isp_3to1 \
  --trace-root outputs/quickstart/isp_morphology \
  --prompt-names summer \
  --trace-token-index 0 \
  --sink-token 0 \
  --output outputs/quickstart/metrics.csv
```

The metrics script reports:

- `sink_spike_alignment`: fraction of full-attention events whose sink token
  reaches its maximum activation at the corresponding pre-attention layer;
- `inter_spike_retention`: fraction of the adjacent PAS reference level retained
  across intervening linear-attention layers.

When full-attention maps are available, pass `--attention-root` to extract sink
tokens directly from attention:

```text
<attention-root>/<model>/<prompt>/layer_<L>.npy
```

where `L` is the 1-based full-attention layer index and each array has shape
`[heads, query, source]` or `[query, source]`.

## Plot a lifecycle atlas

```bash
PYTHONPATH=src python scripts/run_lifecycle_atlas.py \
  --models configs/models/released_gdn_models.yaml \
  --model-names gdn_340m_pas_layer12 \
  --prompts configs/datasets/prompts.yaml \
  --token-index 0 \
  --max-length 32 \
  --output outputs/quickstart/lifecycle_atlas
```

The atlas captures and visualizes the signed contribution of:

- residual input;
- attention / linear-attention update;
- MLP update;
- block output.

## Supported model registries

| Registry | Purpose |
|---|---|
| `configs/models/released_gdn_models.yaml` | Released controlled GDN checkpoints from HuggingFace |
| `configs/models/map_models.yaml` | Public M-A-P / controlled architecture checkpoints |
| `configs/models/modern_hybrid_models.yaml` | Optional modern pretrained hybrid models |

The released GDN registry points to:

```text
startlux-models/Massive-Activations-HLA
```

with one checkpoint per subfolder. If you use a local mirror, copy the config and
replace `hf_id` / `subfolder` with local `path` entries.

## Dataset inputs

The paper evaluates inputs from five regimes: general prose, scientific text,
math reasoning, code, and multilingual text. This repository provides:

- `configs/datasets/prompts.yaml`: the running Summer example;
- `configs/datasets/five_domains.yaml`: HuggingFace dataset loaders;
- `configs/datasets/five_domains_tiny.yaml`: offline smoke-test prompts.

The repository does not publish the exact sampled JSONL files used for every
paper panel. `five_domains.yaml` reproduces the public loading protocol, while
exact input-level reproduction requires the corresponding sampled JSONL files.

## Repository layout

```text
configs/
  datasets/     # prompt and dataset configs
  local/        # ignored local path overrides
  models/       # model registries
docs/
  reproduction.md
examples/
  quickstart_released_gdn.sh
  quickstart_summer_map.sh
scripts/
  run_lifecycle_atlas.py
  run_morphology.py
  run_pas_isp_metrics.py
src/massive_activations_hla/
  analysis/     # metrics and trace loading
  capture/      # model loading and forward hooks
  models/       # registry and released-GDN adapter
  plotting/     # paper-style visualizations
tests/
```

## Notes and limitations

- Large modern hybrid models may require substantial GPU memory and
  model-specific environments.
- Baseline and no-output-gate GDN checkpoints use public FLA v0.5.2. The
  full-attention-gate ablation is weights-only as documented above.
- `five_domains_tiny.yaml` is only a smoke-test input set; it is not a substitute
  for the full paper evaluation.
- This repository is organized for analysis and visualization, not for training
  new models from scratch.

## Licenses

The analysis code in this repository is MIT licensed. Released model weights
are Apache-2.0 licensed in the Hugging Face repository. Flash Linear Attention
and all datasets retain their own licenses and terms.

## Citation

If you use this repository, please cite the
[paper](https://arxiv.org/abs/2608.12149):

```bibtex
@article{su2026massive,
  title={Massive Activations in Hybrid Linear Attention Large Language Models:
         Pre-Attention Spikes and Inter-Spike Plateaus},
  author={Su, Zunhai and Sun, Bohan and Zhuang, Xialie and Zhang, Shuibai and
          Xiao, He and Xiong, Jing and Zhang, Hengyuan and Zhou, Zhongzhu and
          Zhang, Tiantian and Wong, Ngai and Kuo, Chuan-Wei},
  journal={arXiv preprint arXiv:2608.12149},
  year={2026}
}
```
