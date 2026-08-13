---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - fla
  - gated-deltanet
  - hybrid-linear-attention
  - massive-activations
---

# Massive Activations in Hybrid Linear Attention Models

Controlled-pretraining checkpoints accompanying **Massive Activations in Hybrid
Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike
Plateaus** ([arXiv:2608.12149](https://arxiv.org/abs/2608.12149)).

Use the [official analysis repository](https://github.com/StartluxLabs/Massive-Activations-HLA)
for installation, model registries, morphology traces, PAS/ISP metrics, and
lifecycle atlases.

## Compatibility and reproducibility scope

The baseline and `gdn-nooutgate-*` checkpoints load with the public, pinned
environment documented in the GitHub repository:

```bash
conda create -n ma-hla python=3.12 -y
conda activate ma-hla
bash scripts/install_released_gdn_cu126.sh
```

The two `gdn-gatedfa-*` checkpoints are **weights-only research artifacts**.
Their full-attention layers use a post-SDPA, head-specific sigmoid output gate
inspired by the G1 design in [Gated Attention for Large Language Models:
Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
([official code](https://github.com/qiuzh20/gated_attention)). The exact
GatedDeltaNet integration is not distributed. These two checkpoints are not
part of the public from-scratch quickstart.

The repository loader raises an explicit compatibility message for gated-FA
models unless a user deliberately configures a compatible local implementation.

## Loading a baseline checkpoint

Register the public FLA architecture before using Transformers directly:

```python
import fla.models.gated_deltanet  # registers the custom config/model
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "startlux-models/Massive-Activations-HLA"
subfolder = "gdn-340m-pas-fa-layer12-10b"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder=subfolder,
    torch_dtype="auto",
)
```

For analysis, prefer the GitHub registry and scripts because they validate the
FLA version and recover full-attention layer metadata consistently.

## Reproducibility notes

- Public FLA: v0.5.2, commit `9c8e42e762fce087c27b673af4922795d9edb85e`.
- Exact A800/CUDA 12.6 package versions are recorded in
  `requirements/released-gdn-cu126.txt` in the code repository.
- The Summer prompt and inline five-domain examples reproduce the public smoke
  pipeline. Exact input-level regeneration of every paper panel requires the
  sampled JSONL inputs used for that panel.
- Model weights are Apache-2.0; repository analysis code is MIT.

## Limitations

These are research checkpoints, not instruction-tuned or safety-tuned models.
They have not been validated for production use. The gated-FA variants require
an undistributed compatibility implementation as described above.

## Citation

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
