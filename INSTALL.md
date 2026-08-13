# Installation

## Released GDN checkpoints (tested path)

Linux with an NVIDIA GPU is required for the released GDN quickstart. The
following path was tested on NVIDIA A800 GPUs with CUDA 12.6:

```bash
conda create -n ma-hla python=3.12 -y
conda activate ma-hla
python -m pip install --upgrade pip
bash scripts/install_released_gdn_cu126.sh
```

For dataset loading:

```bash
pip install -r requirements/datasets.txt
```

For modern hybrid models, install optional dependencies and model-specific
requirements as needed:

```bash
pip install -r requirements/modern.txt
```

The public baseline/no-output-gate environment pins Flash Linear Attention
v0.5.2. A fully enumerated A800 reference environment is recorded in
`requirements/released-gdn-cu126.txt`. The installation script enforces the
required order, uses the official `causal-conv1d` and `flash-attn` PyTorch 2.7
wheels with checksum verification, and installs FLA after Torch is present.

The default configuration loads checkpoints from:

```text
startlux-models/Massive-Activations-HLA
```

For a local mirror, copy `configs/models/released_gdn_models.yaml`, replace each
`hf_id`/`subfolder` pair with a direct `path`, and pass your copied registry to
the scripts. The loader handles local paths without requiring a subfolder.

The M-A-P registry loads its public `m-a-p/*` checkpoints directly from Hugging
Face. For an offline mirror, copy the registry and replace each `hf_id` with a
local `path`. The `map` adapter registers each checkpoint's public FLA v0.5.2
architecture before Transformers loads it.

Legacy M-A-P GatedDeltaNet checkpoints contain all-zero `attn.D` tensors that
are absent from current public FLA. Transformers may report them as unused; all
published copies checked for this release have `max(abs(D)) == 0`, so dropping
them does not change the forward computation.

## Full-attention-gate ablation

`gdn-gatedfa-*` uses a post-SDPA, head-specific sigmoid output gate inspired by
Gated Attention (arXiv:2505.06708; official implementation:
https://github.com/qiuzh20/gated_attention). The exact Gated DeltaNet integration
is not distributed, so these checkpoints are weights-only and excluded from the
public quickstart. Users with a compatible implementation can explicitly set
`metadata.inference_code` in a local registry.

The pinned released-GDN environment also supports the listed M-A-P checkpoints.
Qwen3.5, Kimi-Linear, Zamba2, and Nemotron-H may require separate model-specific
remote-code dependencies.
