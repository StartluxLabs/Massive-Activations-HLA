# Installation

Create a fresh Python environment:

```bash
conda create -n ma-hla python=3.10 -y
conda activate ma-hla
pip install -e .
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

Some M-A-P / GDN checkpoints require Flash Linear Attention (FLA) or custom
remote code. Install the matching FLA version used by your checkpoints.

For the released controlled GDN checkpoints used by the quickstart, the default
configuration loads from:

```text
startlux-models/Massive-Activations-HLA
```

If using a local mirror instead, point `RELEASED_GDN_ROOT` to the directory
containing both the checkpoint folders and:

```text
_inference_code/gatedfa_fla_compat
```

The loader will prepend this compatibility code to `PYTHONPATH` at runtime and
register the `gated_deltanet` architecture with Transformers.

This repository intentionally does not pin one universal environment for all
models: M-A-P, released GDN, Qwen3.5, Kimi-Linear, Zamba2, and Nemotron-H may
require different remote-code dependencies.
