# Reproduction guide

This project is organized by paper artifact rather than by exploratory scripts.

## Morphology figures

Use:

```bash
PYTHONPATH=src python scripts/run_morphology.py ...
```

Artifacts:

- Fig. 1: PAS/ISP morphology across hybrid ratios.
- Fig. 3: PAS across linear-attention architectures.
- Fig. 4: representative large-scale pretrained hybrid models.
- Fig. 17-20: appendix token/domain/ratio/modern-model morphology panels.

## PAS/ISP metrics

Use:

```bash
PYTHONPATH=src python scripts/run_pas_isp_metrics.py ...
```

Artifacts:

- Table 1: sink-spike alignment.
- Table 2: inter-spike retention.

The script computes:

- sink-spike alignment, i.e. whether a sink token reaches its maximum activation
  in the preceding linear-attention block at the pre-attention layer;
- inter-spike retention, i.e. how much activation is retained between adjacent
  PAS events.

Sink tokens can be provided in three ways:

1. `--attention-root`: strict mode. The script reads full-attention maps from
   `<attention-root>/<model>/<prompt>/layer_<L>.npy` or `.npz`, identifies the
   dominant source token by mean received causal attention, and uses it for PAS
   alignment.
2. `--sink-mode fixed --sink-token 0`: first-token analysis used by the
   lightweight quickstart and by first-token morphology checks.
3. `--sink-mode max_activation`: a control mode that uses each pre-attention
   layer's max-activation token when attention maps are unavailable.

Attention arrays should have shape `[heads, query, source]` or
`[query, source]`; `.npz` files may store the array under `attention`, `attn`,
or any single array key.

## Lifecycle atlas

Use:

```bash
PYTHONPATH=src python scripts/run_lifecycle_atlas.py ...
```

Artifacts:

- Fig. 6: localized write-sink-cancel lifecycle underlying PAS.
- Fig. 7: ISP as delayed cancellation.
- Fig. 14-16: fixed-coordinate PAS formation, sink coupling, cancellation.
- Fig. 21-28: systematic-outlier lifecycle across M-A-P and modern hybrids.

Lifecycle plots are produced by forward hooks over decoder blocks. The generic
adapter captures residual input, attention/linear-attention update, MLP update,
and block output when these modules are exposed by the model implementation.
