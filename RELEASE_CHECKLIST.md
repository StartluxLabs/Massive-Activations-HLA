# Release checklist

Use this checklist before publishing the repository.

## Code health

- [x] Core package imports compile.
- [x] Unit tests cover toy PAS/ISP metrics, attention-map loading, and released
      GDN full-attention layer parsing.
- [x] Local private path config is excluded; use
      `configs/local/local_paths.example.yaml`.
- [x] No model checkpoints, datasets, generated outputs, or credentials are
      stored in the repository.

## Smoke tests

Validated commands:

- [x] `run_morphology.py --help`
- [x] `run_lifecycle_atlas.py --help`
- [x] `run_pas_isp_metrics.py --help`
- [x] released GDN 340M PAS morphology from a clean public-FLA environment
- [x] released GDN 340M PAS lifecycle atlas from a clean public-FLA environment
- [x] released GDN 340M ISP 3:1 morphology + PAS/ISP metrics from a clean environment
- [x] offline five-domain tiny morphology smoke test
- [x] offline five-domain tiny lifecycle smoke test
- [x] multi-model PAS-placement morphology smoke test
- [x] `--attention-root` end-to-end toy attention-map metrics test

Reference smoke metric (refresh after the clean public-FLA release test):

```text
gdn_340m_isp_3to1, summer, sink_mode=fixed:
sink_spike_alignment = 1.0
inter_spike_retention = 0.8708171248435974
```

## Before public release

- [ ] Add the final citation/BibTeX once the paper metadata is stable.
- [x] Released controlled GDN config points to
      `startlux-models/Massive-Activations-HLA` with checkpoint subfolders.
- [x] Run `scripts/check_environment.py` and the full quickstart from a fresh
      Python 3.12 environment on NVIDIA A800 CUDA hardware.
- [x] Verify `gdn-gatedfa-*` fails with the documented weights-only compatibility
      message when no explicit private adapter is configured.
- [ ] If publishing model checkpoints separately, verify license and model card
      text for each checkpoint.
