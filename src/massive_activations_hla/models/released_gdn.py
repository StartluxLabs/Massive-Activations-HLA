from __future__ import annotations

import json
import sys
from pathlib import Path

from massive_activations_hla.models.registry import ModelSpec


def default_inference_code_path(spec: ModelSpec) -> Path | None:
    """Infer the bundled FLA compatibility path for released GDN checkpoints."""
    root = spec.metadata.get("inference_code")
    if root:
        return Path(root).expanduser()
    if not spec.path:
        hf_subfolder = spec.metadata.get("inference_code_subfolder")
        if spec.hf_id and hf_subfolder:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("huggingface_hub is required to load released GDN code from HF.") from exc
            snapshot_root = snapshot_download(
                repo_id=spec.hf_id,
                allow_patterns=[f"{hf_subfolder}/**"],
            )
            return Path(snapshot_root) / hf_subfolder
        return None
    candidate = Path(spec.path).expanduser().parent / "_inference_code" / "gatedfa_fla_compat"
    return candidate if candidate.exists() else None


def register_released_gdn(spec: ModelSpec) -> None:
    """Register the custom `gated_deltanet` architecture with Transformers.

    The released controlled-pretraining checkpoints are intentionally stored as
    ordinary HF checkpoints, but their config uses `model_type=gated_deltanet`.
    They therefore need the accompanying FLA compatibility code to be importable
    before `AutoModelForCausalLM.from_pretrained` is called.
    """
    code_path = default_inference_code_path(spec)
    if code_path is None or not code_path.exists():
        raise FileNotFoundError(
            "Could not find released GDN inference code. Set metadata.inference_code "
            "or place `_inference_code/gatedfa_fla_compat` next to the checkpoint root."
        )
    code = str(code_path)
    if code not in sys.path:
        sys.path.insert(0, code)

    # Import side effects register GatedDeltaNetConfig and GatedDeltaNetForCausalLM.
    import fla.models.gated_deltanet  # noqa: F401


def full_attention_layers_from_checkpoint(path: str | Path) -> list[int]:
    """Read 1-based full-attention layer indices from a released GDN config."""
    config_path = Path(path).expanduser() / "config.json"
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    layers = ((config.get("attn") or {}).get("layers") or [])
    # Checkpoints store zero-based layer indices; paper figures use one-based.
    return [int(x) + 1 for x in layers]


def full_attention_layers_from_hf(repo_id: str, subfolder: str | None = None) -> list[int]:
    """Read 1-based full-attention layer indices from a HF checkpoint config."""
    from huggingface_hub import hf_hub_download

    filename = "config.json" if subfolder is None else f"{subfolder}/config.json"
    config_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return full_attention_layers_from_checkpoint(Path(config_path).parent)


def enrich_released_gdn_spec(spec: ModelSpec) -> ModelSpec:
    """Fill reliable metadata from the checkpoint config when available."""
    if spec.path:
        fa_layers = full_attention_layers_from_checkpoint(Path(spec.path) / spec.subfolder if spec.subfolder else spec.path)
    elif spec.hf_id:
        fa_layers = full_attention_layers_from_hf(spec.hf_id, spec.subfolder)
    else:
        fa_layers = []
    if fa_layers:
        spec.full_attention_layers = fa_layers
    return spec
