from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

from massive_activations_hla.models.registry import ModelSpec

GATED_ATTENTION_PAPER = "https://arxiv.org/abs/2505.06708"
GATED_ATTENTION_CODE = "https://github.com/qiuzh20/gated_attention"
PUBLIC_FLA_VERSION = "0.5.2"


def is_gated_full_attention(spec: ModelSpec) -> bool:
    if spec.metadata.get("variant") == "gated_fa":
        return True
    # Keep the guard effective for hand-written registries that preserve the
    # released checkpoint name but accidentally omit the metadata block.
    identifiers = (spec.name, spec.path, spec.hf_id, spec.subfolder)
    return any("gatedfa" in str(value).lower() for value in identifiers if value)


def explicit_inference_code_path(spec: ModelSpec) -> Path | None:
    """Return an explicitly configured private/local compatibility source."""
    root = spec.metadata.get("inference_code")
    if root:
        return Path(root).expanduser()
    # Do not auto-discover code next to a local mirror. Silent discovery makes
    # an environment appear reproducible while actually importing unpublished
    # cluster-local sources. Private adapters must always be opted into.
    return None


def _register_from_path(code_path: Path) -> None:
    if not code_path.exists():
        raise FileNotFoundError(f"Configured inference-code path does not exist: {code_path}")
    code = str(code_path)
    if code not in sys.path:
        sys.path.insert(0, code)
    import fla.models.gated_deltanet  # noqa: F401


def _register_public_fla() -> None:
    try:
        installed = metadata.version("flash-linear-attention")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Released baseline GDN checkpoints require flash-linear-attention=="
            f"{PUBLIC_FLA_VERSION}. Install with `pip install -e '.[released-gdn]'`."
        ) from exc
    if installed != PUBLIC_FLA_VERSION:
        raise ImportError(
            "Unsupported Flash Linear Attention version for the released GDN "
            f"checkpoints: found {installed}, expected {PUBLIC_FLA_VERSION}. "
            "Use the pinned released-GDN environment from INSTALL.md."
        )
    import fla.models.gated_deltanet  # noqa: F401


def register_released_gdn(spec: ModelSpec) -> None:
    """Register the custom `gated_deltanet` architecture with Transformers.

    The released controlled-pretraining checkpoints are intentionally stored as
    ordinary HF checkpoints, but their config uses `model_type=gated_deltanet`.
    They therefore need the accompanying FLA compatibility code to be importable
    before `AutoModelForCausalLM.from_pretrained` is called.
    """
    code_path = explicit_inference_code_path(spec)
    if code_path is not None:
        _register_from_path(code_path)
        return

    if is_gated_full_attention(spec):
        raise RuntimeError(
            f"{spec.name} uses the full-attention output-gate ablation. Its exact "
            "training-time compatibility implementation is not distributed in this "
            "repository, so this checkpoint is released as weights-only. The design "
            "follows the post-SDPA, head-specific sigmoid gate (G1) studied in "
            f"Gated Attention ({GATED_ATTENTION_PAPER}; official code: "
            f"{GATED_ATTENTION_CODE}). To inspect this checkpoint with your own "
            "compatible implementation, set metadata.inference_code explicitly."
        )

    _register_public_fla()


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
