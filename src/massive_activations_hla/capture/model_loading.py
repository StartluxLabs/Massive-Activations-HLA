from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from massive_activations_hla.models.registry import ModelSpec


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    tokenizer: Any


def torch_dtype_from_name(name: str) -> Any:
    import torch

    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


def load_model(spec: ModelSpec) -> LoadedModel:
    """Load a causal LM and tokenizer from a ModelSpec.

    This thin wrapper intentionally keeps model-specific logic out of scripts.
    Modern hybrid models often need `trust_remote_code=True`.
    """
    if spec.adapter == "released_gdn":
        from massive_activations_hla.models.released_gdn import (
            enrich_released_gdn_spec,
            register_released_gdn,
        )

        register_released_gdn(spec)
        spec = enrich_released_gdn_spec(spec)

    from transformers import AutoModelForCausalLM

    model_id = spec.model_id_or_path
    tokenizer = load_tokenizer(
        model_id,
        trust_remote_code=spec.trust_remote_code,
        subfolder=spec.subfolder,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        subfolder=spec.subfolder,
        torch_dtype=torch_dtype_from_name(spec.torch_dtype),
        trust_remote_code=spec.trust_remote_code,
        device_map=spec.device_map,
    ).eval()
    return LoadedModel(spec=spec, model=model, tokenizer=tokenizer)


def load_tokenizer(
    model_id_or_path: str,
    trust_remote_code: bool = True,
    subfolder: str | None = None,
) -> Any:
    """Load a tokenizer with a small fallback for locally released checkpoints.

    Some controlled-pretraining checkpoints were exported with
    `tokenizer_class=TokenizersBackend`, which is not a Transformers tokenizer
    class. They still contain a standard `tokenizer.json`, so we can recover a
    regular fast tokenizer without requiring users to edit the checkpoint.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
        )
    except ValueError as exc:
        model_path = Path(model_id_or_path).expanduser()
        if subfolder is not None:
            model_path = model_path / subfolder
        tokenizer_json = model_path / "tokenizer.json"
        config_path = model_path / "tokenizer_config.json"

        if not tokenizer_json.exists() and not model_path.exists():
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as hub_exc:
                raise ImportError("huggingface_hub is required for HF tokenizer fallback.") from hub_exc
            try:
                tokenizer_file = "tokenizer.json" if subfolder is None else f"{subfolder}/tokenizer.json"
                config_file = (
                    "tokenizer_config.json"
                    if subfolder is None
                    else f"{subfolder}/tokenizer_config.json"
                )
                tokenizer_json = Path(hf_hub_download(repo_id=model_id_or_path, filename=tokenizer_file))
                config_path = Path(hf_hub_download(repo_id=model_id_or_path, filename=config_file))
            except Exception:
                # If the Hub fallback fails, re-raise the original tokenizer error
                # because it contains the most helpful Transformers message.
                raise exc from None

        if not tokenizer_json.exists() or "Tokenizer class" not in str(exc):
            raise

        kwargs: dict[str, Any] = {"tokenizer_file": str(tokenizer_json)}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
                if key in config:
                    kwargs[key] = config[key]
            if "model_max_length" in config:
                kwargs["model_max_length"] = config["model_max_length"]
        return PreTrainedTokenizerFast(**kwargs)
