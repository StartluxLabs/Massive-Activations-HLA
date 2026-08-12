from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from massive_activations_hla.config import load_yaml
from massive_activations_hla.data.prompts import Prompt


def _load_jsonl(path: str | Path, text_key: str = "text", limit: int | None = None) -> list[Prompt]:
    out: list[Prompt] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            item = json.loads(line)
            out.append(Prompt(name=f"{Path(path).stem}_{i:04d}", text=str(item[text_key])))
    return out


def load_dataset_prompts(path: str | Path, limit: int | None = None) -> list[Prompt]:
    """Load evaluation inputs from a small config.

    Supported loaders:
      - inline: examples embedded in the yaml
      - jsonl: local JSONL with a text column
      - hf: HuggingFace datasets (optional dependency)
    """
    data = load_yaml(path)
    datasets = data.get("datasets", data)
    prompts: list[Prompt] = []
    for name, cfg in datasets.items():
        loader = cfg.get("loader", "inline")
        n = cfg.get("limit", limit)
        if limit is not None:
            n = min(limit, n) if n is not None else limit
        if loader == "inline":
            examples: Iterable[str] = cfg.get("examples", [])
            for i, text in enumerate(examples):
                if n is not None and i >= n:
                    break
                prompts.append(Prompt(name=f"{name}_{i:04d}", text=str(text), domain=name))
        elif loader == "jsonl":
            for p in _load_jsonl(cfg["path"], cfg.get("text_key", "text"), n):
                prompts.append(Prompt(name=f"{name}_{p.name}", text=p.text, domain=name))
        elif loader == "hf":
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError("Install datasets or use loader: jsonl/inline.") from exc
            ds = load_dataset(cfg["path"], cfg.get("name"), split=cfg.get("split", "validation"))
            text_key = cfg.get("text_key", "text")
            for i, item in enumerate(ds):
                if n is not None and i >= n:
                    break
                prompts.append(Prompt(name=f"{name}_{i:04d}", text=str(item[text_key]), domain=name))
        else:
            raise ValueError(f"Unknown dataset loader: {loader}")
    return prompts
