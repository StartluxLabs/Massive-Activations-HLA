from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from massive_activations_hla.config import load_yaml


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    domain: str | None = None


def load_prompts(path: str | Path) -> list[Prompt]:
    data = load_yaml(path)
    raw = data.get("prompts", data)
    prompts: list[Prompt] = []
    if isinstance(raw, dict):
        for name, item in raw.items():
            if isinstance(item, str):
                prompts.append(Prompt(name=name, text=item))
            else:
                prompts.append(
                    Prompt(
                        name=name,
                        text=str(item["text"]),
                        domain=item.get("domain"),
                    )
                )
    elif isinstance(raw, list):
        for item in raw:
            prompts.append(
                Prompt(
                    name=str(item["name"]),
                    text=str(item["text"]),
                    domain=item.get("domain"),
                )
            )
    else:
        raise TypeError(f"Unsupported prompt config format: {type(raw)!r}")
    return prompts
