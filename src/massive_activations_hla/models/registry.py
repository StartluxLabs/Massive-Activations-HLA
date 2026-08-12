from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from massive_activations_hla.config import load_yaml


@dataclass
class ModelSpec:
    name: str
    path: str | None = None
    hf_id: str | None = None
    subfolder: str | None = None
    family: str | None = None
    adapter: str = "auto"
    scale: str | None = None
    num_layers: int | None = None
    full_attention_layers: list[int] = field(default_factory=list)
    hybrid_ratio: str | None = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id_or_path(self) -> str:
        if self.path:
            return self.path
        if self.hf_id:
            return self.hf_id
        raise ValueError(f"Model {self.name} has neither path nor hf_id.")

    @property
    def pre_attention_layers(self) -> list[int]:
        return [x - 1 for x in self.full_attention_layers if x > 1]

    def is_full_attention_layer(self, layer_index: int) -> bool:
        return layer_index in set(self.full_attention_layers)


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec]):
        self.specs = specs

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelRegistry:
        data = load_yaml(path)
        raw = data.get("models", data)
        specs: dict[str, ModelSpec] = {}
        for name, item in raw.items():
            item = dict(item or {})
            metadata = item.pop("metadata", {})
            known = {
                "path",
                "hf_id",
                "subfolder",
                "family",
                "adapter",
                "scale",
                "num_layers",
                "full_attention_layers",
                "hybrid_ratio",
                "trust_remote_code",
                "torch_dtype",
                "device_map",
            }
            extra = {k: item.pop(k) for k in list(item) if k not in known}
            metadata.update(extra)
            specs[name] = ModelSpec(name=name, metadata=metadata, **item)
        return cls(specs)

    def get(self, name: str) -> ModelSpec:
        return self.specs[name]

    def select(self, names: list[str] | None = None, group: str | None = None) -> list[ModelSpec]:
        if names:
            return [self.get(x) for x in names]
        if group:
            return [s for s in self.specs.values() if s.metadata.get("group") == group]
        return list(self.specs.values())
