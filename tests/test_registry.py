from __future__ import annotations

from pathlib import Path

from massive_activations_hla.models.registry import ModelRegistry


def test_registry_loads_released_gdn() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = ModelRegistry.from_yaml(root / "configs/models/released_gdn_models.yaml")
    spec = registry.get("gdn_340m_pas_layer12")
    assert spec.full_attention_layers == [12]
    assert spec.pre_attention_layers == [11]
    assert spec.metadata["group"] == "controlled_pas"
