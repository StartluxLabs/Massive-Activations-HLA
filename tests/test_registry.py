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


def test_modern_registry_prefers_hf_when_no_local_root() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = ModelRegistry.from_yaml(root / "configs/models/modern_hybrid_models.yaml")
    spec = registry.get("qwen35_35b_a3b")
    assert spec.path is None
    assert spec.model_id_or_path == "Qwen/Qwen3.5-35B-A3B"


def test_map_registry_points_to_public_hf_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = ModelRegistry.from_yaml(root / "configs/models/map_models.yaml")
    spec = registry.get("gdn_1p3b_12to1")
    assert spec.path is None
    assert spec.model_id_or_path == "m-a-p/1.3B-100B-GatedDeltaNet-hybrid-12-1"
