from pathlib import Path

from massive_activations_hla.models.registry import ModelSpec
from massive_activations_hla.models.released_gdn import (
    explicit_inference_code_path,
    full_attention_layers_from_checkpoint,
    is_gated_full_attention,
)


def test_full_attention_layers_from_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"attn": {"layers": [2, 5, 8]}}', encoding="utf-8")
    assert full_attention_layers_from_checkpoint(model_dir) == [3, 6, 9]


def test_gated_full_attention_is_explicitly_identified() -> None:
    gated = ModelSpec(name="gated", metadata={"variant": "gated_fa"})
    released_name = ModelSpec(name="gdn-gatedfa-340m-pas-fa-layer12-10b")
    baseline = ModelSpec(name="baseline")
    assert is_gated_full_attention(gated)
    assert is_gated_full_attention(released_name)
    assert not is_gated_full_attention(baseline)


def test_explicit_inference_code_path(tmp_path: Path) -> None:
    code = tmp_path / "compat"
    code.mkdir()
    spec = ModelSpec(name="gated", metadata={"inference_code": str(code)})
    assert explicit_inference_code_path(spec) == code


def test_local_sibling_code_is_not_auto_discovered(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    sibling = tmp_path / "_inference_code" / "gatedfa_fla_compat"
    sibling.mkdir(parents=True)
    spec = ModelSpec(name="baseline", path=str(model))
    assert explicit_inference_code_path(spec) is None
