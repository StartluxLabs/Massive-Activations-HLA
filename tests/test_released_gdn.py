from pathlib import Path

from massive_activations_hla.models.released_gdn import full_attention_layers_from_checkpoint


def test_full_attention_layers_from_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"attn": {"layers": [2, 5, 8]}}', encoding="utf-8")
    assert full_attention_layers_from_checkpoint(model_dir) == [3, 6, 9]
