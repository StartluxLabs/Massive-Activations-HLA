from __future__ import annotations

import numpy as np

from massive_activations_hla.analysis.attention_maps import load_attention_maps
from massive_activations_hla.analysis.pas_isp_metrics import sink_tokens_from_attention_maps


def test_load_attention_maps_and_sink_tokens(tmp_path):
    base = tmp_path / "model_a" / "summer"
    base.mkdir(parents=True)
    attn = np.zeros((2, 4, 4), dtype=np.float32)
    attn[:, 1:, 0] = 0.5
    np.save(base / "layer_03.npy", attn)

    maps = load_attention_maps(str(tmp_path), "model_a", "summer", [3])
    sinks = sink_tokens_from_attention_maps(maps)
    assert sorted(maps) == [3]
    assert sinks == {3: 0}
