from __future__ import annotations

import sys
import types

from massive_activations_hla.capture.model_loading import load_tokenizer, register_map_architecture
from massive_activations_hla.models.registry import ModelSpec


def test_tokenizer_omits_none_subfolder(monkeypatch) -> None:
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return object()

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = FakeAutoTokenizer
    fake.PreTrainedTokenizerFast = object
    monkeypatch.setitem(sys.modules, "transformers", fake)

    load_tokenizer("local-model", subfolder=None)
    assert calls == [("local-model", {"trust_remote_code": True})]


def test_map_architecture_import_mapping(monkeypatch) -> None:
    imported = []
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.5.2")
    monkeypatch.setattr("importlib.import_module", imported.append)
    register_map_architecture(ModelSpec(name="gla", family="GLA", adapter="map"))
    assert imported == ["fla.models.gla"]
