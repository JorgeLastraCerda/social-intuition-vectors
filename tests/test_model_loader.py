import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.utils.config import ModelConfig
from src.utils.model_loader import load_hooked_model


def config_for(model_name: str):
    return SimpleNamespace(
        model=ModelConfig(
            name=model_name,
            dtype="bfloat16",
            device="cpu",
            backend="transformer-bridge",
        )
    )


def install_fake_modules(monkeypatch, bridge_model, processor_or_error):
    bridge_class = type(
        "FakeBridge",
        (),
        {"boot_transformers": staticmethod(lambda *args, **kwargs: bridge_model)},
    )
    bridge_module = ModuleType("transformer_lens.model_bridge")
    bridge_module.TransformerBridge = bridge_class
    package = ModuleType("transformer_lens")
    package.model_bridge = bridge_module
    monkeypatch.setitem(sys.modules, "transformer_lens", package)
    monkeypatch.setitem(sys.modules, "transformer_lens.model_bridge", bridge_module)

    transformers = ModuleType("transformers")
    if isinstance(processor_or_error, Exception):
        class AutoProcessor:
            @staticmethod
            def from_pretrained(model_name):
                raise processor_or_error
    else:
        class AutoProcessor:
            @staticmethod
            def from_pretrained(model_name):
                return processor_or_error
    transformers.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_gemma4_loader_explicitly_recovers_missing_processor(monkeypatch):
    model = SimpleNamespace(processor=None)
    processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
    install_fake_modules(monkeypatch, model, processor)
    loaded = load_hooked_model(config_for("google/gemma-4-31B-it"))
    assert loaded.processor is processor


def test_gemma4_loader_keeps_valid_bridge_processor(monkeypatch):
    processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
    model = SimpleNamespace(processor=processor)
    install_fake_modules(monkeypatch, model, AssertionError("must not reload"))
    loaded = load_hooked_model(config_for("google/gemma-4-12B-it"))
    assert loaded.processor is processor


def test_gemma4_loader_surfaces_processor_exception(monkeypatch):
    model = SimpleNamespace(processor=None)
    install_fake_modules(monkeypatch, model, ValueError("processor failed"))
    with pytest.raises(RuntimeError, match="required Gemma 4 processor") as error:
        load_hooked_model(config_for("google/gemma-4-31B-it"))
    assert isinstance(error.value.__cause__, ValueError)


def test_non_gemma4_loader_does_not_require_processor(monkeypatch):
    model = SimpleNamespace(processor=None)
    install_fake_modules(monkeypatch, model, AssertionError("must not load"))
    assert load_hooked_model(config_for("Qwen/Qwen3-14B")) is model
