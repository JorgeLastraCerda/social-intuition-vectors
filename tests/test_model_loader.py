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
    calls = []

    def boot_transformers(*args, **kwargs):
        calls.append((args, kwargs))
        return bridge_model

    bridge_class = type(
        "FakeBridge",
        (),
        {"boot_transformers": staticmethod(boot_transformers)},
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
    return calls


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


def test_single_device_loader_keeps_existing_device_argument(monkeypatch):
    model = SimpleNamespace(processor=None)
    calls = install_fake_modules(monkeypatch, model, AssertionError("must not load"))

    load_hooked_model(config_for("Qwen/Qwen3-14B"), n_devices=1)

    assert calls[0][1]["device"] == "cpu"
    assert "n_devices" not in calls[0][1]


def test_multi_device_loader_uses_n_devices_without_device(monkeypatch):
    model = SimpleNamespace(processor=None)
    calls = install_fake_modules(monkeypatch, model, AssertionError("must not load"))

    load_hooked_model(config_for("Qwen/Qwen3-14B"), n_devices=2)

    assert calls[0][1]["n_devices"] == 2
    assert "device" not in calls[0][1]


def test_loader_rejects_nonpositive_n_devices(monkeypatch):
    model = SimpleNamespace(processor=None)
    install_fake_modules(monkeypatch, model, AssertionError("must not load"))

    with pytest.raises(ValueError, match="at least 1"):
        load_hooked_model(config_for("Qwen/Qwen3-14B"), n_devices=0)
