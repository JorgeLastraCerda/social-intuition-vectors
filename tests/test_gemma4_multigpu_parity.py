import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from smoke_tests.gemma4_transformerlens import parity_test_multigpu as parity


class FakeTensor:
    def __init__(self, device):
        self.device = device

    def numel(self):
        return 1


class FakeLayer:
    def __init__(self, *devices):
        self._parameters = [FakeTensor(device) for device in devices]

    def parameters(self):
        return iter(self._parameters)

    def buffers(self):
        return iter(())


class FakeOriginalModel:
    def __init__(self, layers, device_map=None):
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=layers)
        )
        self.hf_device_map = device_map

    def parameters(self):
        return iter(
            [parameter for layer in self.model.language_model.layers for parameter in layer._parameters]
        )


def test_topology_accepts_whole_layers_on_both_gpus():
    original = FakeOriginalModel(
        [FakeLayer("cuda:0"), FakeLayer("cuda:1")],
        {"model.language_model.layers.0": 0, "model.language_model.layers.1": 1},
    )

    result = parity.inspect_topology(original, 2)

    assert result["pass"] is True
    assert result["layer_assignment_counts"] == {"cuda:0": 1, "cuda:1": 1}


def test_topology_rejects_a_split_layer():
    original = FakeOriginalModel(
        [FakeLayer("cuda:0", "cuda:1"), FakeLayer("cuda:1")],
        {"model.language_model.layers.0": 0, "model.language_model.layers.1": 1},
    )

    result = parity.inspect_topology(original, 2)

    assert result["pass"] is False
    assert any("spans" in failure for failure in result["failures"])


def write_capture(root: Path, arm: str, activation, logits=None):
    snapshot = root / f"{arm}.npz"
    metadata = root / f"{arm}.json"
    if logits is None:
        logits = np.array([1.0, 2.0], dtype=np.float32)
    np.savez(
        snapshot,
        tokens__smoke=np.array([[1, 2, 3]], dtype=np.int64),
        activation__smoke__layer_00=np.asarray(activation, dtype=np.float32),
        logits__smoke=np.asarray(logits, dtype=np.float32),
    )
    metadata.write_text(
        json.dumps(
            {
                "model": "google/gemma-4-12B-it",
                "seed": 20260527,
                "probe_layer": 31,
                "input_records": [{"id": "smoke-passage"}],
                "mean_resid_norm": 42.0,
                "steering_margins": {"warmth:+0.05": 1.25},
                "topology": {"pass": True},
                "bridge_hf_max_logit_diff": 0.0,
                "peak_allocated_vram_gib": {
                    "cuda:0": 1.0,
                    "cuda:1": 1.0,
                },
                "runtime": {
                    "transformer_lens_version": "3.5.1",
                    "transformers_version": "5.13.0",
                    "torch_version": "test",
                    "dtype": "torch.bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    return snapshot, metadata


def test_compare_captures_accepts_exact_parity(tmp_path):
    paths = {
        arm: write_capture(tmp_path, arm, [1.0, 2.0, 3.0])
        for arm in ("single_a", "single_b", "multi")
    }

    result = parity.compare_captures(paths, run_id="test", git_commit="abc")

    assert result["status"] == "pass"
    assert result["failures"] == []


def test_compare_captures_rejects_activation_drift(tmp_path):
    paths = {
        "single_a": write_capture(tmp_path, "single_a", [1.0, 2.0, 3.0]),
        "single_b": write_capture(tmp_path, "single_b", [1.0, 2.0, 3.0]),
        "multi": write_capture(tmp_path, "multi", [10.0, 20.0, 30.0]),
    }

    result = parity.compare_captures(paths, run_id="test", git_commit="abc")

    assert result["status"] == "fail"
    assert any("numeric parity gate failed" in failure for failure in result["failures"])


def test_run_writes_failure_json_when_capture_process_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        parity.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=17,
            stdout="capture stdout",
            stderr="exact capture failure",
        ),
    )
    output = tmp_path / "decision.json"
    args = Namespace(
        output=str(output),
        work_dir=str(tmp_path / "work"),
        config="config/config.yaml",
        vectors_subdir="concept_vectors_gemma4_12b",
        seed=20260527,
        run_id="test-run",
        git_commit="abc",
    )

    status = parity.run(args)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert status == 1
    assert result["status"] == "fail"
    assert result["capture_failures"][0]["return_code"] == 17
    assert result["capture_failures"][0]["stderr_tail"] == "exact capture failure"
