#!/bin/bash
# Build the isolated Gemma 4 CUDA 12.4 environment on CCU.
set -euo pipefail

REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
VENV_PATH=${VENV_PATH:-/home/jovyan/.venvs/normalcy-gemma4-cu124}
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/bin/python3.11}

cd "$REPO_PATH"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Required Python 3.11 binary not found: $PYTHON_BIN" >&2
  exit 20
fi
"$PYTHON_BIN" -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124
"$VENV_PATH/bin/python" -m pip install -r jobs/ccu/requirements-gemma4.txt
"$VENV_PATH/bin/python" -m pip check
mkdir -p /home/jovyan/work/hf_cache /home/jovyan/work/normalcy-gemma4-state
"$VENV_PATH/bin/python" - <<'PY'
import json
from importlib import metadata
import torch

required = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.0",
    "transformer-lens": "3.5.1",
    "transformers": "5.13.0",
    "accelerate": "1.14.0",
}
observed = {name: metadata.version(name) for name in required}
if observed != required:
    raise SystemExit(f"Version mismatch: {observed!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Exactly one CUDA GPU is required.")
name = torch.cuda.get_device_name(0)
if "H100" not in name:
    raise SystemExit(f"Expected an H100, got {name!r}.")
print(json.dumps({"packages": observed, "gpu": name}, indent=2))
PY
"$VENV_PATH/bin/python" -m pip freeze > /home/jovyan/work/normalcy-gemma4-state/pip-freeze.txt
