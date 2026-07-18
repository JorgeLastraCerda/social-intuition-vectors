#!/bin/bash
# Create or verify the isolated SCCKN environment for native-HF Qwen3.6 hooks.
set -euo pipefail

ENV_NAME="wc-qwen36-hf"
SOURCE_ENV="wc-tl-g4"

module load conda  # ADJUST if SCCKN changes its module name.
cd "$(git rev-parse --show-toplevel)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] $ENV_NAME already exists; verifying without replacing it"
else
  conda create --yes --name "$ENV_NAME" --clone "$SOURCE_ENV"
  conda run -n "$ENV_NAME" python -m pip uninstall --yes \
    transformer-lens nnsight sae-lens nnterp || true
  conda run -n "$ENV_NAME" python -m pip install --upgrade \
    -r requirements-qwen36.txt
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip check
conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import importlib.util
from importlib.metadata import version

import PIL
import torch
import torchvision
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

assert transformers.__version__ == "5.14.1", transformers.__version__
assert importlib.util.find_spec("transformer_lens") is None
print("python environment: wc-qwen36-hf")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("accelerate", version("accelerate"))
print("pillow", PIL.__version__)
print("classes", AutoProcessor.__name__, AutoModelForMultimodalLM.__name__)
PY
