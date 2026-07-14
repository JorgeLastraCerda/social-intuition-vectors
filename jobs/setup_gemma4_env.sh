#!/bin/bash
# Run once on the SCCKN login node. Cloning wc-tl preserves its working,
# CUDA-matched PyTorch build while isolating the Gemma 4 dependency upgrade.
set -euo pipefail

module load conda  # ADJUST: exact SCCKN module name/version if required
conda create --yes --name wc-tl-g4 --clone wc-tl
conda activate wc-tl-g4
# The Gemma 4 run deliberately excludes SAE/nnsight paths; removing them also
# prevents their older TransformerLens/Transformers constraints from contaminating
# the dedicated environment.
python -m pip uninstall --yes sae-lens nnsight nnterp || true
python -m pip install --upgrade -r requirements-gemma4.txt
python -m pip check
python - <<'PY'
from importlib.metadata import version
import torch, transformers
from transformer_lens.model_bridge import TransformerBridge
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("transformer-lens", version("transformer-lens"))
print("TransformerBridge", TransformerBridge.__name__)
PY
