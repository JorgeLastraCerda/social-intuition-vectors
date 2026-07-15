#!/bin/bash
# Gemma 4 12B single-GPU/two-GPU TransformerBridge parity audit.
# Submit through submit_gemma4_12b_multigpu_parity.sh.
#$ -q gpu@scc192,gpu@scc213
#$ -l h_rt=01:00:00
#$ -l h_vmem=32G
#$ -pe smp 2
#$ -l gpu=2
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${RUN_ID:?RUN_ID is required}"
: "${STATE_DIR:?STATE_DIR is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"

module load conda  # ADJUST only if SCCKN changes its module name.
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing parity run: isolated SCCKN checkout is not clean." >&2
  exit 10
fi
if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing parity run: submitted commit is not an ancestor of checkout HEAD." >&2
  exit 11
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  src/utils/model_loader.py \
  smoke_tests/gemma4_transformerlens/parity_test_multigpu.py \
  jobs/sge/gemma4_12b_multigpu_parity.sh; then
  echo "Refusing parity run: parity implementation changed after submission." >&2
  exit 11
fi
if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "2" ]]; then
  echo "Refusing parity run: Grid Engine did not expose exactly two GPUs." >&2
  exit 12
fi

mkdir -p "$STATE_DIR" results/logs
OUTPUT="results/logs/gemma4_parity_12b_${RUN_ID}.json"

set +e
python smoke_tests/gemma4_transformerlens/parity_test_multigpu.py run \
  --vectors-subdir concept_vectors_gemma4_12b \
  --run-id "$RUN_ID" \
  --git-commit "$GIT_COMMIT" \
  --work-dir "$STATE_DIR" \
  --output "$OUTPUT"
parity_status=$?
set -e

if [[ ! -f "$OUTPUT" ]]; then
  echo "Parity runner did not produce its required decision JSON: $OUTPUT" >&2
  exit 13
fi

set +e
bash jobs/sync_outputs.sh "$REPO_PATH"
sync_status=$?
set -e
if ((sync_status != 0)); then
  echo "Parity JSON exists but Git output sync failed with status $sync_status." >&2
  exit "$sync_status"
fi
exit "$parity_status"
