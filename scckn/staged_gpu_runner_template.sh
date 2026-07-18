#!/usr/bin/env bash
# Copy this file into a project and replace the placeholder stage functions.

set -Eeuo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
STATE_ROOT="${STATE_ROOT:?STATE_ROOT is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
PIPELINE_GROUP="${PIPELINE_GROUP:?PIPELINE_GROUP must be common or scarce}"

[[ "$PIPELINE_GROUP" == common || "$PIPELINE_GROUP" == scarce ]] \
  || die "PIPELINE_GROUP must be common or scarce"

RUN_STATE_DIR="$STATE_ROOT/$RUN_ID/$PIPELINE_GROUP"
mkdir -p "$RUN_STATE_DIR"
cd "$PROJECT_ROOT"

current_stage="initialization"
current_temp_sentinel=""
on_error() {
  local line="$1"
  local command="$2"
  if [[ -n "$current_temp_sentinel" ]]; then
    rm -f "$current_temp_sentinel"
  fi
  echo "FAILED stage=$current_stage line=$line command=$command utc=$(date -u +%FT%TZ)" >&2
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

# ADJUST: replace these lists and matching run_<stage> functions.
COMMON_STAGES=(common_stage_1 common_stage_2)
SCARCE_STAGES=(scarce_stage_1 scarce_stage_2)

not_configured() {
  die "replace the placeholder stage functions before submission"
}

run_common_stage_1() { not_configured; }
run_common_stage_2() { not_configured; }
run_scarce_stage_1() { not_configured; }
run_scarce_stage_2() { not_configured; }

validate_before() {
  local stage="$1"
  : "# ADJUST: validate inputs/configuration for $stage"
}

validate_after() {
  local stage="$1"
  : "# ADJUST: validate outputs and metadata for $stage"
}

sync_outputs() {
  local stage="$1"
  : "# ADJUST: sync durable outputs for $stage, or document why no sync is needed"
}

run_checked_stage() {
  local stage="$1"
  local function_name="run_${stage}"
  local sentinel="$RUN_STATE_DIR/${stage}.success"
  local temporary_sentinel="$sentinel.tmp.$$"

  current_stage="$stage"
  current_temp_sentinel="$temporary_sentinel"
  if [[ -f "$sentinel" ]]; then
    echo "SKIP stage=$stage sentinel=$sentinel"
    return 0
  fi
  declare -F "$function_name" >/dev/null || die "missing function: $function_name"

  echo "START stage=$stage utc=$(date -u +%FT%TZ) host=$(hostname) job_id=${JOB_ID:-local}"
  validate_before "$stage"
  "$function_name"
  validate_after "$stage"
  sync_outputs "$stage"
  printf 'stage=%s\nrun_id=%s\nfinished_utc=%s\njob_id=%s\nhost=%s\n' \
    "$stage" "$RUN_ID" "$(date -u +%FT%TZ)" "${JOB_ID:-local}" "$(hostname)" \
    > "$temporary_sentinel"
  mv "$temporary_sentinel" "$sentinel"
  current_temp_sentinel=""
  echo "DONE stage=$stage sentinel=$sentinel"
}

if [[ "$PIPELINE_GROUP" == common ]]; then
  stages=("${COMMON_STAGES[@]}")
else
  stages=("${SCARCE_STAGES[@]}")
fi

for stage in "${stages[@]}"; do
  run_checked_stage "$stage"
done

echo "PIPELINE COMPLETE group=$PIPELINE_GROUP run_id=$RUN_ID utc=$(date -u +%FT%TZ)"
