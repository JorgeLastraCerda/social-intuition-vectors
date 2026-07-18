#!/usr/bin/env bash
# Copy this file into a project and resolve every # ADJUST value before use.

set -euo pipefail

usage() {
  echo "Usage: $0 [--dry-run]" >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) usage; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-# ADJUST: absolute project root on SCCKN}"
RUNNER="${RUNNER:-# ADJUST: absolute path to the staged runner}"
STATE_ROOT="${STATE_ROOT:-# ADJUST: durable directory for run state}"
PRIORITY="${PRIORITY:-# ADJUST: explicit project priority}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/sge}"

# Whitespace-delimited qsub arguments. Keep values free of embedded spaces.
COMMON_QSUB_ARGS="${COMMON_QSUB_ARGS:-# ADJUST: queue and resources for common GPU work}"
SCARCE_QSUB_ARGS="${SCARCE_QSUB_ARGS:-# ADJUST: queue and resources for scarce GPU work}"
COMMON_ENABLED="${COMMON_ENABLED:-1}"
SCARCE_ENABLED="${SCARCE_ENABLED:-1}"
SCARCE_DEPENDS_ON_COMMON="${SCARCE_DEPENDS_ON_COMMON:-0}"

for value in "$PROJECT_ROOT" "$RUNNER" "$STATE_ROOT" "$PRIORITY"; do
  [[ "$value" != *"# ADJUST"* ]] || die "resolve all # ADJUST values"
done
if [[ "$COMMON_ENABLED" == 1 ]]; then
  [[ "$COMMON_QSUB_ARGS" != *"# ADJUST"* ]] || die "set COMMON_QSUB_ARGS"
fi
if [[ "$SCARCE_ENABLED" == 1 ]]; then
  [[ "$SCARCE_QSUB_ARGS" != *"# ADJUST"* ]] || die "set SCARCE_QSUB_ARGS"
fi
[[ "$COMMON_ENABLED" =~ ^[01]$ ]] || die "COMMON_ENABLED must be 0 or 1"
[[ "$SCARCE_ENABLED" =~ ^[01]$ ]] || die "SCARCE_ENABLED must be 0 or 1"
[[ "$SCARCE_DEPENDS_ON_COMMON" =~ ^[01]$ ]] || die "SCARCE_DEPENDS_ON_COMMON must be 0 or 1"
[[ "$PRIORITY" =~ ^-?[0-9]+$ ]] || die "PRIORITY must be an integer"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID contains unsafe characters"
for value in "$PROJECT_ROOT" "$RUNNER" "$STATE_ROOT"; do
  [[ "$value" != *,* ]] || die "SGE -v paths must not contain commas: $value"
done
if [[ "$SCARCE_DEPENDS_ON_COMMON" == 1 && "$COMMON_ENABLED" != 1 ]]; then
  die "SCARCE_DEPENDS_ON_COMMON=1 requires COMMON_ENABLED=1"
fi

print_command() {
  {
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  } >&2
}

submit_group() {
  local group="$1"
  local job_name="$2"
  local resource_string="$3"
  local hold_id="${4:-}"
  local -a resources command

  read -r -a resources <<< "$resource_string"
  command=(
    qsub -terse -p "$PRIORITY" -N "$job_name"
    -o "$LOG_DIR/${job_name}.${RUN_ID}.out"
    -e "$LOG_DIR/${job_name}.${RUN_ID}.err"
    -v "PIPELINE_GROUP=$group,PROJECT_ROOT=$PROJECT_ROOT,STATE_ROOT=$STATE_ROOT,RUN_ID=$RUN_ID"
  )
  command+=("${resources[@]}")
  if [[ -n "$hold_id" ]]; then
    command+=(-hold_jid "$hold_id")
  fi
  command+=("$RUNNER")

  if [[ "$DRY_RUN" == 1 ]]; then
    print_command "${command[@]}"
    printf '<%s-job-id>\n' "$group"
  else
    "${command[@]}"
  fi
}

if [[ "$DRY_RUN" != 1 ]]; then
  command -v qsub >/dev/null 2>&1 || die "qsub is not available"
  [[ -d "$PROJECT_ROOT" ]] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
  [[ -f "$RUNNER" ]] || die "RUNNER does not exist: $RUNNER"
  mkdir -p "$LOG_DIR" "$STATE_ROOT/$RUN_ID"
fi

common_id=""
if [[ "$COMMON_ENABLED" == 1 ]]; then
  common_id=$(submit_group common "gpu_common" "$COMMON_QSUB_ARGS")
  echo "Common GPU job: $common_id"
fi

if [[ "$SCARCE_ENABLED" == 1 ]]; then
  hold_id=""
  if [[ "$SCARCE_DEPENDS_ON_COMMON" == 1 ]]; then
    hold_id="$common_id"
  fi
  scarce_id=$(submit_group scarce "gpu_scarce" "$SCARCE_QSUB_ARGS" "$hold_id")
  echo "Scarce GPU job: $scarce_id"
fi

echo "Run ID: $RUN_ID"
