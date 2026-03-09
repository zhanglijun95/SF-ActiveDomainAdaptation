#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "/home/ljzhang/conda/envs/sfada/bin/python" ]]; then
  PY_RUNNER="/home/ljzhang/conda/envs/sfada/bin/python"
elif [[ -x "/local/home/ljzhang/conda/envs/sfada/bin/python" ]]; then
  PY_RUNNER="/local/home/ljzhang/conda/envs/sfada/bin/python"
elif command -v conda >/dev/null 2>&1; then
  PY_RUNNER="conda run -n sfada --no-capture-output python"
else
  PY_RUNNER="python"
fi

CFG_DIR="$ROOT_DIR/configs/ablations/office_31"
LOG_DIR="$ROOT_DIR/runs/logs/ablations/office_31"
mkdir -p "$LOG_DIR"

JOBS=(
  "$CFG_DIR/00_random_pick.yaml"
  "$CFG_DIR/01_margin_only.yaml"
  "$CFG_DIR/02_change_only.yaml"
  "$CFG_DIR/03_margin_change.yaml"
  "$CFG_DIR/04_margin_change_debias.yaml"
  "$CFG_DIR/05_margin_change_pseudo_ce.yaml"
  "$CFG_DIR/06_margin_change_debias_pseudo_aml.yaml"
)

# 7 jobs, map to GPU 0..6 by default.
GPU_MAP=(0 1 2 3 4 5 6)

SMOKE_CMD="CUDA_VISIBLE_DEVICES=0 $PY_RUNNER -m src.methods.run_rounds --config ${JOBS[0]}"

COMMANDS_FILE="$ROOT_DIR/scripts/run_office31_ablations.commands.txt"
{
  echo "# Smoke command (first ablation)"
  echo "$SMOKE_CMD"
  echo
  echo "# Full Office-31 ablation jobs"
  for i in "${!JOBS[@]}"; do
    echo "CUDA_VISIBLE_DEVICES=${GPU_MAP[$i]} $PY_RUNNER -m src.methods.run_rounds --config ${JOBS[$i]}"
  done
} > "$COMMANDS_FILE"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [--list | --smoke | --run-all]

Modes:
  --list    Print smoke command and all ablation commands (default)
  --smoke   Run first ablation only (00_random_pick)
  --run-all Run all Office-31 ablations in parallel, logs in $LOG_DIR

Commands are also written to:
  $COMMANDS_FILE
EOF
}

MODE="${1:---list}"
case "$MODE" in
  --list)
    print_usage
    echo
    cat "$COMMANDS_FILE"
    ;;
  --smoke)
    echo "$SMOKE_CMD"
    eval "$SMOKE_CMD"
    ;;
  --run-all)
    for i in "${!JOBS[@]}"; do
      cfg="${JOBS[$i]}"
      gpu="${GPU_MAP[$i]}"
      cmd="CUDA_VISIBLE_DEVICES=$gpu $PY_RUNNER -m src.methods.run_rounds --config $cfg"
      log="$LOG_DIR/job_${i}.log"
      echo "[job $i][gpu $gpu] $cmd"
      nohup bash -lc "$cmd" > "$log" 2>&1 &
    done
    echo "Launched ${#JOBS[@]} ablation jobs. Logs: $LOG_DIR"
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
