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

CONFIG_DIR="$ROOT_DIR/configs/generated/source_train"
LOG_DIR="$ROOT_DIR/runs/logs/eval_source"
mkdir -p "$LOG_DIR"

SMOKE_CFG="$CONFIG_DIR/smoke_office_home_art.yaml"
JOBS=(
  "$CONFIG_DIR/office_home_art.yaml"
  "$CONFIG_DIR/office_home_clipart.yaml"
  "$CONFIG_DIR/office_home_product.yaml"
  "$CONFIG_DIR/office_home_real_world.yaml"
  "$CONFIG_DIR/office_31_amazon.yaml"
  "$CONFIG_DIR/office_31_dslr.yaml"
  "$CONFIG_DIR/office_31_webcam.yaml"
  "$CONFIG_DIR/visda_c_train.yaml"
)

SMOKE_CMD="CUDA_VISIBLE_DEVICES=0 $PY_RUNNER -m src.engine.eval_source --config $SMOKE_CFG"
GPU_MAP=(0 1 2 3 4 5 6 7)

COMMANDS_FILE="$ROOT_DIR/scripts/eval_source_jobs.commands.txt"
{
  echo "# Smoke eval command"
  echo "$SMOKE_CMD"
  echo
  echo "# Full eval jobs (run in parallel, one per GPU)"
  for i in "${!JOBS[@]}"; do
    echo "CUDA_VISIBLE_DEVICES=${GPU_MAP[$i]} $PY_RUNNER -m src.engine.eval_source --config ${JOBS[$i]}"
  done
} > "$COMMANDS_FILE"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [--list | --smoke | --run-all]

Modes:
  --list    Print smoke command and all eval commands (default)
  --smoke   Run the smoke eval command only
  --run-all Run all 8 eval jobs in parallel (GPU 0-7), logs in $LOG_DIR

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
      cmd="CUDA_VISIBLE_DEVICES=${GPU_MAP[$i]} $PY_RUNNER -m src.engine.eval_source --config $cfg"
      log="$LOG_DIR/job_${i}.log"
      echo "[job $i] $cmd"
      nohup bash -lc "$cmd" > "$log" 2>&1 &
    done
    echo "Launched ${#JOBS[@]} eval jobs. Logs: $LOG_DIR"
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
