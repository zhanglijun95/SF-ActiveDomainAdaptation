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

CFG_DIR="$ROOT_DIR/configs/generated/pseudo_ablations_all_pairs"
LOG_DIR="$ROOT_DIR/runs/logs/ablations/visda_c_only"
COMMANDS_FILE="$ROOT_DIR/scripts/run_visdac_only_ablations.commands.txt"
mkdir -p "$LOG_DIR"

JOBS=(
  "$CFG_DIR/visda_c__train__to__validation__ab00_random_pick.yaml"
  "$CFG_DIR/visda_c__train__to__validation__ab05_margin_change_pseudo_ce.yaml"
  "$CFG_DIR/visda_c__train__to__validation__ab06_margin_change_debias_pseudo_aml.yaml"
  "$CFG_DIR/visda_c__train__to__validation__ab07_margin_pseudo_ce.yaml"
  "$CFG_DIR/visda_c__train__to__validation__ab08_margin_debias_pseudo_aml.yaml"
)

GPU_MAP=(0 1 2 3 4)

{
  echo "# One-wave VisDA-C ablations"
  for i in "${!JOBS[@]}"; do
    echo "CUDA_VISIBLE_DEVICES=${GPU_MAP[$i]} $PY_RUNNER -m src.methods.run_rounds --config ${JOBS[$i]}"
  done
} > "$COMMANDS_FILE"

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [--list | --run-wave]

Modes:
  --list      Print the 5 VisDA-C ablation commands (default)
  --run-wave  Launch all 5 jobs in one wave on GPUs 0..4

Artifacts:
  Commands: $COMMANDS_FILE
  Logs    : $LOG_DIR
USAGE
}

MODE="${1:---list}"
case "$MODE" in
  --list)
    print_usage
    echo
    cat "$COMMANDS_FILE"
    ;;
  --run-wave)
    for i in "${!JOBS[@]}"; do
      cfg="${JOBS[$i]}"
      gpu="${GPU_MAP[$i]}"
      cmd="CUDA_VISIBLE_DEVICES=$gpu $PY_RUNNER -m src.methods.run_rounds --config $cfg"
      log="$LOG_DIR/job_${i}.log"
      echo "[job $i][gpu $gpu] $cmd"
      nohup bash -lc "$cmd" > "$log" 2>&1 &
    done
    echo "Launched 5 VisDA-C ablation jobs. Logs: $LOG_DIR"
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
