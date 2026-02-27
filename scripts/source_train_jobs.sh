#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Activate expected runtime env.
if [[ -x "/home/ljzhang/conda/envs/sfada/bin/python" ]]; then
  PY_RUNNER="/home/ljzhang/conda/envs/sfada/bin/python"
elif [[ -x "/local/home/ljzhang/conda/envs/sfada/bin/python" ]]; then
  PY_RUNNER="/local/home/ljzhang/conda/envs/sfada/bin/python"
elif command -v conda >/dev/null 2>&1; then
  # Fallback only when explicit env python is unavailable.
  PY_RUNNER="conda run -n sfada --no-capture-output python"
else
  PY_RUNNER="python"
fi

CONFIG_DIR="$ROOT_DIR/configs/generated/source_train"
LOG_DIR="$ROOT_DIR/runs/logs/source_train"
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

write_source_cfg() {
  local out="$1"
  local dataset="$2"
  local data_root="$3"
  local source_domain="$4"
  local num_classes="$5"
  local source_epochs="$6"
  local backbone="${7:-resnet50}"
  local log_every_iters="${8:-10}"
  cat > "$out" <<EOF
seed: 42

run:
  root_dir: runs

data:
  dataset_name: ${dataset}
  root: ${data_root}
  source_domain: "${source_domain}"
  num_classes: ${num_classes}
  image_size: 224

model:
  backbone: ${backbone}
  pretrained: true
  bottleneck_dim: 256
  bottleneck_relu: true

train:
  source_epochs: ${source_epochs}
  batch_size: 64
  num_workers: 4
  log_every_iters: ${log_every_iters}
  lr: 0.0001
  weight_decay: 0.0005
  use_scheduler: true

eval:
  batch_size: 128
  num_workers: 4
  source_ckpt: best
  monitor_source_val: true
EOF
}

infer_log_every_iters() {
  local dataset="$1"
  local data_root="$2"
  local source_domain="$3"
  local batch_size="$4"
  local split_file=""

  case "$dataset" in
    office_home|office_31)
      split_file="$data_root/splits/$source_domain/seed_42/source_train.txt"
      ;;
    visda_c)
      split_file="$data_root/splits/train/seed_42/source_train.txt"
      ;;
    *)
      echo "10"
      return
      ;;
  esac

  if [[ ! -f "$split_file" ]]; then
    echo "10"
    return
  fi

  local n_samples
  n_samples="$(wc -l < "$split_file" | tr -d ' ')"
  local steps=$(( (n_samples + batch_size - 1) / batch_size ))
  local log_every=$(( steps / 3 ))
  if [[ "$log_every" -lt 1 ]]; then
    log_every=1
  fi
  echo "$log_every"
}

# Smoke config (fast sanity check)
SMOKE_CFG="$CONFIG_DIR/smoke_office_home_art.yaml"
BATCH_SIZE=64
SMOKE_LOG_EVERY="$(infer_log_every_iters "office_home" "/home/ljzhang/data/sfada/office-home" "Art" "$BATCH_SIZE")"
write_source_cfg "$SMOKE_CFG" "office_home" "/home/ljzhang/data/sfada/office-home" "Art" 65 1 "resnet50" "$SMOKE_LOG_EVERY"

# Office-Home (4 jobs)
OH_ART_LOG="$(infer_log_every_iters "office_home" "/home/ljzhang/data/sfada/office-home" "Art" "$BATCH_SIZE")"
OH_CLIP_LOG="$(infer_log_every_iters "office_home" "/home/ljzhang/data/sfada/office-home" "Clipart" "$BATCH_SIZE")"
OH_PROD_LOG="$(infer_log_every_iters "office_home" "/home/ljzhang/data/sfada/office-home" "Product" "$BATCH_SIZE")"
OH_REAL_LOG="$(infer_log_every_iters "office_home" "/home/ljzhang/data/sfada/office-home" "Real World" "$BATCH_SIZE")"
write_source_cfg "$CONFIG_DIR/office_home_art.yaml" "office_home" "/home/ljzhang/data/sfada/office-home" "Art" 65 20 "resnet50" "$OH_ART_LOG"
write_source_cfg "$CONFIG_DIR/office_home_clipart.yaml" "office_home" "/home/ljzhang/data/sfada/office-home" "Clipart" 65 20 "resnet50" "$OH_CLIP_LOG"
write_source_cfg "$CONFIG_DIR/office_home_product.yaml" "office_home" "/home/ljzhang/data/sfada/office-home" "Product" 65 20 "resnet50" "$OH_PROD_LOG"
write_source_cfg "$CONFIG_DIR/office_home_real_world.yaml" "office_home" "/home/ljzhang/data/sfada/office-home" "Real World" 65 20 "resnet50" "$OH_REAL_LOG"

# Office-31 (3 jobs)
O31_AMZ_LOG="$(infer_log_every_iters "office_31" "/home/ljzhang/data/sfada/office-31" "amazon" "$BATCH_SIZE")"
O31_DSLR_LOG="$(infer_log_every_iters "office_31" "/home/ljzhang/data/sfada/office-31" "dslr" "$BATCH_SIZE")"
O31_WEBCAM_LOG="$(infer_log_every_iters "office_31" "/home/ljzhang/data/sfada/office-31" "webcam" "$BATCH_SIZE")"
write_source_cfg "$CONFIG_DIR/office_31_amazon.yaml" "office_31" "/home/ljzhang/data/sfada/office-31" "amazon" 31 20 "resnet50" "$O31_AMZ_LOG"
write_source_cfg "$CONFIG_DIR/office_31_dslr.yaml" "office_31" "/home/ljzhang/data/sfada/office-31" "dslr" 31 20 "resnet50" "$O31_DSLR_LOG"
write_source_cfg "$CONFIG_DIR/office_31_webcam.yaml" "office_31" "/home/ljzhang/data/sfada/office-31" "webcam" 31 20 "resnet50" "$O31_WEBCAM_LOG"

# VisDA-C source=train (1 job)
VISDA_LOG="$(infer_log_every_iters "visda_c" "/home/ljzhang/data/sfada/visda-c" "train" "$BATCH_SIZE")"
write_source_cfg "$CONFIG_DIR/visda_c_train.yaml" "visda_c" "/home/ljzhang/data/sfada/visda-c" "train" 12 20 "resnet101" "$VISDA_LOG"

SMOKE_CMD="CUDA_VISIBLE_DEVICES=0 $PY_RUNNER -m src.engine.train_source --config $SMOKE_CFG"
JOBS=(
  "CUDA_VISIBLE_DEVICES=0 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/office_home_art.yaml"
  "CUDA_VISIBLE_DEVICES=1 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/office_home_clipart.yaml"
  "CUDA_VISIBLE_DEVICES=2 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/office_home_product.yaml"
  "CUDA_VISIBLE_DEVICES=3 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/office_home_real_world.yaml"
  "CUDA_VISIBLE_DEVICES=4 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/office_31_amazon.yaml"
  "CUDA_VISIBLE_DEVICES=5 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/office_31_dslr.yaml"
  "CUDA_VISIBLE_DEVICES=6 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/office_31_webcam.yaml"
  "CUDA_VISIBLE_DEVICES=7 $PY_RUNNER -m src.engine.train_source --config $CONFIG_DIR/visda_c_train.yaml"
)

COMMANDS_FILE="$ROOT_DIR/scripts/source_train_jobs.commands.txt"
{
  echo "# Smoke test command (1 epoch)"
  echo "$SMOKE_CMD"
  echo
  echo "# Full source training jobs (run in parallel, one per GPU)"
  for cmd in "${JOBS[@]}"; do
    echo "$cmd"
  done
} > "$COMMANDS_FILE"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [--list | --smoke | --run-all]

Modes:
  --list    Print smoke command and all job commands (default)
  --smoke   Run the smoke test command only
  --run-all Run all 8 source training jobs in parallel (GPU 0-7), logs in $LOG_DIR

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
      log="$LOG_DIR/job_${i}.log"
      echo "[job $i] ${JOBS[$i]}"
      nohup bash -lc "${JOBS[$i]}" > "$log" 2>&1 &
    done
    echo "Launched ${#JOBS[@]} jobs. Logs: $LOG_DIR"
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
