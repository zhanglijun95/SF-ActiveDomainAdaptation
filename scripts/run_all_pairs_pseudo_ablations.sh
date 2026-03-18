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
LOG_DIR="$ROOT_DIR/runs/logs/ablations/all_pairs_pseudo"
COMMANDS_FILE="$ROOT_DIR/scripts/run_all_pairs_pseudo_ablations.commands.txt"
PLAN_FILE="$ROOT_DIR/scripts/run_all_pairs_pseudo_ablations.plan.txt"
mkdir -p "$CFG_DIR" "$LOG_DIR"

slug() {
  local s="$1"
  s="${s// /_}"
  s="${s//\//_}"
  echo "$s" | tr '[:upper:]' '[:lower:]'
}

emit_cfg() {
  local cfg_path="$1"
  local dataset="$2"
  local root="$3"
  local num_classes="$4"
  local backbone="$5"
  local source="$6"
  local target="$7"
  local exp_name="$8"
  local score_use_change="$9"
  local w_margin="${10}"
  local w_change="${11}"
  local use_debias="${12}"
  local train_batch_size="${13:-64}"
  local eval_batch_size="${14:-128}"
  local random_pick="${15:-false}"
  local score_use_margin="${16:-true}"
  local use_pseudo="${17:-true}"

  cat > "$cfg_path" <<YAML
seed: 42

run:
  root_dir: runs

data:
  dataset_name: ${dataset}
  root: ${root}
  source_domain: "${source}"
  target_domain: "${target}"
  num_classes: ${num_classes}
  image_size: 224

model:
  source_ckpt: best
  backbone: ${backbone}
  pretrained: true
  bottleneck_dim: 256
  bottleneck_relu: true
  lora:
    enabled: false
    r: 8
    rank_schedule:
      layer1: 16
      layer2: 8
      layer3: 4
      layer4: 2
    alpha: 16
    dropout: 0.0
    target_modules:
      - layer1.*.conv1
      - layer1.*.conv2
      - layer1.*.conv3
      - layer2.*.conv1
      - layer2.*.conv2
      - layer2.*.conv3
      - layer3.*.conv1
      - layer3.*.conv2
      - layer3.*.conv3
      - layer4.*.conv1
      - layer4.*.conv2
      - layer4.*.conv3

train:
  epochs: 5
  batch_size: ${train_batch_size}
  num_workers: 4
  log_every_iters: 20
  save_ckpt: false
  lr: 0.0001
  weight_decay: 0.0005
  use_scheduler: false
  finetune_mode: backbone_only

eval:
  batch_size: ${eval_batch_size}
  num_workers: 4
  monitor_source_val: true

method:
  exp_name: ${exp_name}
  num_rounds: 10
  round_epochs: 20
  budget_total: 0.05
  random_pick: ${random_pick}
  score_use_margin: ${score_use_margin}
  score_use_change: ${score_use_change}
  w_margin: ${w_margin}
  w_change: ${w_change}
  use_debias: ${use_debias}
  debias_lambda: 1.0
  prior_momentum: 0.9
  use_pseudo: ${use_pseudo}
  pseudo_keep_ratio: 0.5
  pseudo_loss_weight: 1.0
  aml_lambda: 1.0
YAML
}

AB_IDS=(
  "ab00_random_pick"
  "ab05_margin_change_pseudo_ce"
  "ab06_margin_change_debias_pseudo_aml"
  "ab07_margin_pseudo_ce"
  "ab08_margin_debias_pseudo_aml"
)

# Ablation settings.
ab_random_of() {
  case "$1" in
    ab00_random_pick) echo "true" ;;
    *) echo "false" ;;
  esac
}
ab_margin_of() {
  case "$1" in
    ab00_random_pick) echo "false" ;;
    *) echo "true" ;;
  esac
}
ab_change_of() {
  case "$1" in
    ab00_random_pick) echo "false" ;;
    ab05_margin_change_pseudo_ce) echo "true" ;;
    ab06_margin_change_debias_pseudo_aml) echo "true" ;;
    ab07_margin_pseudo_ce) echo "false" ;;
    ab08_margin_debias_pseudo_aml) echo "false" ;;
    *) echo "false" ;;
  esac
}
ab_w_margin_of() {
  case "$1" in
    ab00_random_pick) echo "0.5" ;;
    ab05_margin_change_pseudo_ce|ab06_margin_change_debias_pseudo_aml) echo "0.5" ;;
    ab07_margin_pseudo_ce|ab08_margin_debias_pseudo_aml) echo "1.0" ;;
    *) echo "1.0" ;;
  esac
}
ab_w_change_of() {
  case "$1" in
    ab00_random_pick) echo "0.5" ;;
    ab05_margin_change_pseudo_ce|ab06_margin_change_debias_pseudo_aml) echo "0.5" ;;
    ab07_margin_pseudo_ce|ab08_margin_debias_pseudo_aml) echo "0.0" ;;
    *) echo "0.0" ;;
  esac
}
ab_debias_of() {
  case "$1" in
    ab06_margin_change_debias_pseudo_aml|ab08_margin_debias_pseudo_aml) echo "true" ;;
    *) echo "false" ;;
  esac
}
ab_pseudo_of() {
  case "$1" in
    ab00_random_pick) echo "false" ;;
    *) echo "true" ;;
  esac
}

build_all() {
  : > "$COMMANDS_FILE"
  : > "$PLAN_FILE"

  local -a cfgs=()

  # office_31: all ordered source->target pairs (3*2=6)
  local -a d31=("amazon" "dslr" "webcam")
  for s in "${d31[@]}"; do
    for t in "${d31[@]}"; do
      [[ "$s" == "$t" ]] && continue
      for ab in "${AB_IDS[@]}"; do
        local cfg="$CFG_DIR/office_31__$(slug "$s")__to__$(slug "$t")__${ab}.yaml"
        emit_cfg "$cfg" "office_31" "/home/ljzhang/data/sfada/office-31" "31" "resnet50" "$s" "$t" "$ab" "$(ab_change_of "$ab")" "$(ab_w_margin_of "$ab")" "$(ab_w_change_of "$ab")" "$(ab_debias_of "$ab")" "64" "128" "$(ab_random_of "$ab")" "$(ab_margin_of "$ab")" "$(ab_pseudo_of "$ab")"
        cfgs+=("$cfg")
      done
    done
  done

  # office_home: all ordered source->target pairs (4*3=12)
  local -a d_home=("Art" "Clipart" "Product" "Real World")
  for s in "${d_home[@]}"; do
    for t in "${d_home[@]}"; do
      [[ "$s" == "$t" ]] && continue
      for ab in "${AB_IDS[@]}"; do
        local cfg="$CFG_DIR/office_home__$(slug "$s")__to__$(slug "$t")__${ab}.yaml"
        emit_cfg "$cfg" "office_home" "/home/ljzhang/data/sfada/office-home" "65" "resnet50" "$s" "$t" "$ab" "$(ab_change_of "$ab")" "$(ab_w_margin_of "$ab")" "$(ab_w_change_of "$ab")" "$(ab_debias_of "$ab")" "64" "128" "$(ab_random_of "$ab")" "$(ab_margin_of "$ab")" "$(ab_pseudo_of "$ab")"
        cfgs+=("$cfg")
      done
    done
  done

  # visda_c: only train->validation (1 pair)
  for ab in "${AB_IDS[@]}"; do
    local cfg="$CFG_DIR/visda_c__train__to__validation__${ab}.yaml"
    emit_cfg "$cfg" "visda_c" "/home/ljzhang/data/sfada/visda-c" "12" "resnet101" "train" "validation" "$ab" "$(ab_change_of "$ab")" "$(ab_w_margin_of "$ab")" "$(ab_w_change_of "$ab")" "$(ab_debias_of "$ab")" "16" "32" "$(ab_random_of "$ab")" "$(ab_margin_of "$ab")" "$(ab_pseudo_of "$ab")"
    cfgs+=("$cfg")
  done

  local total="${#cfgs[@]}"
  local waves=$(( (total + 7) / 8 ))

  {
    echo "# Total jobs: $total"
    echo "# GPUs: 8"
    echo "# Planned waves: $waves"
    echo "# Job formula: (office_31:6 + office_home:12 + visda_c:1) * 5 ablations = 95"
    echo
  } > "$PLAN_FILE"

  {
    echo "# Full commands for all datasets/domain pairs and 5 ablations"
    echo "# ABLATIONS: ab00_random_pick, ab05_margin_change_pseudo_ce, ab06_margin_change_debias_pseudo_aml, ab07_margin_pseudo_ce, ab08_margin_debias_pseudo_aml"
    echo
  } > "$COMMANDS_FILE"

  local i=0
  while [[ $i -lt $total ]]; do
    local wave=$(( i / 8 ))
    echo "Wave ${wave}" >> "$PLAN_FILE"
    local j=0
    while [[ $j -lt 8 && $i -lt $total ]]; do
      local gpu=$j
      local cfg="${cfgs[$i]}"
      local base
      base="$(basename "$cfg" .yaml)"
      local log="$LOG_DIR/job_${i}__${base}.log"
      local cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PY_RUNNER} -m src.methods.run_rounds --config ${cfg}"
      echo "  [slot gpu${gpu}] job_${i}: ${base}" >> "$PLAN_FILE"
      echo "$cmd" >> "$COMMANDS_FILE"
      i=$((i + 1))
      j=$((j + 1))
    done
    echo >> "$PLAN_FILE"
  done

  echo "Generated ${total} configs under: $CFG_DIR"
  echo "Commands: $COMMANDS_FILE"
  echo "Plan: $PLAN_FILE"
}

run_waves() {
  local start_wave="${1:-0}"
  if ! [[ "$start_wave" =~ ^[0-9]+$ ]]; then
    echo "start_wave must be a non-negative integer, got: $start_wave" >&2
    exit 1
  fi
  # Rebuild to ensure commands/plan match current script logic.
  build_all >/dev/null

  mapfile -t cmds < <(grep -E '^CUDA_VISIBLE_DEVICES=' "$COMMANDS_FILE")
  local total="${#cmds[@]}"
  local wave_size=8
  local idx=$(( start_wave * wave_size ))
  local wave="$start_wave"

  if [[ "$idx" -ge "$total" ]]; then
    echo "start_wave=$start_wave is out of range (total jobs=$total, wave_size=$wave_size)." >&2
    exit 1
  fi

  while [[ $idx -lt $total ]]; do
    echo "[wave ${wave}] launching jobs $idx..$(( idx + wave_size - 1 < total ? idx + wave_size - 1 : total - 1 ))"
    local -a pids=()
    local slot=0
    while [[ $slot -lt $wave_size && $idx -lt $total ]]; do
      local cmd="${cmds[$idx]}"
      local log
      log="$LOG_DIR/job_${idx}.log"
      echo "  [job $idx][gpu $slot] $cmd"
      nohup bash -lc "$cmd" > "$log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      slot=$((slot + 1))
    done
    for p in "${pids[@]}"; do
      wait "$p"
    done
    echo "[wave ${wave}] completed"
    wave=$((wave + 1))
  done

  echo "All jobs finished. Logs: $LOG_DIR"
}

run_random_only() {
  build_all >/dev/null

  mapfile -t cmds < <(grep -E '^CUDA_VISIBLE_DEVICES=.*ab00_random_pick' "$COMMANDS_FILE")
  local total="${#cmds[@]}"
  local wave_size=8
  local idx=0
  local wave=0

  if [[ "$total" -eq 0 ]]; then
    echo "No random baseline jobs found in $COMMANDS_FILE" >&2
    exit 1
  fi

  while [[ $idx -lt $total ]]; do
    echo "[random wave ${wave}] launching jobs $idx..$(( idx + wave_size - 1 < total ? idx + wave_size - 1 : total - 1 ))"
    local -a pids=()
    local slot=0
    while [[ $slot -lt $wave_size && $idx -lt $total ]]; do
      local cmd="${cmds[$idx]}"
      local log="$LOG_DIR/random_only_job_${idx}.log"
      echo "  [random job $idx][gpu $slot] $cmd"
      nohup bash -lc "$cmd" > "$log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      slot=$((slot + 1))
    done
    for p in "${pids[@]}"; do
      wait "$p"
    done
    echo "[random wave ${wave}] completed"
    wave=$((wave + 1))
  done

  echo "Random-only jobs finished. Logs: $LOG_DIR"
}

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [--list | --plan | --run-waves | --run-waves-from <wave_idx> | --run-random-only]

Modes:
  --list       Generate configs + commands and print command file
  --plan       Generate configs + print execution plan (waves on 8 GPUs)
  --run-waves  Run all jobs wave-by-wave (max 8 concurrent, no GPU conflict)
  --run-waves-from <wave_idx>
               Resume wave execution starting from the given wave index
  --run-random-only
               Run only ab00_random_pick jobs in 8-GPU waves

Artifacts:
  Configs : $CFG_DIR
  Commands: $COMMANDS_FILE
  Plan    : $PLAN_FILE
  Logs    : $LOG_DIR
USAGE
}

MODE="${1:---list}"
case "$MODE" in
  --list)
    build_all
    echo
    sed -n '1,220p' "$COMMANDS_FILE"
    ;;
  --plan)
    build_all
    echo
    cat "$PLAN_FILE"
    ;;
  --run-waves)
    run_waves 0
    ;;
  --run-waves-from)
    run_waves "${2:-}"
    ;;
  --run-random-only)
    run_random_only
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
