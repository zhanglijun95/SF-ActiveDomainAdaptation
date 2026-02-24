#!/usr/bin/env bash
set -euo pipefail

source /home/ljzhang/conda/etc/profile.d/conda.sh
conda activate sfada

python /home/ljzhang/code/SFADA/scripts/generate_source_splits.py \
  --data-root /home/ljzhang/data/sfada \
  --seed 42 \
  --val-ratio 0.1
