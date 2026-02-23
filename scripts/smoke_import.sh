#!/usr/bin/env bash
set -euo pipefail
python3 -c "import src; from src.data import build_dataset; from src.models import build_model; print('ok')"
