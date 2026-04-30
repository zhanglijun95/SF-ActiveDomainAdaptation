"""Config helpers for the isolated PETS DAOD baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baselines.sfod_common.config import resolve_daod_source_ckpt_path, resolve_sfod_run_dir


def resolve_pets_daod_run_dir(cfg: Any) -> Path:
    return resolve_sfod_run_dir(cfg, baseline_key="pets_daod")


def resolve_pets_daod_source_ckpt_path(cfg: Any, which: str = "final") -> Path:
    return resolve_daod_source_ckpt_path(cfg, which=which)

