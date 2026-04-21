"""Config and path helpers for the isolated DDT baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import slug


def resolve_ddt_daod_run_dir(cfg: Any) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    source = slug(str(cfg.data.source_domain))
    target = slug(str(cfg.data.target_domain))
    model_name = slug(str(cfg.detector.model_name))
    method_cfg = getattr(cfg, "method", object())
    exp_name = str(getattr(method_cfg, "exp_name", "")).strip()
    if exp_name:
        exp_tag = slug(exp_name)
    else:
        epochs = int(getattr(method_cfg, "epochs", 1))
        exp_tag = f"epochs{epochs}"
    return root / "baselines" / "ddt_daod" / exp_tag / f"{source}__to__{target}" / model_name


def resolve_ddt_daod_source_ckpt_path(cfg: Any, which: str = "best") -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    source = slug(str(cfg.data.source_domain))
    target = slug(str(cfg.data.target_domain))
    model_name = slug(str(cfg.detector.model_name))
    output_dir = root / "daod_source" / f"{source}__to__{target}" / model_name
    key = str(which).strip().lower()
    if key in {"best", "model_best", "best_ckpt"}:
        return output_dir / "model_best.pth"
    if key in {"last", "latest", "final", "model_final"}:
        return output_dir / "model_final.pth"
    raise ValueError(f"Unsupported DAOD source checkpoint selector: {which}. Use one of: best, final")
