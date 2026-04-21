"""Engine utility functions."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import torch
import yaml

from src.config import to_plain


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())


def resolve_source_run_dir(cfg: Any) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    dataset = _slug(str(cfg.data.dataset_name))
    source = _slug(str(cfg.data.source_domain))
    return root / "source" / dataset / source


def resolve_daod_source_run_dir(cfg: Any) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    source = _slug(str(cfg.data.source_domain))
    target = _slug(str(cfg.data.target_domain))
    model_name = _slug(str(cfg.detector.model_name))
    return root / "daod_source" / f"{source}__to__{target}" / model_name


def resolve_daod_oracle_run_dir(cfg: Any) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    source = _slug(str(cfg.data.source_domain))
    target = _slug(str(cfg.data.target_domain))
    model_name = _slug(str(cfg.detector.model_name))
    return root / "daod_oracle" / f"{source}__to__{target}" / model_name


def resolve_daod_method_run_dir(cfg: Any) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    source = _slug(str(cfg.data.source_domain))
    target = _slug(str(cfg.data.target_domain))
    model_name = _slug(str(cfg.detector.model_name))
    method_cfg = getattr(cfg, "method", object())
    exp_name = str(getattr(method_cfg, "exp_name", "")).strip()
    if exp_name:
        exp_tag = _slug(exp_name)
    else:
        num_rounds = int(getattr(method_cfg, "num_rounds", 1))
        budget_total = str(getattr(method_cfg, "budget_total", "na")).replace(".", "p")
        exp_tag = f"rounds{num_rounds}_budget{budget_total}"
    return root / "daod_method" / exp_tag / f"{source}__to__{target}" / model_name


def resolve_optional_daod_checkpoint_path(path_value: Any, *, which: str = "best") -> Path | None:
    if path_value is None or not str(path_value).strip():
        return None

    checkpoint_path = Path(str(path_value))
    key = str(which).strip().lower()
    if checkpoint_path.is_dir():
        if key in {"best", "model_best", "best_ckpt"}:
            checkpoint_path = checkpoint_path / "model_best.pth"
        elif key in {"last", "latest", "final", "model_final"}:
            checkpoint_path = checkpoint_path / "model_final.pth"
        else:
            raise ValueError(f"Unsupported DAOD checkpoint selector: {which}. Use one of: best, final")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Explicit DAOD checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_daod_source_ckpt_path(cfg: Any, which: str = "best") -> Path:
    override_path = resolve_optional_daod_checkpoint_path(
        getattr(getattr(cfg, "detector", object()), "source_ckpt_path", None),
        which=which,
    )
    if override_path is not None:
        return override_path

    output_dir = resolve_daod_source_run_dir(cfg)
    key = str(which).strip().lower()
    if key in {"best", "model_best", "best_ckpt"}:
        return output_dir / "model_best.pth"
    if key in {"last", "latest", "final", "model_final"}:
        return output_dir / "model_final.pth"
    raise ValueError(f"Unsupported DAOD source checkpoint selector: {which}. Use one of: best, final")


def resolve_source_ckpt_path(cfg: Any, which: str = "best") -> Path:
    ckpt_dir = resolve_source_run_dir(cfg) / "ckpt"
    key = str(which).strip().lower()
    if key in {"best", "ckpt_best", "best_ckpt"}:
        return ckpt_dir / "ckpt_best.pt"
    if key in {"last", "latest", "ckpt_last"}:
        return ckpt_dir / "ckpt_last.pt"
    raise ValueError(f"Unsupported source checkpoint selector: {which}. Use one of: best, last")


def build_optimizer(cfg: Any, model) -> torch.optim.Optimizer:
    lr = float(cfg.train.lr)
    wd = float(getattr(cfg.train, "weight_decay", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    # return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)


def build_scheduler(cfg: Any, optimizer) -> Any:
    if not bool(getattr(cfg.train, "use_scheduler", False)):
        return None
    max_epochs = int(cfg.train.source_epochs)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_resolved_config(path: str | Path, cfg: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False)
