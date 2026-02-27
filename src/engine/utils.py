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
