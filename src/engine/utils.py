"""Engine utility functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from src.config import to_plain
from src.models.lora import apply_finetune_mode


def apply_train_mode(cfg: Any, model, mode: str) -> None:
    _ = cfg
    apply_finetune_mode(model, mode)


def build_optimizer(cfg: Any, model) -> torch.optim.Optimizer:
    lr = float(cfg.train.lr)
    wd = float(getattr(cfg.train, "weight_decay", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)


def build_scheduler(cfg: Any, optimizer) -> Any:
    if not bool(getattr(cfg.train, "use_scheduler", False)):
        return None
    max_epochs = int(cfg.train.epochs)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_resolved_config(path: str | Path, cfg: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False)
