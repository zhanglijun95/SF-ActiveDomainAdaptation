"""Small helpers for the isolated FNP baseline package."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import torch
import yaml

from src.config import to_plain


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_resolved_config(path: str | Path, cfg: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain(cfg), handle, sort_keys=False)


def resolve_aux_device(cfg: Any, primary_device: torch.device) -> torch.device:
    if primary_device.type != "cuda" or not torch.cuda.is_available():
        return primary_device

    train_cfg = getattr(getattr(cfg, "method", object()), "train", object())
    teacher_device_cfg = getattr(train_cfg, "teacher_device", None)
    if teacher_device_cfg is None:
        return primary_device

    raw = str(teacher_device_cfg).strip().lower()
    if raw in {"", "same", "student"}:
        return primary_device
    if raw == "auto":
        visible_count = torch.cuda.device_count()
        if visible_count <= 1:
            return primary_device
        primary_index = primary_device.index if primary_device.index is not None else torch.cuda.current_device()
        for idx in range(visible_count):
            if idx != primary_index:
                return torch.device(f"cuda:{idx}")
        return primary_device
    return torch.device(str(teacher_device_cfg))


def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
