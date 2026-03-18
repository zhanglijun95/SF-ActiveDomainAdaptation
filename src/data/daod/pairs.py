"""DAOD dataset-pair routing."""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image

from .cityscapes_to_foggy_cityscapes import (
    CITYSCAPES_THING_CLASSES,
    build_dataset as build_cityscapes_to_foggy_cityscapes_dataset,
)


_PAIR_BUILDERS = {
    ("cityscapes", "foggy_cityscapes"): build_cityscapes_to_foggy_cityscapes_dataset,
}

_PAIR_THING_CLASSES = {
    ("cityscapes", "foggy_cityscapes"): CITYSCAPES_THING_CLASSES,
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _pair_key(cfg: Any) -> tuple[str, str]:
    return (
        _normalize_name(cfg.data.source_domain),
        _normalize_name(cfg.data.target_domain),
    )


def build_daod_dataset(
    cfg: Any,
    split: str,
    transform: Callable[[Image.Image], Any] | None,
):
    key = _pair_key(cfg)
    if key not in _PAIR_BUILDERS:
        raise ValueError(
            f"Unsupported DAOD dataset pair: source_domain={cfg.data.source_domain}, "
            f"target_domain={cfg.data.target_domain}"
        )
    return _PAIR_BUILDERS[key](cfg, split, transform)


def get_daod_thing_classes(cfg: Any) -> tuple[str, ...]:
    key = _pair_key(cfg)
    if key not in _PAIR_THING_CLASSES:
        raise ValueError(
            f"Unsupported DAOD dataset pair: source_domain={cfg.data.source_domain}, "
            f"target_domain={cfg.data.target_domain}"
        )
    return _PAIR_THING_CLASSES[key]
