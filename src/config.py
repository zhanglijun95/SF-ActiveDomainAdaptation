"""YAML config loader with attribute access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class AttrDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_attr(x: Any) -> Any:
    if isinstance(x, dict):
        return AttrDict({k: _to_attr(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_attr(v) for v in x]
    return x


def to_plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_plain(v) for v in x]
    return x


def load_config(path: str | Path) -> AttrDict:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return _to_attr(payload)
