"""DAOD-specific dataset and sample wrappers.

These helpers keep the round trainer simpler without forcing the classification
wrappers in `src/data/wrappers.py` to take on object-detection-specific view
logic. They are intentionally lightweight:

- `DAODListDataset` wraps a list of sample dicts for `DataLoader`
- `collate_daod_batch` keeps a batch as a plain list of sample dicts
- `cycle_daod_loader` repeats a loader so labeled/unlabeled streams can be
  consumed together even if their lengths differ
- `build_weak_view_sample` / `build_strong_view_sample` clone a sample and add
  an in-memory transformed image plus the associated view metadata
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from torch.utils.data import Dataset

from .transforms import make_seeded_strong_view, make_strong_view, make_weak_view


class DAODListDataset(Dataset):
    """Wrap a list of DAOD sample dicts for `DataLoader` use."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_daod_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a DAOD batch as a plain list of sample dicts."""

    return batch


def cycle_daod_loader(loader: Iterable[list[dict[str, Any]]]):
    """Repeat a DAOD loader until the caller stops consuming it."""

    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            return


def _clone_sample_with_view(
    sample: dict[str, Any],
    *,
    image,
    suffix: str,
    view_meta: dict[str, Any],
) -> dict[str, Any]:
    """Clone a DAOD sample and attach an in-memory transformed image."""

    cloned = dict(sample)
    cloned["image"] = image
    cloned["sample_id"] = f"{sample['sample_id']}::{suffix}"
    cloned["view_meta"] = dict(view_meta)
    return cloned


def build_weak_view_sample(
    sample: dict[str, Any],
    *,
    rng: random.Random | None = None,
    flip_prob: float = 0.5,
    suffix: str = "weak",
) -> dict[str, Any]:
    """Clone one sample with the DAOD weak view attached in-memory."""

    image = sample["image"] if "image" in sample else None
    if image is None:
        from PIL import Image

        image = Image.open(sample["file_name"]).convert("RGB")
    weak_image, weak_meta = make_weak_view(image.copy(), rng=rng, flip_prob=flip_prob)
    return _clone_sample_with_view(sample, image=weak_image, suffix=suffix, view_meta=weak_meta)


def build_strong_view_sample(
    sample: dict[str, Any],
    *,
    rng: random.Random | None = None,
    suffix: str = "strong",
) -> dict[str, Any]:
    """Clone one sample with the DAOD strong view attached in-memory."""

    image = sample["image"] if "image" in sample else None
    if image is None:
        from PIL import Image

        image = Image.open(sample["file_name"]).convert("RGB")
    if rng is None:
        strong_image, strong_meta = make_strong_view(image.copy())
    else:
        strong_image, strong_meta = make_seeded_strong_view(image.copy(), rng=rng)
    return _clone_sample_with_view(sample, image=strong_image, suffix=suffix, view_meta=strong_meta)
