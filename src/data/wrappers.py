"""Dataset wrappers for SF-ADA label routing and split views."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class LabelRouterDataset(Dataset):
    """Routes target-adapt labels by queried/pseudo state."""

    def __init__(
        self,
        base_ds: Dataset,
        queried_ids: set[Any],
        pseudo_store: dict[Any, int] | None = None,
        unlabeled_label: int = -1,
    ) -> None:
        self.base_ds = base_ds
        self.queried_ids = queried_ids
        self.pseudo_store = pseudo_store or {}
        self.unlabeled_label = unlabeled_label

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base_ds[index])
        sid = item["sample_id"]
        if sid in self.queried_ids:
            return item
        if sid in self.pseudo_store:
            item["label"] = int(self.pseudo_store[sid])
            return item
        item["label"] = self.unlabeled_label
        return item


class IdFilteredDataset(Dataset):
    """Filters samples by id using string mode."""

    VALID_MODES = {"labeled", "pseudo", "pool"}

    def __init__(
        self,
        base_ds: Dataset,
        mode: str,
        queried_ids: set[Any] | None = None,
        pseudo_ids: set[Any] | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}")
        self.base_ds = base_ds
        self.mode = mode
        self.queried_ids = queried_ids or set()
        self.pseudo_ids = pseudo_ids or set()
        self.indices = self._build_indices()

    def _keep(self, sample_id: Any) -> bool:
        if self.mode == "labeled":
            return sample_id in self.queried_ids
        if self.mode == "pseudo":
            return sample_id in self.pseudo_ids
        return sample_id not in self.queried_ids and sample_id not in self.pseudo_ids

    def _build_indices(self) -> list[int]:
        indices: list[int] = []
        for i in range(len(self.base_ds)):
            sid = self.base_ds[i]["sample_id"]
            if self._keep(sid):
                indices.append(i)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base_ds[self.indices[index]]


class TwoViewDataset(Dataset):
    """Produces weak/strong views with stable label/id fields."""

    def __init__(self, base_ds: Dataset, weak_tf, strong_tf) -> None:
        self.base_ds = base_ds
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_ds[index]
        image = item["image"]
        return {
            "x_w": self.weak_tf(image),
            "x_s": self.strong_tf(image),
            "label": item["label"],
            "sample_id": item["sample_id"],
        }
