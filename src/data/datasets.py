"""Dataset registry and placeholder dataset implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


SplitName = str
DomainName = str


@dataclass
class DatasetRecord:
    image_path: str
    label: int
    sample_id: str


class ClassificationFolderDataset(Dataset):
    """Generic dataset reading from a list file.

    Expected list format per line: `relative_path label sample_id`.
    """

    def __init__(
        self,
        root: str,
        records: list[DatasetRecord],
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        img = Image.open(self.root / rec.image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": rec.label, "sample_id": rec.sample_id}


class PlaceholderDataset(Dataset):
    """Import-safe placeholder until dataset parsers are implemented."""

    def __init__(self, dataset_name: str, split: SplitName, domain: DomainName) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.domain = domain

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise IndexError("Empty placeholder dataset")


def _load_records_from_list(root: str, list_path: str) -> list[DatasetRecord]:
    path = Path(root) / list_path
    if not path.exists():
        return []
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label, sample_id = line.split(maxsplit=2)
            records.append(DatasetRecord(rel_path, int(label), sample_id))
    return records


def build_dataset(
    cfg: Any,
    dataset_name: str,
    split: SplitName,
    domain: DomainName,
    transform: Callable[[Image.Image], Any] | None,
    return_id: bool = True,
) -> Dataset:
    """Build dataset by name/split/domain.

    The concrete dataset parsing should be plugged in here. Current scaffold supports a
    generic list-file fallback if cfg.data.list_files contains entries for the split.
    """
    _ = return_id
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        return PlaceholderDataset(dataset_name, split, domain)

    root = getattr(data_cfg, "root", "")
    list_files = getattr(data_cfg, "list_files", {})
    split_list = list_files.get(split) if isinstance(list_files, dict) else None
    if split_list:
        records = _load_records_from_list(root, split_list)
        return ClassificationFolderDataset(root=root, records=records, transform=transform)

    return PlaceholderDataset(dataset_name, split, domain)
