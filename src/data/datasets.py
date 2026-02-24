"""Dataset builder and list-file based dataset implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


SplitName = str


@dataclass
class DatasetRecord:
    image_path: str
    label: int
    sample_id: str


class ClassificationFolderDataset(Dataset):
    """Dataset backed by a list file.

    List format per line:
    - `rel_path label sample_id` or
    - `rel_path label` (sample_id auto-generated)
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


def _canon_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _context(cfg: Any, split: SplitName) -> dict[str, str]:
    data_cfg = cfg.data
    if split in {"source_train", "source_val"}:
        domain = str(data_cfg.source_domain)
    elif split in {"target_adapt", "target_test"}:
        domain = str(data_cfg.target_domain)
    else:
        raise ValueError(f"Unsupported split: {split}")

    return {
        "dataset_name": str(data_cfg.dataset_name),
        "source_domain": str(data_cfg.source_domain),
        "target_domain": str(getattr(data_cfg, "target_domain", "")),
        "domain": domain,
        "split": str(split),
        "seed": str(cfg.seed),
    }


def _default_split_list_path(ctx: dict[str, str]) -> str:
    ds = _canon_dataset_name(ctx["dataset_name"])
    split = ctx["split"]
    source = ctx["source_domain"]
    target = ctx["target_domain"]
    seed = ctx["seed"]

    if ds in {"office_home", "office_31"}:
        if split in {"source_train", "source_val"}:
            return f"splits/{source}/seed_{seed}/{split}.txt"
        return f"{target}/{target}_list.txt"

    if ds == "visda_c":
        if split in {"source_train", "source_val"}:
            return f"splits/train/seed_{seed}/{split}.txt"
        # Project rule: both target_adapt and target_test use validation split.
        return "validation/real_list.txt"

    raise ValueError(f"No default split list mapping for dataset={ctx['dataset_name']}, split={split}")


def _default_image_root_rel(ctx: dict[str, str]) -> str:
    ds = _canon_dataset_name(ctx["dataset_name"])
    split = ctx["split"]

    if ds in {"office_home", "office_31"}:
        return ctx["source_domain"] if split in {"source_train", "source_val"} else ctx["target_domain"]

    if ds == "visda_c":
        return "train" if split in {"source_train", "source_val"} else "validation"

    return ""


def _load_records_from_list(list_path: Path, sample_id_prefix: str) -> list[DatasetRecord]:
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")

    records: list[DatasetRecord] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid list line in {list_path}: {line}")

            rel_path = parts[0]
            label = int(parts[1])
            sample_id = " ".join(parts[2:]) if len(parts) >= 3 else f"{sample_id_prefix}:{rel_path}"
            records.append(DatasetRecord(rel_path, label, sample_id))
    return records


def build_dataset(
    cfg: Any,
    split: SplitName,
    transform: Callable[[Image.Image], Any] | None,
) -> Dataset:
    """Build dataset for a split from cfg."""
    ctx = _context(cfg, split)
    root = Path(cfg.data.root)

    split_list_rel = _default_split_list_path(ctx)
    split_list_path = root / split_list_rel

    image_root_rel = _default_image_root_rel(ctx)
    image_root = root / image_root_rel if image_root_rel else root

    sample_id_prefix = f"{ctx['dataset_name']}:{ctx['domain']}:{split}"
    records = _load_records_from_list(split_list_path, sample_id_prefix=sample_id_prefix)
    return ClassificationFolderDataset(root=str(image_root), records=records, transform=transform)
