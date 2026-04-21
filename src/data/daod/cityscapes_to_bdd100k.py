"""Cityscapes -> BDD100K DAOD dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from .cityscapes_to_foggy_cityscapes import (
    CITYSCAPES_CATEGORY_TO_ID,
    CITYSCAPES_THING_CLASSES,
    DAODCityscapesDataset,
    resolve_daod_split_root,
)


SplitName = str

BDD100K_THING_CLASSES = CITYSCAPES_THING_CLASSES
BDD100K_CATEGORY_TO_ID = CITYSCAPES_CATEGORY_TO_ID

BDD100K_TO_CITYSCAPES_CATEGORY = {
    "person": "person",
    "pedestrian": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
}

VIEW_SPECS = {
    "target_train": {"split": "train"},
    "target_val": {"split": "val"},
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class BDD100KRecord:
    image_path: str
    sample_id: str
    annotations: list[dict[str, Any]]
    height: int | None = None
    width: int | None = None


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower()


def _image_root(root: Path, split: SplitName) -> Path:
    return root / "images" / "100k" / str(VIEW_SPECS[split]["split"])


def _labels_path(root: Path, split: SplitName) -> Path:
    split_name = str(VIEW_SPECS[split]["split"])
    return root / "labels" / f"bdd100k_labels_images_{split_name}.json"


def _load_labels(labels_path: Path) -> list[dict[str, Any]]:
    with labels_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of annotations in {labels_path}")
    return payload


def _build_image_index(image_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    # The local copy may shard one official split across nested folders like
    # trainA/trainB, so we index recursively and rely on the JSON split file as
    # the actual source of truth.
    for image_path in sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        image_name = image_path.name
        if image_name in image_index:
            raise ValueError(f"Duplicate BDD100K image basename under {image_root}: {image_name}")
        image_index[image_name] = image_path
    return image_index


def _copy_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for ann in annotations:
        copied.append(
            {
                "bbox": list(ann["bbox"]),
                "bbox_mode": int(ann["bbox_mode"]),
                "category_id": int(ann["category_id"]),
                "iscrowd": int(ann.get("iscrowd", 0)),
                "area": float(ann["area"]),
            }
        )
    return copied


def _to_bbox(box2d: dict[str, Any]) -> list[float]:
    x0 = float(box2d["x1"])
    y0 = float(box2d["y1"])
    x1 = float(box2d["x2"])
    y1 = float(box2d["y2"])
    return [x0, y0, x1, y1]


def _bbox_area(bbox: list[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _parse_annotations(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for obj in labels or []:
        raw_category = _normalize_token(obj.get("category", ""))
        mapped_category = BDD100K_TO_CITYSCAPES_CATEGORY.get(raw_category)
        if mapped_category is None:
            continue
        box2d = obj.get("box2d")
        if not isinstance(box2d, dict):
            continue
        bbox = _to_bbox(box2d)
        area = _bbox_area(bbox)
        if area <= 0.0:
            continue
        annotations.append(
            {
                "bbox": bbox,
                "bbox_mode": 0,
                "category_id": BDD100K_CATEGORY_TO_ID[mapped_category],
                "iscrowd": 0,
                "area": area,
            }
        )
    return annotations


class DAODBDD100KDataset(Dataset):
    """Target-side BDD100K detection dataset in Cityscapes class order."""

    thing_classes = BDD100K_THING_CLASSES
    category_to_id = BDD100K_CATEGORY_TO_ID

    def __init__(
        self,
        root: str | Path,
        split: SplitName,
        transform: Callable[[Image.Image], Any] | None = None,
        timeofday: str = "daytime",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.timeofday = str(timeofday).strip()
        self.records = self._build_records()

    def _keep_entry(self, attributes: dict[str, Any]) -> bool:
        if not self.timeofday:
            return True
        return _normalize_token(attributes.get("timeofday", "")) == _normalize_token(self.timeofday)

    def _build_records(self) -> list[BDD100KRecord]:
        image_root = _image_root(self.root, self.split)
        labels_path = _labels_path(self.root, self.split)
        image_index = _build_image_index(image_root)

        missing_images: list[str] = []
        records: list[BDD100KRecord] = []
        for entry in _load_labels(labels_path):
            attributes = entry.get("attributes", {})
            if not self._keep_entry(attributes):
                continue

            image_name = str(entry["name"])
            image_path = image_index.get(image_name)
            if image_path is None:
                missing_images.append(image_name)
                continue

            annotations = _parse_annotations(entry.get("labels", []))
            records.append(
                BDD100KRecord(
                    image_path=str(image_path),
                    sample_id=f"{self.split}:bdd100k:{Path(image_name).stem}",
                    annotations=annotations,
                )
            )

        if missing_images:
            sample = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_images)} labeled BDD100K images under {image_root}; sample: {sample}"
            )
        if not records:
            raise ValueError(f"No BDD100K records found for split={self.split} with timeofday={self.timeofday!r}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def get_sample_id(self, index: int) -> str:
        return self.records[index].sample_id

    def _image_size(self, record: BDD100KRecord) -> tuple[int, int]:
        if record.height is None or record.width is None:
            with Image.open(record.image_path) as img:
                width, height = img.size
            record.height = int(height)
            record.width = int(width)
        return int(record.height), int(record.width)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        height, width = self._image_size(record)
        return {
            "sample_id": record.sample_id,
            "file_name": record.image_path,
            "image_id": record.sample_id,
            "height": height,
            "width": width,
            "annotations": _copy_annotations(record.annotations),
        }


def build_dataset(
    cfg: Any,
    split: SplitName,
    transform: Callable[[Image.Image], Any] | None,
):
    if split in {"source_train", "source_val"}:
        return DAODCityscapesDataset(
            root=resolve_daod_split_root(cfg, split),
            split=split,
            transform=transform,
            foggy_beta=str(getattr(cfg.data, "foggy_beta", "0.02")),
        )

    if split not in VIEW_SPECS:
        raise ValueError(f"Unsupported Cityscapes -> BDD100K split: {split}")

    return DAODBDD100KDataset(
        root=resolve_daod_split_root(cfg, split),
        split=split,
        transform=transform,
        timeofday=str(getattr(cfg.data, "target_timeofday", "daytime")),
    )
