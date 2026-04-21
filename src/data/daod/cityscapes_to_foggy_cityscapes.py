"""Cityscapes -> Foggy Cityscapes DAOD dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


SplitName = str

CITYSCAPES_THING_CLASSES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
CITYSCAPES_CATEGORY_TO_ID = {name: idx for idx, name in enumerate(CITYSCAPES_THING_CLASSES)}

VIEW_SPECS = {
    "source_train": {"domain": "source", "split": "train", "foggy": False},
    "source_val": {"domain": "source", "split": "val", "foggy": False},
    "target_train": {"domain": "target", "split": "train", "foggy": True},
    "target_val": {"domain": "target", "split": "val", "foggy": True},
}


@dataclass
class DAODRecord:
    image_path: str
    annotation_path: str
    sample_id: str


def _annotation_root(root: Path) -> Path:
    return root / "gtFine"


def _image_root(root: Path, foggy: bool) -> Path:
    return root / ("leftImg8bit_foggy" if foggy else "leftImg8bit")


def _load_annotation(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _polygon_to_bbox(polygon: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _polygon_to_segmentation(polygon: list[list[float]]) -> list[list[float]]:
    flat: list[float] = []
    for x, y in polygon:
        flat.extend([float(x), float(y)])
    return [flat]


def _polygon_area(polygon: list[list[float]]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _parse_annotations(annotation_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    payload = _load_annotation(annotation_path)
    annotations: list[dict[str, Any]] = []
    for obj in payload["objects"]:
        label = obj["label"]
        if label not in CITYSCAPES_CATEGORY_TO_ID:
            continue
        polygon = obj["polygon"]
        if len(polygon) < 3:
            continue
        annotations.append(
            {
                "bbox": _polygon_to_bbox(polygon),
                "bbox_mode": 0,
                "category_id": CITYSCAPES_CATEGORY_TO_ID[label],
                "segmentation": _polygon_to_segmentation(polygon),
                "iscrowd": 0,
                "area": _polygon_area(polygon),
            }
        )
    return annotations, int(payload["imgHeight"]), int(payload["imgWidth"])


def _build_sample_id(view_name: str, city: str, stem: str, foggy_beta: str | None) -> str:
    if foggy_beta is None:
        return f"{view_name}:{city}:{stem}"
    return f"{view_name}:{city}:{stem}:beta={foggy_beta}"


def resolve_daod_split_root(cfg: Any, split: SplitName) -> Path:
    spec = VIEW_SPECS[split]
    root_attr = "source_root" if str(spec["domain"]) == "source" else "target_root"
    root_value = getattr(getattr(cfg, "data", object()), root_attr, None)
    if root_value is not None and str(root_value).strip():
        return Path(str(root_value))
    return Path(str(cfg.data.root))


class DAODCityscapesDataset(Dataset):
    """Single-view detection dataset compatible with existing wrappers."""

    thing_classes = CITYSCAPES_THING_CLASSES
    category_to_id = CITYSCAPES_CATEGORY_TO_ID

    def __init__(
        self,
        root: str | Path,
        split: SplitName,
        transform: Callable[[Image.Image], Any] | None = None,
        foggy_beta: str = "0.02",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.foggy_beta = str(foggy_beta)
        self.spec = VIEW_SPECS[split]
        self.records = self._build_records()

    def _build_records(self) -> list[DAODRecord]:
        image_root = _image_root(self.root, foggy=bool(self.spec["foggy"])) / str(self.spec["split"])
        ann_root = _annotation_root(self.root) / str(self.spec["split"])
        records: list[DAODRecord] = []
        for city_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            ann_city_dir = ann_root / city_dir.name
            for image_path in sorted(path for path in city_dir.iterdir() if path.is_file()):
                stem = self._stem_from_image_name(image_path.name)
                if stem is None:
                    continue
                ann_path = ann_city_dir / f"{stem}_gtFine_polygons.json"
                records.append(
                    DAODRecord(
                        image_path=str(image_path),
                        annotation_path=str(ann_path),
                        sample_id=_build_sample_id(
                            self.split,
                            city_dir.name,
                            stem,
                            self.foggy_beta if self.spec["foggy"] else None,
                        ),
                    )
                )
        return records

    def _stem_from_image_name(self, image_name: str) -> str | None:
        if not self.spec["foggy"]:
            suffix = "_leftImg8bit.png"
            if image_name.endswith(suffix):
                return image_name[: -len(suffix)]
            return None

        suffix = f"_leftImg8bit_foggy_beta_{self.foggy_beta}.png"
        if image_name.endswith(suffix):
            return image_name[: -len(suffix)]
        return None

    def __len__(self) -> int:
        return len(self.records)

    def get_sample_id(self, index: int) -> str:
        return self.records[index].sample_id

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        annotations, height, width = _parse_annotations(Path(record.annotation_path))
        return {
            "sample_id": record.sample_id,
            "file_name": record.image_path,
            "image_id": record.sample_id,
            "height": height,
            "width": width,
            "annotations": annotations,
        }


def build_dataset(
    cfg: Any,
    split: SplitName,
    transform: Callable[[Image.Image], Any] | None,
) -> DAODCityscapesDataset:
    return DAODCityscapesDataset(
        root=resolve_daod_split_root(cfg, split),
        split=split,
        transform=transform,
        foggy_beta=str(getattr(cfg.data, "foggy_beta", "0.02")),
    )
