"""Detectron2/detrex data bridge for DAOD source training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import detectron2.data.transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detrex.data import DetrDatasetMapper

from .pairs import build_daod_dataset, get_daod_thing_classes


def materialize_daod_dicts(cfg: Any, split: str) -> list[dict[str, Any]]:
    dataset = build_daod_dataset(cfg, split, transform=None)
    dicts: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dict(dataset[index])
        sample["image_id"] = index + 1
        dicts.append(sample)
    return dicts


def _build_train_mapper() -> DetrDatasetMapper:
    return DetrDatasetMapper(
        augmentation=[
            T.RandomFlip(),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            T.RandomFlip(),
            T.ResizeShortestEdge(short_edge_length=(400, 500, 600), sample_style="choice"),
            T.RandomCrop(crop_type="absolute_range", crop_size=(384, 600)),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    )


def _build_test_mapper(min_size_test: int, max_size_test: int) -> DetrDatasetMapper:
    return DetrDatasetMapper(
        augmentation=[
            T.ResizeShortestEdge(short_edge_length=min_size_test, max_size=max_size_test),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    )


def build_daod_detection_train_loader(cfg: Any, dataset_dicts: list[dict[str, Any]]):
    return build_detection_train_loader(
        dataset=dataset_dicts,
        mapper=_build_train_mapper(),
        total_batch_size=int(cfg.train.batch_size),
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
    )


def build_daod_detection_test_loader(
    cfg: Any,
    dataset_dicts: list[dict[str, Any]],
    *,
    min_size_test: int = 800,
    max_size_test: int = 1333,
):
    return build_detection_test_loader(
        dataset=dataset_dicts,
        mapper=_build_test_mapper(min_size_test=min_size_test, max_size_test=max_size_test),
        num_workers=int(getattr(cfg.eval, "num_workers", 4)),
    )


def export_daod_coco_json(cfg: Any, dataset_dicts: list[dict[str, Any]], json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    thing_classes = list(get_daod_thing_classes(cfg))

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx + 1, "name": name}
            for idx, name in enumerate(thing_classes)
        ],
    }

    ann_id = 1
    for sample in dataset_dicts:
        coco["images"].append(
            {
                "id": int(sample["image_id"]),
                "file_name": sample["file_name"],
                "height": int(sample["height"]),
                "width": int(sample["width"]),
            }
        )
        for ann in sample["annotations"]:
            x0, y0, x1, y1 = ann["bbox"]
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": int(sample["image_id"]),
                    "category_id": int(ann["category_id"]) + 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "area": float(ann["area"]),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
            ann_id += 1

    json_path.write_text(json.dumps(coco), encoding="utf-8")
    return json_path


def register_daod_eval_dataset(
    name: str,
    cfg: Any,
    dataset_dicts: list[dict[str, Any]],
    json_path: str | Path,
) -> None:
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name, lambda data=dataset_dicts: data)

    thing_classes = list(get_daod_thing_classes(cfg))
    metadata = MetadataCatalog.get(name)
    metadata.json_file = str(json_path)
    metadata.thing_classes = thing_classes
    metadata.thing_dataset_id_to_contiguous_id = {idx + 1: idx for idx in range(len(thing_classes))}
