"""DAOD data package."""

from .cityscapes_to_foggy_cityscapes import (
    CITYSCAPES_CATEGORY_TO_ID,
    CITYSCAPES_THING_CLASSES,
    DAODCityscapesDataset,
    build_dataset as build_cityscapes_to_foggy_cityscapes_dataset,
)
from .analysis import (
    classify_detection_errors,
    compute_decoder_proxy_summary,
    compute_logit_proxy_summary,
    compute_proxy_summary,
    greedy_match_rows,
    instances_to_prediction_rows,
    match_predictions_to_gt,
    xyxy_iou,
    zscore,
)
from .detectron2 import (
    build_daod_detection_test_loader,
    build_daod_detection_train_loader,
    export_daod_coco_json,
    materialize_daod_dicts,
    register_daod_eval_dataset,
)
from .pairs import build_daod_dataset, get_daod_thing_classes
from .transforms import (
    build_strong_view_transform,
    build_weak_view_transform,
    make_strong_view,
    make_weak_view,
    map_boxes_to_original_view,
)

__all__ = [
    "CITYSCAPES_THING_CLASSES",
    "CITYSCAPES_CATEGORY_TO_ID",
    "DAODCityscapesDataset",
    "build_cityscapes_to_foggy_cityscapes_dataset",
    "xyxy_iou",
    "greedy_match_rows",
    "match_predictions_to_gt",
    "classify_detection_errors",
    "instances_to_prediction_rows",
    "compute_proxy_summary",
    "compute_logit_proxy_summary",
    "compute_decoder_proxy_summary",
    "zscore",
    "build_daod_dataset",
    "get_daod_thing_classes",
    "materialize_daod_dicts",
    "build_daod_detection_train_loader",
    "build_daod_detection_test_loader",
    "export_daod_coco_json",
    "register_daod_eval_dataset",
    "build_weak_view_transform",
    "build_strong_view_transform",
    "make_weak_view",
    "make_strong_view",
    "map_boxes_to_original_view",
]
