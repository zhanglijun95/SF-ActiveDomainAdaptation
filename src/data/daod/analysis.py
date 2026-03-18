"""Small DAOD target-analysis helpers used by the target-val notebook.

The goal of this module is to keep the notebook readable while avoiding a large
analysis pipeline. The functions here are intentionally simple and explicit:
they operate on one image at a time and return plain Python dict/list outputs
that are easy to inspect in a notebook.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two `[x0, y0, x1, y1]` boxes."""

    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def instances_to_prediction_rows(
    instances: Any,
    thing_classes: tuple[str, ...] | list[str],
    *,
    score_thresh: float,
) -> list[dict[str, Any]]:
    """Convert detectron2 `Instances` predictions into plain row dicts.

    Inputs:
    - `instances`: detectron2 prediction output with `pred_boxes`, `scores`,
      and `pred_classes`
    - `thing_classes`: class-name lookup for contiguous category ids
    - `score_thresh`: predictions below this score are dropped

    Output:
    - a list of rows, one per retained prediction, each with:
      `bbox`, `score`, `category_id`, and `category_name`
    """

    if hasattr(instances, "to"):
        instances = instances.to("cpu")

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    rows: list[dict[str, Any]] = []
    for box, score, category_id in zip(boxes, scores, classes):
        score = float(score)
        if score < score_thresh:
            continue
        category_id = int(category_id)
        rows.append(
            {
                "bbox": [float(v) for v in box.tolist()],
                "score": score,
                "category_id": category_id,
                "category_name": thing_classes[category_id],
            }
        )
    return rows


def greedy_match_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
    class_aware: bool,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Greedily match two detection sets by highest IoU.

    Inputs:
    - `left_rows`, `right_rows`: prediction rows with `bbox` and `category_id`
    - `iou_thresh`: minimum IoU required for a match
    - `class_aware`: if true, only same-class pairs may match

    Output:
    - list of matched pair dicts with indices and IoU
    - unmatched indices from `left_rows`
    - unmatched indices from `right_rows`

    Assumptions:
    - matching is intentionally simple and explainable
    - greedy highest-IoU matching is good enough for notebook-level analysis
      even though it is not the only possible assignment rule
    """

    candidates: list[tuple[float, int, int]] = []
    for left_idx, left in enumerate(left_rows):
        for right_idx, right in enumerate(right_rows):
            if class_aware and left["category_id"] != right["category_id"]:
                continue
            iou = xyxy_iou(left["bbox"], right["bbox"])
            if iou >= iou_thresh:
                candidates.append((iou, left_idx, right_idx))

    candidates.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, left_idx, right_idx in candidates:
        if left_idx in used_left or right_idx in used_right:
            continue
        used_left.add(left_idx)
        used_right.add(right_idx)
        matches.append(
            {
                "left_idx": left_idx,
                "right_idx": right_idx,
                "iou": float(iou),
                "left": left_rows[left_idx],
                "right": right_rows[right_idx],
            }
        )

    unmatched_left = [idx for idx in range(len(left_rows)) if idx not in used_left]
    unmatched_right = [idx for idx in range(len(right_rows)) if idx not in used_right]
    return matches, unmatched_left, unmatched_right


def match_predictions_to_gt(
    gt_annotations: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Match predictions to GT with class-aware greedy IoU matching.

    Output:
    - `tp_rows`: matched GT/prediction pairs
    - `fp_rows`: predictions left unmatched after GT matching
    - `fn_rows`: GT objects left unmatched after prediction matching

    Score meaning:
    - this produces the simple TP / FP / FN counts used to define notebook
      ground-truth difficulty scores at the image level.
    """

    gt_rows = [
        {
            "bbox": [float(v) for v in ann["bbox"]],
            "category_id": int(ann["category_id"]),
            "area": float(ann.get("area", 0.0)),
        }
        for ann in gt_annotations
    ]

    matches, unmatched_gt, unmatched_pred = greedy_match_rows(
        gt_rows,
        pred_rows,
        iou_thresh=iou_thresh,
        class_aware=True,
    )

    tp_rows = [
        {
            "gt_idx": match["left_idx"],
            "pred_idx": match["right_idx"],
            "iou": match["iou"],
            "gt": match["left"],
            "pred": match["right"],
        }
        for match in matches
    ]
    fp_rows = [pred_rows[idx] for idx in unmatched_pred]
    fn_rows = [gt_rows[idx] for idx in unmatched_gt]
    return tp_rows, fp_rows, fn_rows


def compute_proxy_summary(
    original_rows: list[dict[str, Any]],
    weak_rows: list[dict[str, Any]],
    strong_rows: list[dict[str, Any]],
    *,
    weak_strong_iou_thresh: float,
) -> dict[str, float]:
    """Compute image-level no-GT proxy scores from model outputs only.

    Inputs:
    - prediction rows from the original, weak, and strong views for one image
    - `weak_strong_iou_thresh`: IoU threshold used for weak/strong matching

    Output:
    - a flat dict of image-level proxy scores

    Score meanings:
    - confidence proxies: low-confidence mass, low-confidence fraction, score
      variance, and prediction count
    - disagreement proxies: unmatched weak/strong predictions, mean matched IoU,
      class-disagreement count, and score-difference statistics

    Assumptions:
    - weak and strong boxes are already expressed in the original image
      coordinate system before calling this function
    """

    orig_scores = np.asarray([row["score"] for row in original_rows], dtype=float)
    low_conf_mask = orig_scores < 0.5 if len(orig_scores) else np.asarray([], dtype=bool)

    matches, unmatched_weak, unmatched_strong = greedy_match_rows(
        weak_rows,
        strong_rows,
        iou_thresh=weak_strong_iou_thresh,
        class_aware=False,
    )

    matched_ious = np.asarray([match["iou"] for match in matches], dtype=float)
    matched_score_diffs = np.asarray(
        [abs(match["left"]["score"] - match["right"]["score"]) for match in matches],
        dtype=float,
    )
    class_disagree_count = sum(
        int(match["left"]["category_id"] != match["right"]["category_id"])
        for match in matches
    )

    pred_count = float(len(original_rows))
    match_count = float(len(matches))
    disagreement_count = float(len(unmatched_weak) + len(unmatched_strong) + class_disagree_count)
    low_conf_count = float(low_conf_mask.sum()) if len(orig_scores) else 0.0

    return {
        "proxy_pred_count": pred_count,
        "proxy_mean_score": float(orig_scores.mean()) if len(orig_scores) else 0.0,
        "proxy_score_std": float(orig_scores.std()) if len(orig_scores) else 0.0,
        "proxy_low_conf_count": low_conf_count,
        "proxy_low_conf_frac": float(low_conf_count / pred_count) if pred_count else 0.0,
        "proxy_low_conf_mass": float(np.maximum(0.0, 0.5 - orig_scores).sum()) if len(orig_scores) else 0.0,
        "proxy_ws_match_count": match_count,
        "proxy_ws_unmatched_count": float(len(unmatched_weak) + len(unmatched_strong)),
        "proxy_ws_unmatched_frac": float((len(unmatched_weak) + len(unmatched_strong)) / max(1.0, len(weak_rows) + len(strong_rows))),
        "proxy_ws_mean_iou": float(matched_ious.mean()) if len(matched_ious) else 0.0,
        "proxy_ws_iou_gap": float(1.0 - matched_ious.mean()) if len(matched_ious) else 1.0,
        "proxy_ws_mean_score_diff": float(matched_score_diffs.mean()) if len(matched_score_diffs) else 0.0,
        "proxy_ws_class_disagree_count": float(class_disagree_count),
        "proxy_ws_class_disagree_frac": float(class_disagree_count / match_count) if match_count else 0.0,
        "proxy_ws_disagreement_count": disagreement_count,
    }


def zscore(values: list[float] | np.ndarray) -> np.ndarray:
    """Return a stable z-score array, falling back to zeros for constant input."""

    values = np.asarray(values, dtype=float)
    std = values.std()
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std
