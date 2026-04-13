"""Baseline-local geometry and matching utilities."""

from __future__ import annotations

from typing import Any


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _greedy_class_aware_matches(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt in enumerate(gt_rows):
        for pred_idx, pred in enumerate(pred_rows):
            if int(gt["category_id"]) != int(pred["category_id"]):
                continue
            iou = xyxy_iou(gt["bbox"], pred["bbox"])
            if iou >= iou_thresh:
                candidates.append((iou, gt_idx, pred_idx))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx))

    unmatched_gt = [idx for idx in range(len(gt_rows)) if idx not in used_gt]
    unmatched_pred = [idx for idx in range(len(pred_rows)) if idx not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def count_false_negatives(
    gt_annotations: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
    score_floor: float = 0.0,
) -> int:
    gt_rows = [
        {
            "bbox": [float(v) for v in ann["bbox"]],
            "category_id": int(ann["category_id"]),
        }
        for ann in gt_annotations
    ]
    filtered_preds = [
        {
            "bbox": [float(v) for v in row["bbox"]],
            "category_id": int(row["category_id"]),
            "score": float(row.get("score", 0.0)),
        }
        for row in pred_rows
        if float(row.get("score", 0.0)) >= float(score_floor)
    ]
    _, unmatched_gt, _ = _greedy_class_aware_matches(gt_rows, filtered_preds, iou_thresh=iou_thresh)
    return int(len(unmatched_gt))


def deduplicate_rows(rows: list[dict[str, Any]], *, iou_thresh: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    kept: list[dict[str, Any]] = []
    rows_sorted = sorted(rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    for row in rows_sorted:
        should_drop = False
        for kept_row in kept:
            if int(kept_row["category_id"]) != int(row["category_id"]):
                continue
            if xyxy_iou(kept_row["bbox"], row["bbox"]) >= iou_thresh:
                should_drop = True
                break
        if not should_drop:
            kept.append(row)
    return kept


def rows_to_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for row in rows:
        x0, y0, x1, y1 = [float(v) for v in row["bbox"]]
        annotations.append(
            {
                "bbox": [x0, y0, x1, y1],
                "bbox_mode": 0,
                "category_id": int(row["category_id"]),
                "iscrowd": 0,
                "area": max(0.0, x1 - x0) * max(0.0, y1 - y0),
            }
        )
    return annotations
