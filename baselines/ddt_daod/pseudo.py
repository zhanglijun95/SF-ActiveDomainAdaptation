"""Pseudo-label helpers for the DINO-adapted DDT baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def deduplicate_rows(rows: list[dict[str, Any]], *, iou_thresh: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        suppress = False
        for kept_row in kept:
            if int(kept_row["category_id"]) != int(row["category_id"]):
                continue
            if xyxy_iou(kept_row["bbox"], row["bbox"]) >= float(iou_thresh):
                suppress = True
                break
        if not suppress:
            kept.append(row)
    return kept


def rows_to_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations = []
    for row in rows:
        x0, y0, x1, y1 = [float(v) for v in row["bbox"]]
        if x1 <= x0 or y1 <= y0:
            continue
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


def filter_pseudo_rows(
    query_rows: list[dict[str, Any]],
    *,
    thresholds: list[float],
    dedup_iou_thresh: float,
) -> list[dict[str, Any]]:
    filtered = []
    for row in query_rows:
        class_id = int(row["category_id"])
        if class_id < 0 or class_id >= len(thresholds):
            continue
        if float(row.get("score", 0.0)) >= float(thresholds[class_id]):
            filtered.append(dict(row))
    return deduplicate_rows(filtered, iou_thresh=dedup_iou_thresh)


def update_dynamic_thresholds(
    thresholds: list[float],
    score_sums: list[float],
    score_counts: list[int],
    *,
    alpha_dt: float,
    gamma_dt: float,
    max_dt: float,
    min_dt: float,
) -> list[float]:
    updated = []
    for threshold, score_sum, count in zip(thresholds, score_sums, score_counts):
        mean_score = float(score_sum) / max(int(count), 1)
        candidate = float(gamma_dt) * float(threshold) + (1.0 - float(gamma_dt)) * float(alpha_dt) * np.sqrt(max(mean_score, 0.0))
        updated.append(float(max(min(candidate, float(max_dt)), float(min_dt))))
    return updated
