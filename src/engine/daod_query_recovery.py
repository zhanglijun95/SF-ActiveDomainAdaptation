"""Sparse-GT guided missed-query recovery for DAOD.

This module approximates the oracle "add back missed objects" diagnostic
without using hidden GT at training time.  Sparse labeled target images define
which teacher queries correspond to objects missed by the normal pseudo-label
threshold, then the learned scorer recovers similar queries on unlabeled target
images.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _sigmoid(value: float) -> float:
    value = float(np.clip(float(value), -40.0, 40.0))
    return float(1.0 / (1.0 + math.exp(-value)))


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
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


def _valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    x0, y0, x1, y1 = [float(v) for v in box]
    return x1 > x0 and y1 > y0


def _box_features(row: dict[str, Any], sample: dict[str, Any]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in row["bbox"]]
    width = max(float(sample.get("width", 1)), 1.0)
    height = max(float(sample.get("height", 1)), 1.0)
    bw = max(x1 - x0, 1e-6)
    bh = max(y1 - y0, 1e-6)
    area_frac = _clip01((bw * bh) / max(width * height, 1.0))
    aspect_log = float(np.clip(math.log(bw / bh), -4.0, 4.0) / 4.0)
    cx = _clip01(((x0 + x1) * 0.5) / width)
    cy = _clip01(((y0 + y1) * 0.5) / height)
    return area_frac, aspect_log, cx, cy


def _enrich_row_features(
    row: dict[str, Any],
    sample: dict[str, Any],
    *,
    threshold: float,
    min_score: float,
    num_views: int,
) -> dict[str, Any]:
    enriched = dict(row)
    area_frac, aspect_log, center_x, center_y = _box_features(enriched, sample)
    score = float(enriched.get("score", 0.0))
    denom = max(float(threshold) - float(min_score), 1e-6)
    enriched["_recovery_score_norm"] = _clip01((score - float(min_score)) / denom)
    enriched["_recovery_below_threshold_gap"] = _clip01((float(threshold) - score) / denom)
    enriched["_area_frac"] = area_frac
    enriched["_aspect_log"] = aspect_log
    enriched["_center_x"] = center_x
    enriched["_center_y"] = center_y
    enriched["_mv_support_frac"] = _clip01(
        float(enriched.get("_mv_support_views", 1)) / max(float(num_views), 1.0)
    )
    enriched["_mv_best_iou"] = _clip01(float(enriched.get("_mv_best_iou", 0.0)))
    enriched["_mv_mean_iou"] = _clip01(float(enriched.get("_mv_mean_iou", 0.0)))
    enriched["_view_index_norm"] = _clip01(float(enriched.get("_recovery_view_index", 0)) / max(float(num_views - 1), 1.0))
    return enriched


def _feature_vector(row: dict[str, Any], *, num_classes: int) -> np.ndarray:
    values = [
        float(row.get("score", 0.0)),
        float(row.get("_recovery_score_norm", 0.0)),
        float(row.get("_recovery_below_threshold_gap", 0.0)),
        float(row.get("softmax_margin", 0.0)),
        1.0 - float(row.get("softmax_entropy", 1.0)),
        1.0 - float(row.get("decoder_box_iou_gap", 1.0)),
        1.0 - float(row.get("decoder_center_shift", 1.0)),
        float(row.get("_area_frac", 0.0)),
        float(row.get("_aspect_log", 0.0)),
        float(row.get("_center_x", 0.5)),
        float(row.get("_center_y", 0.5)),
        float(row.get("_mv_support_frac", 1.0)),
        float(row.get("_mv_best_iou", 0.0)),
        float(row.get("_mv_mean_iou", 0.0)),
        float(row.get("_view_index_norm", 0.0)),
    ]
    class_id = int(row.get("category_id", -1))
    values.extend(1.0 if idx == class_id else 0.0 for idx in range(int(num_classes)))
    return np.asarray(values, dtype=np.float32)


def _deduplicate_rows(
    rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
    existing_rows: list[dict[str, Any]] | None = None,
    score_key: str = "_query_recovery_score",
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    existing = list(existing_rows or [])
    for row in sorted(rows, key=lambda item: float(item.get(score_key, item.get("score", 0.0))), reverse=True):
        suppress = False
        for other in [*existing, *kept]:
            if int(other.get("category_id", -1)) != int(row.get("category_id", -2)):
                continue
            if _xyxy_iou(other["bbox"], row["bbox"]) >= float(iou_thresh):
                suppress = True
                break
        if not suppress:
            kept.append(row)
    return kept


def _standard_pseudo_rows(
    rows: list[dict[str, Any]],
    *,
    thresholds: list[float],
    dedup_iou_thresh: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        class_id = int(row.get("category_id", -1))
        if class_id < 0 or class_id >= len(thresholds):
            continue
        if not _valid_box(row.get("bbox")):
            continue
        if float(row.get("score", 0.0)) >= float(thresholds[class_id]):
            candidates.append(dict(row))
    return _deduplicate_rows(candidates, iou_thresh=dedup_iou_thresh, existing_rows=[], score_key="score")


def _covered_gt_indices(
    pseudo_rows: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    match_iou: float,
) -> set[int]:
    covered: set[int] = set()
    for gt_idx, ann in enumerate(annotations):
        class_id = int(ann.get("category_id", -1))
        for row in pseudo_rows:
            if int(row.get("category_id", -2)) != class_id:
                continue
            if _xyxy_iou(row["bbox"], ann["bbox"]) >= float(match_iou):
                covered.add(int(gt_idx))
                break
    return covered


def _best_iou_to_gt(
    row: dict[str, Any],
    annotations: list[dict[str, Any]],
    *,
    indices: set[int] | None = None,
) -> float:
    class_id = int(row.get("category_id", -1))
    best = 0.0
    for gt_idx, ann in enumerate(annotations):
        if indices is not None and gt_idx not in indices:
            continue
        if int(ann.get("category_id", -2)) != class_id:
            continue
        best = max(best, _xyxy_iou(row["bbox"], ann["bbox"]))
    return float(best)


def _is_duplicate_of_existing(
    row: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> bool:
    class_id = int(row.get("category_id", -1))
    for other in existing_rows:
        if int(other.get("category_id", -2)) != class_id:
            continue
        if _xyxy_iou(row["bbox"], other["bbox"]) >= float(iou_thresh):
            return True
    return False


def _candidate_allowed(
    row: dict[str, Any],
    *,
    thresholds: list[float],
    min_score: float,
    below_threshold_only: bool,
) -> bool:
    class_id = int(row.get("category_id", -1))
    if class_id < 0 or class_id >= len(thresholds):
        return False
    if not _valid_box(row.get("bbox")):
        return False
    score = float(row.get("score", 0.0))
    if score < float(min_score):
        return False
    return not bool(below_threshold_only) or score < float(thresholds[class_id])


@dataclass
class QueryRecoverySelectionStats:
    candidates: int = 0
    selected: int = 0
    score_sum: float = 0.0

    def add(self, other: "QueryRecoverySelectionStats") -> None:
        self.candidates += int(other.candidates)
        self.selected += int(other.selected)
        self.score_sum += float(other.score_sum)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": int(self.candidates),
            "selected": int(self.selected),
            "mean_score": float(self.score_sum / self.selected) if self.selected > 0 else None,
            "selection_rate": float(self.selected / max(self.candidates, 1)),
        }


class QueryRecoveryScorer:
    def __init__(
        self,
        *,
        num_classes: int,
        min_score: float,
        max_per_image: int,
        per_class_max: int,
        below_threshold_only: bool,
        class_thresholds: dict[int, float],
        global_threshold: float | None,
        feature_mean: list[float],
        feature_std: list[float],
        weights: list[float],
        bias: float,
        num_views: int,
        summary: dict[str, Any],
        class_gates: list[float] | None = None,
        class_budgets: list[float] | None = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.min_score = float(min_score)
        self.max_per_image = int(max_per_image)
        self.per_class_max = int(per_class_max)
        self.below_threshold_only = bool(below_threshold_only)
        self.class_thresholds = {int(key): float(value) for key, value in class_thresholds.items()}
        self.global_threshold = None if global_threshold is None else float(global_threshold)
        self.feature_mean = [float(value) for value in feature_mean]
        self.feature_std = [float(value) for value in feature_std]
        self.weights = [float(value) for value in weights]
        self.bias = float(bias)
        self.num_views = int(num_views)
        self._summary = dict(summary)
        self.class_gates = [float(value) for value in (class_gates or [1.0] * self.num_classes)]
        self.class_budgets = None if class_budgets is None else [float(value) for value in class_budgets]

    def summary(self) -> dict[str, Any]:
        return {
            **self._summary,
            "min_score": self.min_score,
            "max_per_image": self.max_per_image,
            "per_class_max": self.per_class_max,
            "below_threshold_only": self.below_threshold_only,
            "global_threshold": self.global_threshold,
            "class_thresholds": {str(key): value for key, value in sorted(self.class_thresholds.items())},
            "num_views": self.num_views,
            "class_gates": [float(value) for value in self.class_gates],
            "class_budgets": None if self.class_budgets is None else [float(value) for value in self.class_budgets],
        }

    def recovery_score(self, row: dict[str, Any], *, threshold: float, sample: dict[str, Any]) -> float:
        enriched = _enrich_row_features(
            row,
            sample,
            threshold=threshold,
            min_score=self.min_score,
            num_views=self.num_views,
        )
        features = _feature_vector(enriched, num_classes=self.num_classes)
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        std = np.maximum(np.asarray(self.feature_std, dtype=np.float32), 1e-6)
        weights = np.asarray(self.weights, dtype=np.float32)
        x = (features - mean) / std
        return _sigmoid(float(np.dot(x, weights) + self.bias))

    def select(
        self,
        query_rows: list[dict[str, Any]],
        *,
        thresholds: list[float],
        dedup_iou_thresh: float,
        sample: dict[str, Any],
        existing_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], QueryRecoverySelectionStats]:
        candidates: list[dict[str, Any]] = []
        candidate_count = 0
        existing = list(existing_rows or [])
        for row in query_rows:
            if not _candidate_allowed(
                row,
                thresholds=thresholds,
                min_score=self.min_score,
                below_threshold_only=self.below_threshold_only,
            ):
                continue
            if _is_duplicate_of_existing(row, existing, iou_thresh=dedup_iou_thresh):
                continue
            candidate_count += 1
            class_id = int(row["category_id"])
            class_gate = self.class_gates[class_id] if 0 <= class_id < len(self.class_gates) else 1.0
            if class_gate <= 0.0:
                continue
            threshold = float(thresholds[class_id])
            recovery_score = self.recovery_score(row, threshold=threshold, sample=sample)
            min_recovery = self.class_thresholds.get(class_id, self.global_threshold)
            if min_recovery is None or recovery_score < float(min_recovery):
                continue
            recovered = dict(row)
            recovered["_query_recovery"] = True
            recovered["_query_recovery_score"] = float(recovery_score)
            recovered["_query_recovery_gate"] = float(class_gate)
            recovered["_pseudo_source"] = "query_recovery"
            candidates.append(recovered)

        selected = _deduplicate_rows(
            candidates,
            iou_thresh=float(dedup_iou_thresh),
            existing_rows=existing,
            score_key="_query_recovery_score",
        )
        if self.per_class_max > 0:
            per_class_counts = [0] * self.num_classes
            capped: list[dict[str, Any]] = []
            for row in selected:
                class_id = int(row.get("category_id", -1))
                if class_id < 0 or class_id >= self.num_classes:
                    continue
                if per_class_counts[class_id] >= self.per_class_max:
                    continue
                per_class_counts[class_id] += 1
                capped.append(row)
            selected = capped
        if self.max_per_image > 0:
            selected = selected[: self.max_per_image]

        stats = QueryRecoverySelectionStats(
            candidates=int(candidate_count),
            selected=len(selected),
            score_sum=float(sum(float(row.get("_query_recovery_score", 0.0)) for row in selected)),
        )
        return selected, stats


def fit_query_recovery_scorer(
    teacher_items: list[dict[str, Any]],
    *,
    thresholds: list[float],
    num_classes: int,
    recovery_cfg: Any,
    seed: int,
    dedup_iou_thresh: float,
) -> QueryRecoveryScorer:
    min_score = float(_cfg_get(recovery_cfg, "min_score", 0.01))
    below_threshold_only = bool(_cfg_get(recovery_cfg, "below_threshold_only", True))
    positive_iou = float(_cfg_get(recovery_cfg, "positive_iou", 0.5))
    negative_iou = float(_cfg_get(recovery_cfg, "negative_iou", 0.3))
    miss_iou = float(_cfg_get(recovery_cfg, "miss_iou", 0.5))
    min_precision = float(_cfg_get(recovery_cfg, "precision_floor", 0.55))
    f_beta = float(_cfg_get(recovery_cfg, "f_beta", 2.0))
    min_class_positives = int(_cfg_get(recovery_cfg, "min_class_positives", 3))
    min_class_candidates = int(_cfg_get(recovery_cfg, "min_class_candidates", 20))
    max_per_image = int(_cfg_get(recovery_cfg, "max_per_image", 8))
    per_class_max = int(_cfg_get(recovery_cfg, "per_class_max", 4))
    num_views = max(1, int(_cfg_get(recovery_cfg, "_resolved_num_views", 1)))

    records = _collect_recovery_records(
        teacher_items,
        thresholds=thresholds,
        num_classes=num_classes,
        min_score=min_score,
        below_threshold_only=below_threshold_only,
        positive_iou=positive_iou,
        negative_iou=negative_iou,
        miss_iou=miss_iou,
        dedup_iou_thresh=dedup_iou_thresh,
        num_views=num_views,
    )
    records = _subsample_records(
        records,
        max_negative_records=int(_cfg_get(recovery_cfg, "max_negative_records", 30000)),
        seed=seed,
    )
    logistic = _fit_logistic(
        records,
        num_classes=num_classes,
        train_steps=int(_cfg_get(recovery_cfg, "train_steps", 300)),
        lr=float(_cfg_get(recovery_cfg, "lr", 0.05)),
        l2=float(_cfg_get(recovery_cfg, "l2", 0.001)),
        max_pos_weight=float(_cfg_get(recovery_cfg, "max_pos_weight", 10.0)),
        seed=seed,
    )
    for record in records:
        record["recovery_score"] = _predict_logistic(record["features"], logistic)
    class_thresholds, global_threshold, threshold_stats = _fit_recovery_thresholds(
        records,
        score_key="recovery_score",
        min_precision=min_precision,
        f_beta=f_beta,
        min_class_positives=min_class_positives,
        min_class_candidates=min_class_candidates,
    )
    class_gates, class_budgets, risk_gate_summary = _fit_risk_gate(
        threshold_stats.get("class_threshold_stats", {}),
        num_classes=num_classes,
        fit_images=len(teacher_items),
        gate_cfg=_cfg_get(recovery_cfg, "risk_gate", {}),
    )

    summary = {
        "fit_images": len(teacher_items),
        "fit_candidates": len(records),
        "fit_positive": int(sum(1 for record in records if int(record["label"]) == 1)),
        "fit_negative": int(sum(1 for record in records if int(record["label"]) == 0)),
        "positive_iou": positive_iou,
        "negative_iou": negative_iou,
        "miss_iou": miss_iou,
        "precision_floor": min_precision,
        "f_beta": f_beta,
        "multi_view": {
            "enabled": bool(_cfg_get(_cfg_get(recovery_cfg, "multi_view", {}), "enabled", False)),
            "views": int(num_views),
            "support_iou": float(_cfg_get(_cfg_get(recovery_cfg, "multi_view", {}), "support_iou", 0.5)),
        },
        "risk_gate": risk_gate_summary,
        "logistic_fit": {key: value for key, value in logistic.items() if key not in {"weights", "bias", "mean", "std"}},
        **threshold_stats,
    }
    return QueryRecoveryScorer(
        num_classes=num_classes,
        min_score=min_score,
        max_per_image=max_per_image,
        per_class_max=per_class_max,
        below_threshold_only=below_threshold_only,
        class_thresholds=class_thresholds,
        global_threshold=global_threshold,
        feature_mean=logistic["mean"],
        feature_std=logistic["std"],
        weights=logistic["weights"],
        bias=logistic["bias"],
        num_views=num_views,
        summary=summary,
        class_gates=class_gates,
        class_budgets=class_budgets,
    )


def _collect_recovery_records(
    teacher_items: list[dict[str, Any]],
    *,
    thresholds: list[float],
    num_classes: int,
    min_score: float,
    below_threshold_only: bool,
    positive_iou: float,
    negative_iou: float,
    miss_iou: float,
    dedup_iou_thresh: float,
    num_views: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in teacher_items:
        sample = item["sample"]
        annotations = [
            {"bbox": [float(v) for v in ann["bbox"]], "category_id": int(ann["category_id"])}
            for ann in sample.get("annotations", [])
            if 0 <= int(ann.get("category_id", -1)) < int(num_classes) and _valid_box(ann.get("bbox"))
        ]
        primary_rows = item.get("primary_query_rows", item.get("query_rows", []))
        standard_rows = _standard_pseudo_rows(
            primary_rows,
            thresholds=thresholds,
            dedup_iou_thresh=dedup_iou_thresh,
        )
        covered = _covered_gt_indices(standard_rows, annotations, match_iou=miss_iou)
        missed = set(range(len(annotations))) - covered
        for row in item.get("query_rows", []):
            if not _candidate_allowed(
                row,
                thresholds=thresholds,
                min_score=min_score,
                below_threshold_only=below_threshold_only,
            ):
                continue
            if _is_duplicate_of_existing(row, standard_rows, iou_thresh=dedup_iou_thresh):
                continue
            class_id = int(row["category_id"])
            threshold = float(thresholds[class_id])
            enriched = _enrich_row_features(
                row,
                sample,
                threshold=threshold,
                min_score=min_score,
                num_views=num_views,
            )
            missed_iou = _best_iou_to_gt(enriched, annotations, indices=missed)
            any_iou = _best_iou_to_gt(enriched, annotations, indices=None)
            if missed_iou >= float(positive_iou):
                label = 1
            elif any_iou <= float(negative_iou) or any_iou >= float(positive_iou):
                label = 0
            else:
                continue
            records.append(
                {
                    "class_id": class_id,
                    "label": int(label),
                    "missed_iou": float(missed_iou),
                    "any_iou": float(any_iou),
                    "features": _feature_vector(enriched, num_classes=num_classes),
                }
            )
    return records


def _subsample_records(records: list[dict[str, Any]], *, max_negative_records: int, seed: int) -> list[dict[str, Any]]:
    positives = [record for record in records if int(record["label"]) == 1]
    negatives = [record for record in records if int(record["label"]) == 0]
    if int(max_negative_records) <= 0 or len(negatives) <= int(max_negative_records):
        return records
    rng = np.random.default_rng(int(seed))
    keep_indices = set(int(idx) for idx in rng.choice(len(negatives), size=int(max_negative_records), replace=False))
    kept_negatives = [record for idx, record in enumerate(negatives) if idx in keep_indices]
    return [*positives, *kept_negatives]


def _fit_logistic(
    records: list[dict[str, Any]],
    *,
    num_classes: int,
    train_steps: int,
    lr: float,
    l2: float,
    max_pos_weight: float,
    seed: int,
) -> dict[str, Any]:
    feature_dim = 15 + int(num_classes)
    if not records:
        return {
            "weights": [0.0 for _ in range(feature_dim)],
            "bias": -20.0,
            "mean": [0.0 for _ in range(feature_dim)],
            "std": [1.0 for _ in range(feature_dim)],
            "trained": False,
            "reason": "no_records",
        }
    x = np.stack([record["features"] for record in records]).astype(np.float32)
    y = np.asarray([float(record["label"]) for record in records], dtype=np.float32)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    x_norm = (x - mean) / std
    if positives <= 0 or negatives <= 0:
        return {
            "weights": [0.0 for _ in range(x.shape[1])],
            "bias": 20.0 if positives > 0 else -20.0,
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "trained": False,
            "reason": "single_class_labels",
            "positives": positives,
            "negatives": negatives,
        }

    torch.manual_seed(int(seed))
    x_t = torch.as_tensor(x_norm, dtype=torch.float32)
    y_t = torch.as_tensor(y[:, None], dtype=torch.float32)
    linear = torch.nn.Linear(x_t.shape[1], 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=float(lr))
    pos_weight_value = min(max(float(negatives) / max(float(positives), 1.0), 1.0), float(max_pos_weight))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
    for _ in range(max(int(train_steps), 1)):
        logits = linear(x_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        if l2 > 0.0:
            loss = loss + float(l2) * linear.weight.pow(2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = torch.sigmoid(linear(x_t))
        train_loss = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), y_t)
    return {
        "weights": linear.weight.detach().cpu().numpy().reshape(-1).astype(float).tolist(),
        "bias": float(linear.bias.detach().cpu().item()),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "trained": True,
        "positives": positives,
        "negatives": negatives,
        "pos_weight": float(pos_weight_value),
        "train_loss": float(train_loss.detach().cpu().item()),
    }


def _predict_logistic(features: np.ndarray, logistic: dict[str, Any]) -> float:
    mean = np.asarray(logistic["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(logistic["std"], dtype=np.float32), 1e-6)
    weights = np.asarray(logistic["weights"], dtype=np.float32)
    x = (features.astype(np.float32) - mean) / std
    return _sigmoid(float(np.dot(x, weights) + float(logistic["bias"])))


def _fit_recovery_thresholds(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    min_precision: float,
    f_beta: float,
    min_class_positives: int,
    min_class_candidates: int,
) -> tuple[dict[int, float], float | None, dict[str, Any]]:
    global_threshold, global_stats = _best_recovery_threshold(
        records,
        score_key=score_key,
        min_precision=min_precision,
        f_beta=f_beta,
    )
    class_thresholds: dict[int, float] = {}
    class_stats: dict[str, Any] = {}
    for class_id in sorted({int(record["class_id"]) for record in records}):
        class_records = [record for record in records if int(record["class_id"]) == class_id]
        positives = int(sum(1 for record in class_records if int(record["label"]) == 1))
        if positives < int(min_class_positives) or len(class_records) < int(min_class_candidates):
            class_stats[str(class_id)] = {
                "threshold": None,
                "num_records": len(class_records),
                "positives": positives,
                "fallback": "global",
            }
            continue
        threshold, stats = _best_recovery_threshold(
            class_records,
            score_key=score_key,
            min_precision=min_precision,
            f_beta=f_beta,
        )
        if threshold is not None:
            class_thresholds[class_id] = float(threshold)
        class_stats[str(class_id)] = stats
    return class_thresholds, global_threshold, {
        "global_threshold_stats": global_stats,
        "class_threshold_stats": class_stats,
    }


def _fit_risk_gate(
    class_threshold_stats: dict[str, Any],
    *,
    num_classes: int,
    fit_images: int,
    gate_cfg: Any,
) -> tuple[list[float], list[float] | None, dict[str, Any]]:
    """Convert sparse-label recovery audit stats into per-class safety gates."""

    if not bool(_cfg_get(gate_cfg, "enabled", False)):
        return [1.0 for _ in range(int(num_classes))], None, {"enabled": False}

    min_precision = float(_cfg_get(gate_cfg, "min_precision", 0.55))
    min_recall = float(_cfg_get(gate_cfg, "min_recall", 0.02))
    min_total_positive = int(_cfg_get(gate_cfg, "min_total_positive", 10))
    min_selected = int(_cfg_get(gate_cfg, "min_selected", 5))
    precision_power = max(float(_cfg_get(gate_cfg, "precision_power", 1.0)), 0.0)
    recall_power = max(float(_cfg_get(gate_cfg, "recall_power", 0.5)), 0.0)
    normalize = bool(_cfg_get(gate_cfg, "normalize", True))
    gate_floor = float(_cfg_get(gate_cfg, "gate_floor", 0.0))
    gate_max = float(_cfg_get(gate_cfg, "gate_max", 1.0))
    budget_cfg = _cfg_get(gate_cfg, "budget", {})
    budget_enabled = bool(_cfg_get(budget_cfg, "enabled", False))
    budget_scale = float(_cfg_get(budget_cfg, "scale", 0.5))
    budget_min = float(_cfg_get(budget_cfg, "min_budget", 0.0))
    budget_max = float(_cfg_get(budget_cfg, "max_budget", 2.0))

    raw_gates: list[float] = []
    class_stats: dict[str, Any] = {}
    for class_id in range(int(num_classes)):
        stats = class_threshold_stats.get(str(class_id), {})
        selected = int(stats.get("selected", 0) or 0)
        positives = int(stats.get("positives", 0) or 0)
        total_positive = int(stats.get("total_positive", stats.get("positives", 0)) or 0)
        precision = _safe_float(stats.get("precision"))
        recall = _safe_float(stats.get("recall"))
        if precision is None and selected > 0:
            precision = float(positives / max(selected, 1))
        if recall is None and total_positive > 0:
            recall = float(positives / max(total_positive, 1))

        disabled_reason = None
        if total_positive < min_total_positive:
            disabled_reason = "insufficient_positive_support"
        elif selected < min_selected:
            disabled_reason = "insufficient_selected_support"
        elif precision is None or precision < min_precision:
            disabled_reason = "low_precision"
        elif recall is None or recall < min_recall:
            disabled_reason = "low_recall"

        if disabled_reason is None:
            raw_gate = float((precision ** precision_power) * (recall ** recall_power))
        else:
            raw_gate = 0.0
        raw_gates.append(raw_gate)
        class_stats[str(class_id)] = {
            "selected": int(selected),
            "positives": int(positives),
            "total_positive": int(total_positive),
            "precision": precision,
            "recall": recall,
            "raw_gate": float(raw_gate),
            "disabled_reason": disabled_reason,
        }

    max_gate = max(raw_gates) if raw_gates else 0.0
    if normalize and max_gate > 0.0:
        gates = [gate / max_gate for gate in raw_gates]
    else:
        gates = list(raw_gates)
    gates = [float(np.clip(max(gate, gate_floor), 0.0, gate_max)) for gate in gates]

    budgets = None
    if budget_enabled:
        budgets = []
        denom_images = max(int(fit_images), 1)
        for class_id, gate in enumerate(gates):
            if gate <= 0.0:
                budgets.append(0.0)
                continue
            selected = float(class_stats[str(class_id)]["selected"])
            per_image_need = selected / float(denom_images)
            budget = float(budget_scale) * float(gate) * per_image_need
            budget = max(budget, budget_min)
            budget = min(budget, budget_max)
            budgets.append(float(max(budget, 0.0)))

    enabled_classes = []
    for class_id, gate in enumerate(gates):
        class_stats[str(class_id)]["gate"] = float(gate)
        class_stats[str(class_id)]["budget"] = None if budgets is None else float(budgets[class_id])
        if gate > 0.0:
            enabled_classes.append(int(class_id))

    return gates, budgets, {
        "enabled": True,
        "method": "threshold_stats_risk_gate",
        "min_precision": float(min_precision),
        "min_recall": float(min_recall),
        "min_total_positive": int(min_total_positive),
        "min_selected": int(min_selected),
        "precision_power": float(precision_power),
        "recall_power": float(recall_power),
        "normalize": bool(normalize),
        "gate_floor": float(gate_floor),
        "gate_max": float(gate_max),
        "budget_enabled": bool(budget_enabled),
        "budget_scale": float(budget_scale) if budget_enabled else None,
        "budget_min": float(budget_min) if budget_enabled else None,
        "budget_max": float(budget_max) if budget_enabled else None,
        "enabled_classes": enabled_classes,
        "class_stats": class_stats,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _best_recovery_threshold(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    min_precision: float,
    f_beta: float,
) -> tuple[float | None, dict[str, Any]]:
    usable = [
        record
        for record in records
        if isinstance(record.get(score_key), (int, float)) and math.isfinite(float(record[score_key]))
    ]
    usable.sort(key=lambda record: float(record[score_key]), reverse=True)
    total_positive = int(sum(1 for record in usable if int(record["label"]) == 1))
    best: dict[str, Any] | None = None
    positives_so_far = 0
    beta2 = float(f_beta) ** 2
    for selected, record in enumerate(usable, start=1):
        positives_so_far += int(record["label"]) == 1
        precision = float(positives_so_far / selected)
        recall = float(positives_so_far / max(total_positive, 1))
        if precision < float(min_precision):
            continue
        denom = beta2 * precision + recall
        f_score = 0.0 if denom <= 0.0 else float((1.0 + beta2) * precision * recall / denom)
        if best is None or f_score > best["f_score"] or (
            math.isclose(f_score, best["f_score"]) and recall > best["recall"]
        ):
            best = {
                "threshold": float(record[score_key]),
                "selected": int(selected),
                "positives": int(positives_so_far),
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
    stats = {
        "threshold": None if best is None else best["threshold"],
        "selected": 0 if best is None else best["selected"],
        "positives": 0 if best is None else best["positives"],
        "precision": None if best is None else best["precision"],
        "recall": None if best is None else best["recall"],
        "f_score": None if best is None else best["f_score"],
        "num_records": len(usable),
        "total_positive": total_positive,
    }
    return (None if best is None else float(best["threshold"])), stats


def merge_multiview_teacher_items(
    primary_items: list[dict[str, Any]],
    extra_view_items: list[list[dict[str, Any]]],
    *,
    support_iou: float,
) -> list[dict[str, Any]]:
    """Merge primary and extra weak-view teacher rows into one candidate pool."""

    all_views = [primary_items, *extra_view_items]
    num_views = len(all_views)
    merged_items: list[dict[str, Any]] = []
    for item_idx, primary in enumerate(primary_items):
        rows: list[dict[str, Any]] = []
        for view_idx, view_items in enumerate(all_views):
            if item_idx >= len(view_items):
                continue
            for row_idx, row in enumerate(view_items[item_idx].get("query_rows", [])):
                merged = dict(row)
                merged["_recovery_view_index"] = int(view_idx)
                merged["_recovery_row_index"] = int(row_idx)
                rows.append(merged)
        _annotate_multiview_support(rows, num_views=num_views, support_iou=support_iou)
        merged_items.append(
            {
                "sample": primary["sample"],
                "raw_output": primary.get("raw_output"),
                "primary_query_rows": [dict(row) for row in primary.get("query_rows", [])],
                "query_rows": rows,
                "num_views": int(num_views),
            }
        )
    return merged_items


def _annotate_multiview_support(rows: list[dict[str, Any]], *, num_views: int, support_iou: float) -> None:
    if not rows:
        return
    for row in rows:
        class_id = int(row.get("category_id", -1))
        support_views = {int(row.get("_recovery_view_index", 0))}
        ious: list[float] = []
        for other in rows:
            if other is row:
                continue
            if int(other.get("category_id", -2)) != class_id:
                continue
            iou = _xyxy_iou(row["bbox"], other["bbox"])
            if iou >= float(support_iou):
                support_views.add(int(other.get("_recovery_view_index", 0)))
                ious.append(float(iou))
        row["_mv_support_views"] = int(len(support_views))
        row["_mv_support_frac"] = _clip01(float(len(support_views)) / max(float(num_views), 1.0))
        row["_mv_best_iou"] = float(max(ious)) if ious else 0.0
        row["_mv_mean_iou"] = float(np.mean(ious)) if ious else 0.0
