"""Sparse-GT calibrated latent query activation for DINO-style DAOD.

The detector is unchanged.  This module learns a training-time query selector
from sparse target labels, then activates below-threshold teacher queries that
look like real target objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


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


def _best_iou_for_predicted_class(row: dict[str, Any], annotations: list[dict[str, Any]]) -> float:
    class_id = int(row["category_id"])
    box = [float(v) for v in row["bbox"]]
    best = 0.0
    for ann in annotations:
        if int(ann["category_id"]) != class_id:
            continue
        best = max(best, _xyxy_iou(box, [float(v) for v in ann["bbox"]]))
    return float(best)


def _deduplicate_rows(
    rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
    existing_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    existing = existing_rows or []
    for row in sorted(rows, key=lambda item: float(item.get("_latent_activation_score", item.get("score", 0.0))), reverse=True):
        suppress = False
        for other in [*existing, *kept]:
            if int(other["category_id"]) != int(row["category_id"]):
                continue
            if _xyxy_iou(other["bbox"], row["bbox"]) >= float(iou_thresh):
                suppress = True
                break
        if not suppress:
            kept.append(row)
    return kept


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


def _query_quality_score(
    row: dict[str, Any],
    *,
    threshold: float,
    min_score: float,
    weights: dict[str, float],
) -> float:
    score = float(row.get("score", 0.0))
    denom = max(float(threshold) - float(min_score), 1e-6)
    score_norm = _clip01((score - float(min_score)) / denom)
    margin = _clip01(float(row.get("softmax_margin", 0.0)))
    confidence = 1.0 - _clip01(float(row.get("softmax_entropy", 1.0)))
    box_stability = 1.0 - _clip01(float(row.get("decoder_box_iou_gap", 1.0)))
    center_stability = 1.0 - _clip01(float(row.get("decoder_center_shift", 1.0)))
    features = {
        "score": score_norm,
        "margin": margin,
        "confidence": confidence,
        "box_stability": box_stability,
        "center_stability": center_stability,
    }
    weight_sum = 0.0
    value_sum = 0.0
    for name, value in features.items():
        weight = max(float(weights.get(name, 0.0)), 0.0)
        if weight <= 0.0:
            continue
        value_sum += weight * float(value)
        weight_sum += weight
    return float(value_sum / max(weight_sum, 1e-12))


@dataclass
class LatentActivationSelectionStats:
    candidates: int = 0
    activated: int = 0
    score_sum: float = 0.0

    def update(self, *, candidates: int, activated_scores: list[float]) -> None:
        self.candidates += int(candidates)
        self.activated += len(activated_scores)
        self.score_sum += float(sum(float(v) for v in activated_scores))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": int(self.candidates),
            "activated": int(self.activated),
            "mean_activation_score": float(self.score_sum / self.activated) if self.activated > 0 else None,
        }


class LatentQueryActivator:
    def __init__(
        self,
        *,
        method: str,
        num_classes: int,
        min_score: float,
        max_per_image: int,
        class_thresholds: dict[int, float],
        global_threshold: float | None,
        precision_target: float,
        positive_iou: float,
        negative_iou: float,
        summary: dict[str, Any],
        quality_weights: dict[str, float] | None = None,
        logistic_weights: list[float] | None = None,
        logistic_bias: float | None = None,
        feature_mean: list[float] | None = None,
        feature_std: list[float] | None = None,
    ) -> None:
        self.method = method
        self.num_classes = int(num_classes)
        self.min_score = float(min_score)
        self.max_per_image = int(max_per_image)
        self.class_thresholds = {int(k): float(v) for k, v in class_thresholds.items()}
        self.global_threshold = None if global_threshold is None else float(global_threshold)
        self.precision_target = float(precision_target)
        self.positive_iou = float(positive_iou)
        self.negative_iou = float(negative_iou)
        self._summary = dict(summary)
        self.quality_weights = dict(quality_weights or {})
        self.logistic_weights = logistic_weights
        self.logistic_bias = logistic_bias
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def summary(self) -> dict[str, Any]:
        return {
            **self._summary,
            "method": self.method,
            "min_score": self.min_score,
            "max_per_image": self.max_per_image,
            "precision_target": self.precision_target,
            "positive_iou": self.positive_iou,
            "negative_iou": self.negative_iou,
            "global_threshold": self.global_threshold,
            "class_thresholds": {str(k): v for k, v in sorted(self.class_thresholds.items())},
        }

    def activation_score(self, row: dict[str, Any], *, threshold: float) -> float:
        if self.method == "reliability_model":
            return self._reliability_score(row)
        return _query_quality_score(
            row,
            threshold=threshold,
            min_score=self.min_score,
            weights=self.quality_weights,
        )

    def select(
        self,
        query_rows: list[dict[str, Any]],
        *,
        thresholds: list[float],
        dedup_iou_thresh: float,
        sample: dict[str, Any] | None = None,
        existing_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], LatentActivationSelectionStats]:
        candidates: list[dict[str, Any]] = []
        candidate_count = 0
        for row in query_rows:
            class_id = int(row.get("category_id", -1))
            if class_id < 0 or class_id >= self.num_classes:
                continue
            score = float(row.get("score", 0.0))
            threshold = float(thresholds[class_id])
            if score < self.min_score or score >= threshold:
                continue
            candidate_count += 1
            scored_row = dict(row)
            if sample is not None:
                area_frac, aspect_log, center_x, center_y = _box_features(scored_row, sample)
                scored_row["_area_frac"] = area_frac
                scored_row["_aspect_log"] = aspect_log
                scored_row["_center_x"] = center_x
                scored_row["_center_y"] = center_y
            activation_score = self.activation_score(scored_row, threshold=threshold)
            min_activation = self.class_thresholds.get(class_id, self.global_threshold)
            if min_activation is None or activation_score < float(min_activation):
                continue
            activated = dict(scored_row)
            activated["_latent_query_activation"] = True
            activated["_latent_activation_method"] = self.method
            activated["_latent_activation_score"] = float(activation_score)
            candidates.append(activated)

        selected = _deduplicate_rows(
            candidates,
            iou_thresh=float(dedup_iou_thresh),
            existing_rows=existing_rows,
        )
        if self.max_per_image > 0:
            selected = selected[: self.max_per_image]
        stats = LatentActivationSelectionStats()
        stats.update(
            candidates=candidate_count,
            activated_scores=[float(row["_latent_activation_score"]) for row in selected],
        )
        return selected, stats

    def _feature_vector(self, row: dict[str, Any]) -> np.ndarray:
        values = [
            float(row.get("score", 0.0)),
            float(row.get("softmax_margin", 0.0)),
            1.0 - float(row.get("softmax_entropy", 1.0)),
            1.0 - float(row.get("decoder_box_iou_gap", 1.0)),
            1.0 - float(row.get("decoder_center_shift", 1.0)),
            float(row.get("_area_frac", 0.0)),
            float(row.get("_aspect_log", 0.0)),
            float(row.get("_center_x", 0.5)),
            float(row.get("_center_y", 0.5)),
        ]
        class_id = int(row.get("category_id", -1))
        values.extend(1.0 if idx == class_id else 0.0 for idx in range(self.num_classes))
        return np.asarray(values, dtype=np.float32)

    def _reliability_score(self, row: dict[str, Any]) -> float:
        if (
            self.logistic_weights is None
            or self.logistic_bias is None
            or self.feature_mean is None
            or self.feature_std is None
        ):
            return 0.0
        x = self._feature_vector(row)
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        std = np.asarray(self.feature_std, dtype=np.float32)
        x = (x - mean) / np.maximum(std, 1e-6)
        w = np.asarray(self.logistic_weights, dtype=np.float32)
        return _sigmoid(float(np.dot(x, w) + float(self.logistic_bias)))


def fit_latent_query_activator(
    teacher_items: list[dict[str, Any]],
    *,
    thresholds: list[float],
    num_classes: int,
    activation_cfg: Any,
    seed: int,
) -> LatentQueryActivator:
    method = str(_cfg_get(activation_cfg, "method", "precision_rule")).strip().lower()
    if method not in {"precision_rule", "reliability_model"}:
        raise ValueError(
            "method.latent_query_activation.method must be precision_rule or reliability_model, "
            f"got {method!r}"
        )
    min_score = float(_cfg_get(activation_cfg, "min_score", 0.02))
    precision_target = float(_cfg_get(activation_cfg, "precision_target", 0.95))
    positive_iou = float(_cfg_get(activation_cfg, "positive_iou", 0.5))
    negative_iou = float(_cfg_get(activation_cfg, "negative_iou", 0.3))
    max_per_image = int(_cfg_get(activation_cfg, "max_per_image", 5))
    min_class_positives = int(_cfg_get(activation_cfg, "min_class_positives", 5))
    min_class_candidates = int(_cfg_get(activation_cfg, "min_class_candidates", 20))
    quality_weights = _quality_weights_from_config(activation_cfg)

    records = _collect_fit_records(
        teacher_items,
        thresholds=thresholds,
        num_classes=num_classes,
        min_score=min_score,
        positive_iou=positive_iou,
        negative_iou=negative_iou,
    )
    summary: dict[str, Any] = {
        "fit_images": len(teacher_items),
        "fit_candidates": len(records),
        "fit_positive": int(sum(1 for record in records if int(record["label"]) == 1)),
        "fit_negative": int(sum(1 for record in records if int(record["label"]) == 0)),
        "min_class_positives": min_class_positives,
        "min_class_candidates": min_class_candidates,
    }

    if method == "precision_rule":
        for record in records:
            class_id = int(record["class_id"])
            record["activation_score"] = _query_quality_score(
                record["row"],
                threshold=float(thresholds[class_id]),
                min_score=min_score,
                weights=quality_weights,
            )
        class_thresholds, global_threshold, threshold_stats = _fit_precision_thresholds(
            records,
            score_key="activation_score",
            precision_target=precision_target,
            min_class_positives=min_class_positives,
            min_class_candidates=min_class_candidates,
        )
        summary.update(threshold_stats)
        return LatentQueryActivator(
            method=method,
            num_classes=num_classes,
            min_score=min_score,
            max_per_image=max_per_image,
            class_thresholds=class_thresholds,
            global_threshold=global_threshold,
            precision_target=precision_target,
            positive_iou=positive_iou,
            negative_iou=negative_iou,
            quality_weights=quality_weights,
            summary=summary,
        )

    logistic = _fit_logistic_reliability(
        records,
        num_classes=num_classes,
        train_steps=int(_cfg_get(activation_cfg, "train_steps", 300)),
        lr=float(_cfg_get(activation_cfg, "lr", 0.05)),
        l2=float(_cfg_get(activation_cfg, "l2", 0.001)),
        seed=seed,
    )
    for record in records:
        record["activation_score"] = _predict_logistic(record["features"], logistic)
    class_thresholds, global_threshold, threshold_stats = _fit_precision_thresholds(
        records,
        score_key="activation_score",
        precision_target=precision_target,
        min_class_positives=min_class_positives,
        min_class_candidates=min_class_candidates,
    )
    summary.update(threshold_stats)
    summary["logistic_fit"] = {
        key: value for key, value in logistic.items() if key not in {"weights", "bias", "mean", "std"}
    }
    return LatentQueryActivator(
        method=method,
        num_classes=num_classes,
        min_score=min_score,
        max_per_image=max_per_image,
        class_thresholds=class_thresholds,
        global_threshold=global_threshold,
        precision_target=precision_target,
        positive_iou=positive_iou,
        negative_iou=negative_iou,
        logistic_weights=logistic["weights"],
        logistic_bias=logistic["bias"],
        feature_mean=logistic["mean"],
        feature_std=logistic["std"],
        summary=summary,
    )


def _quality_weights_from_config(activation_cfg: Any) -> dict[str, float]:
    weights_cfg = _cfg_get(activation_cfg, "quality_weights", {})
    defaults = {
        "score": 0.40,
        "margin": 0.15,
        "confidence": 0.15,
        "box_stability": 0.20,
        "center_stability": 0.10,
    }
    if isinstance(weights_cfg, dict):
        defaults.update({str(key): float(value) for key, value in weights_cfg.items()})
    return defaults


def _collect_fit_records(
    teacher_items: list[dict[str, Any]],
    *,
    thresholds: list[float],
    num_classes: int,
    min_score: float,
    positive_iou: float,
    negative_iou: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in teacher_items:
        sample = item["sample"]
        annotations = sample.get("annotations", [])
        for row in item.get("query_rows", []):
            class_id = int(row.get("category_id", -1))
            if class_id < 0 or class_id >= num_classes:
                continue
            score = float(row.get("score", 0.0))
            if score < float(min_score) or score >= float(thresholds[class_id]):
                continue
            same_class_iou = _best_iou_for_predicted_class(row, annotations)
            if same_class_iou >= float(positive_iou):
                label = 1
            elif same_class_iou <= float(negative_iou):
                label = 0
            else:
                continue
            enriched_row = dict(row)
            area_frac, aspect_log, center_x, center_y = _box_features(enriched_row, sample)
            enriched_row["_area_frac"] = area_frac
            enriched_row["_aspect_log"] = aspect_log
            enriched_row["_center_x"] = center_x
            enriched_row["_center_y"] = center_y
            records.append(
                {
                    "class_id": class_id,
                    "label": int(label),
                    "same_class_iou": float(same_class_iou),
                    "row": enriched_row,
                    "features": _feature_vector(enriched_row, num_classes=num_classes),
                }
            )
    return records


def _feature_vector(row: dict[str, Any], *, num_classes: int) -> np.ndarray:
    values = [
        float(row.get("score", 0.0)),
        float(row.get("softmax_margin", 0.0)),
        1.0 - float(row.get("softmax_entropy", 1.0)),
        1.0 - float(row.get("decoder_box_iou_gap", 1.0)),
        1.0 - float(row.get("decoder_center_shift", 1.0)),
        float(row.get("_area_frac", 0.0)),
        float(row.get("_aspect_log", 0.0)),
        float(row.get("_center_x", 0.5)),
        float(row.get("_center_y", 0.5)),
    ]
    class_id = int(row.get("category_id", -1))
    values.extend(1.0 if idx == class_id else 0.0 for idx in range(int(num_classes)))
    return np.asarray(values, dtype=np.float32)


def _fit_precision_thresholds(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    precision_target: float,
    min_class_positives: int,
    min_class_candidates: int,
) -> tuple[dict[int, float], float | None, dict[str, Any]]:
    global_threshold, global_stats = _best_threshold_for_records(
        records,
        score_key=score_key,
        precision_target=precision_target,
    )
    class_thresholds: dict[int, float] = {}
    class_stats: dict[str, Any] = {}
    class_ids = sorted({int(record["class_id"]) for record in records})
    for class_id in class_ids:
        class_records = [record for record in records if int(record["class_id"]) == class_id]
        positives = sum(1 for record in class_records if int(record["label"]) == 1)
        if positives < int(min_class_positives) or len(class_records) < int(min_class_candidates):
            class_stats[str(class_id)] = {
                "threshold": None,
                "num_records": len(class_records),
                "positives": int(positives),
                "fallback": "global",
            }
            continue
        threshold, stats = _best_threshold_for_records(
            class_records,
            score_key=score_key,
            precision_target=precision_target,
        )
        if threshold is not None:
            class_thresholds[class_id] = float(threshold)
        class_stats[str(class_id)] = stats
    return class_thresholds, global_threshold, {
        "global_threshold_stats": global_stats,
        "class_threshold_stats": class_stats,
    }


def _best_threshold_for_records(
    records: list[dict[str, Any]],
    *,
    score_key: str,
    precision_target: float,
) -> tuple[float | None, dict[str, Any]]:
    usable = [
        record
        for record in records
        if isinstance(record.get(score_key), (int, float)) and math.isfinite(float(record[score_key]))
    ]
    usable.sort(key=lambda record: float(record[score_key]), reverse=True)
    best: dict[str, Any] | None = None
    positives_so_far = 0
    for idx, record in enumerate(usable, start=1):
        positives_so_far += int(record["label"]) == 1
        precision = float(positives_so_far / idx)
        if precision < float(precision_target):
            continue
        if best is None or positives_so_far > best["positives"] or (
            positives_so_far == best["positives"] and idx > best["selected"]
        ):
            best = {
                "threshold": float(record[score_key]),
                "selected": int(idx),
                "positives": int(positives_so_far),
                "precision": precision,
            }
    total_positive = int(sum(1 for record in usable if int(record["label"]) == 1))
    stats = {
        "threshold": None if best is None else best["threshold"],
        "selected": 0 if best is None else best["selected"],
        "positives": 0 if best is None else best["positives"],
        "precision": None if best is None else best["precision"],
        "num_records": len(usable),
        "total_positive": total_positive,
    }
    return (None if best is None else float(best["threshold"])), stats


def _fit_logistic_reliability(
    records: list[dict[str, Any]],
    *,
    num_classes: int,
    train_steps: int,
    lr: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    if not records:
        feature_dim = 9 + int(num_classes)
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
            "mean": mean.tolist(),
            "std": std.tolist(),
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
    pos_weight = torch.tensor([max(float(negatives) / max(float(positives), 1.0), 1.0)], dtype=torch.float32)
    for _ in range(max(int(train_steps), 1)):
        logits = linear(x_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        if l2 > 0.0:
            loss = loss + float(l2) * linear.weight.pow(2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = linear(x_t)
        pred = torch.sigmoid(logits)
        train_loss = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), y_t)
    return {
        "weights": linear.weight.detach().cpu().numpy().reshape(-1).astype(float).tolist(),
        "bias": float(linear.bias.detach().cpu().item()),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "trained": True,
        "positives": positives,
        "negatives": negatives,
        "train_loss": float(train_loss.detach().cpu().item()),
    }


def _predict_logistic(features: np.ndarray, logistic: dict[str, Any]) -> float:
    mean = np.asarray(logistic["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(logistic["std"], dtype=np.float32), 1e-6)
    weights = np.asarray(logistic["weights"], dtype=np.float32)
    x = (features.astype(np.float32) - mean) / std
    return _sigmoid(float(np.dot(x, weights) + float(logistic["bias"])))
