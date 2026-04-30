"""Reusable score calibration for pseudo labels on labeled target images.

This module learns a low-capacity mapping from raw pseudo-label confidence to
target-domain correctness using the currently labeled target subset.

Design goals:
- reuse across different SFOD foundations (our trainer, DDT later)
- operate only on pseudo rows with scores, not on method-specific losses
- stay conservative via a holdout check that can fall back to identity
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.engine.daod_round_trainer import _teacher_outputs_for_unlabeled, _xyxy_iou


@dataclass
class PseudoScoreCalibrator:
    method: str
    global_slope: float
    global_bias: float
    class_biases: list[float]
    min_output_score: float
    max_output_score: float
    raw_score_eps: float
    num_classes: int
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identity_calibrator(
    num_classes: int,
    *,
    method: str = "identity",
    min_output_score: float = 0.0,
    max_output_score: float = 1.0,
    raw_score_eps: float = 1e-4,
    fallback_reason: str | None = None,
) -> PseudoScoreCalibrator:
    return PseudoScoreCalibrator(
        method=method,
        global_slope=1.0,
        global_bias=0.0,
        class_biases=[0.0 for _ in range(num_classes)],
        min_output_score=float(min_output_score),
        max_output_score=float(max_output_score),
        raw_score_eps=float(raw_score_eps),
        num_classes=int(num_classes),
        fallback_reason=fallback_reason,
    )


def _clip_prob(value: float | np.ndarray, *, eps: float) -> float | np.ndarray:
    return np.clip(value, eps, 1.0 - eps)


def _score_logit(score: float, *, eps: float) -> float:
    score_clipped = float(_clip_prob(float(score), eps=eps))
    return float(np.log(score_clipped / (1.0 - score_clipped)))


def _calibrated_prob(
    raw_score: float,
    *,
    class_id: int,
    calibrator: PseudoScoreCalibrator,
) -> float:
    raw_logit = _score_logit(raw_score, eps=calibrator.raw_score_eps)
    class_bias = 0.0
    if 0 <= int(class_id) < len(calibrator.class_biases):
        class_bias = float(calibrator.class_biases[int(class_id)])
    calibrated_logit = (
        float(calibrator.global_slope) * raw_logit
        + float(calibrator.global_bias)
        + class_bias
    )
    calibrated_score = 1.0 / (1.0 + float(np.exp(-calibrated_logit)))
    return float(
        np.clip(
            calibrated_score,
            float(calibrator.min_output_score),
            float(calibrator.max_output_score),
        )
    )


def apply_pseudo_score_calibrator_to_rows(
    rows: list[dict[str, Any]],
    calibrator: PseudoScoreCalibrator | None,
    *,
    replace_score: bool = True,
) -> list[dict[str, Any]]:
    if calibrator is None:
        return list(rows)
    calibrated_rows: list[dict[str, Any]] = []
    for row in rows:
        class_id = int(row.get("category_id", -1))
        raw_score = float(row.get("raw_score", row["score"]))
        calibrated_score = _calibrated_prob(raw_score, class_id=class_id, calibrator=calibrator)
        updated = dict(row)
        updated["raw_score"] = raw_score
        if replace_score:
            updated["score"] = calibrated_score
        updated["calibrated_score"] = calibrated_score
        updated["score_calibration_method"] = str(calibrator.method)
        calibrated_rows.append(updated)
    return calibrated_rows


def apply_pseudo_score_calibrator_to_items(
    teacher_items: list[dict[str, Any]],
    calibrator: PseudoScoreCalibrator | None,
    *,
    replace_score: bool = True,
) -> list[dict[str, Any]]:
    if calibrator is None:
        return list(teacher_items)
    calibrated_items: list[dict[str, Any]] = []
    for item in teacher_items:
        updated = dict(item)
        updated["query_rows"] = apply_pseudo_score_calibrator_to_rows(
            item.get("query_rows", []),
            calibrator,
            replace_score=replace_score,
        )
        calibrated_items.append(updated)
    return calibrated_items


def apply_pseudo_score_calibrator_to_thresholds(
    thresholds: list[float],
    calibrator: PseudoScoreCalibrator | None,
) -> list[float]:
    if calibrator is None:
        return [float(value) for value in thresholds]
    return [
        _calibrated_prob(float(threshold), class_id=class_id, calibrator=calibrator)
        for class_id, threshold in enumerate(thresholds)
    ]


def pseudo_reliability_weight_for_rows(
    rows: list[dict[str, Any]],
    cfg: Any,
    *,
    thresholds: list[float] | dict[int, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    if not rows:
        return 1.0, {"num_rows": 0, "weight": 1.0}

    score_key = str(getattr(cfg, "score_key", "calibrated_score"))
    fallback_score_key = str(getattr(cfg, "fallback_score_key", "score"))
    min_weight = float(getattr(cfg, "min_weight", 0.25))
    max_weight = float(getattr(cfg, "max_weight", 1.0))
    min_weight = float(np.clip(min_weight, 0.0, 1.0))
    max_weight = float(np.clip(max_weight, min_weight, 1.0))
    power = max(1e-6, float(getattr(cfg, "power", 1.0)))
    aggregation = str(getattr(cfg, "aggregation", "mean")).strip().lower()
    relative_to_threshold = bool(getattr(cfg, "relative_to_threshold", False))

    weights: list[float] = []
    scores: list[float] = []
    for row in rows:
        score = float(row.get(score_key, row.get(fallback_score_key, row.get("score", 1.0))))
        score = float(np.clip(score, 0.0, 1.0))
        scores.append(score)
        if relative_to_threshold and thresholds is not None:
            class_id = int(row.get("category_id", -1))
            if isinstance(thresholds, dict):
                threshold = float(thresholds.get(class_id, 0.0))
            else:
                threshold = float(thresholds[class_id]) if 0 <= class_id < len(thresholds) else 0.0
            if threshold < 1.0:
                score = float(np.clip((score - threshold) / max(1.0 - threshold, 1e-12), 0.0, 1.0))
        reliability = float(np.clip(score, 0.0, 1.0)) ** power
        weights.append(float(min_weight + (max_weight - min_weight) * reliability))

    if aggregation == "min":
        weight = min(weights)
    elif aggregation == "max":
        weight = max(weights)
    else:
        weight = float(np.mean(weights))
        aggregation = "mean"

    return float(np.clip(weight, min_weight, max_weight)), {
        "num_rows": len(rows),
        "weight": float(np.clip(weight, min_weight, max_weight)),
        "score_key": score_key,
        "aggregation": aggregation,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "mean_row_weight": float(np.mean(weights)) if weights else 1.0,
        "min_row_weight": float(min(weights)) if weights else 1.0,
        "max_row_weight": float(max(weights)) if weights else 1.0,
    }


def pseudo_reliability_weight_for_samples(samples: list[dict[str, Any]]) -> float:
    weights = []
    row_counts = []
    for sample in samples:
        weights.append(float(sample.get("_pseudo_reliability_weight", 1.0)))
        row_counts.append(max(1.0, float(sample.get("_pseudo_reliability_num_rows", 1.0))))
    if not weights:
        return 1.0
    return float(np.clip(float(np.average(weights, weights=row_counts)), 0.0, 1.0))


def _labeled_items(
    target_train_dicts: list[dict[str, Any]],
    labeled_ids: set[str],
    *,
    max_images: int,
    seed: int,
) -> list[dict[str, Any]]:
    items = [sample for sample in target_train_dicts if str(sample["sample_id"]) in labeled_ids]
    if max_images <= 0 or max_images >= len(items):
        return items
    rng = np.random.default_rng(seed)
    keep = sorted(int(idx) for idx in rng.choice(len(items), size=max_images, replace=False))
    return [items[idx] for idx in keep]


def _teacher_items(
    teacher_adapter,
    items: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if teacher_adapter is None or not items:
        return []
    rng = np.random.default_rng(seed)
    return _teacher_outputs_for_unlabeled(teacher_adapter, items, weak_view_rng=rng)


def _gt_boxes_by_class(sample: dict[str, Any], num_classes: int) -> list[list[list[float]]]:
    boxes = [[] for _ in range(num_classes)]
    for ann in sample.get("annotations", []):
        category_id = int(ann.get("category_id", -1))
        bbox = ann.get("bbox", [])
        if 0 <= category_id < num_classes and len(bbox) == 4:
            boxes[category_id].append([float(v) for v in bbox])
    return boxes


def _examples_from_teacher_items(
    teacher_items: list[dict[str, Any]],
    *,
    num_classes: int,
    candidate_score_floor: float,
    iou_thresh: float,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for teacher_item in teacher_items:
        sample = teacher_item["sample"]
        sample_id = str(sample["sample_id"])
        gt_by_class = _gt_boxes_by_class(sample, num_classes)
        rows_by_class: list[list[dict[str, Any]]] = [[] for _ in range(num_classes)]
        for row in teacher_item.get("query_rows", []):
            class_id = int(row.get("category_id", -1))
            score = float(row.get("score", 0.0))
            if 0 <= class_id < num_classes and score >= candidate_score_floor:
                rows_by_class[class_id].append(row)

        for class_id, rows in enumerate(rows_by_class):
            sorted_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
            matched_gt: set[int] = set()
            gt_boxes = gt_by_class[class_id]
            for row in sorted_rows:
                best_gt_idx = -1
                best_iou = 0.0
                pred_box = [float(v) for v in row["bbox"]]
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = _xyxy_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                is_tp = best_gt_idx >= 0 and best_iou >= iou_thresh
                if is_tp:
                    matched_gt.add(best_gt_idx)
                examples.append(
                    {
                        "sample_id": sample_id,
                        "category_id": int(class_id),
                        "score": float(row["score"]),
                        "label": int(is_tp),
                        "best_iou": float(best_iou),
                    }
                )
    return examples


def _split_examples_by_sample(
    examples: list[dict[str, Any]],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return [], []
    holdout_ratio = float(np.clip(float(holdout_ratio), 0.0, 0.5))
    sample_ids = sorted({str(example["sample_id"]) for example in examples})
    if holdout_ratio <= 0.0 or len(sample_ids) < 2:
        return examples, []
    holdout_count = int(round(len(sample_ids) * holdout_ratio))
    holdout_count = min(max(1, holdout_count), len(sample_ids) - 1)
    rng = np.random.default_rng(seed)
    holdout_ids = {sample_ids[int(idx)] for idx in rng.choice(len(sample_ids), size=holdout_count, replace=False)}
    train_examples = [example for example in examples if str(example["sample_id"]) not in holdout_ids]
    val_examples = [example for example in examples if str(example["sample_id"]) in holdout_ids]
    if not train_examples or not val_examples:
        return examples, []
    return train_examples, val_examples


def _probs_for_examples(
    examples: list[dict[str, Any]],
    calibrator: PseudoScoreCalibrator,
) -> np.ndarray:
    return np.asarray(
        [
            _calibrated_prob(
                float(example["score"]),
                class_id=int(example["category_id"]),
                calibrator=calibrator,
            )
            for example in examples
        ],
        dtype=float,
    )


def _raw_probs_for_examples(examples: list[dict[str, Any]], *, eps: float) -> np.ndarray:
    if not examples:
        return np.asarray([], dtype=float)
    return np.asarray([float(_clip_prob(float(example["score"]), eps=eps)) for example in examples], dtype=float)


def _metrics_for_probs(
    examples: list[dict[str, Any]],
    probs: np.ndarray,
    *,
    eps: float,
) -> dict[str, float]:
    if not examples:
        return {
            "num_examples": 0.0,
            "positive_rate": 0.0,
            "mean_score": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "accuracy_at_050": 0.0,
        }
    labels = np.asarray([float(example["label"]) for example in examples], dtype=float)
    clipped = np.asarray(_clip_prob(probs, eps=eps), dtype=float)
    return {
        "num_examples": float(len(examples)),
        "positive_rate": float(labels.mean()),
        "mean_score": float(clipped.mean()),
        "brier": float(np.mean((clipped - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))),
        "accuracy_at_050": float(np.mean((clipped >= 0.5) == (labels >= 0.5))),
    }


def _fit_global_platt(
    train_examples: list[dict[str, Any]],
    *,
    cfg: Any,
    raw_score_eps: float,
) -> tuple[float, float]:
    x = torch.as_tensor(
        [_score_logit(float(example["score"]), eps=raw_score_eps) for example in train_examples],
        dtype=torch.float64,
    )
    y = torch.as_tensor([float(example["label"]) for example in train_examples], dtype=torch.float64)
    log_slope = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    slope_reg = max(0.0, float(getattr(cfg, "slope_reg", 1e-2)))
    bias_reg = max(0.0, float(getattr(cfg, "bias_reg", 1e-2)))
    optimizer = torch.optim.LBFGS(
        [log_slope, bias],
        lr=float(getattr(cfg, "lbfgs_lr", 0.5)),
        max_iter=int(getattr(cfg, "lbfgs_max_iter", 50)),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        slope = torch.exp(log_slope)
        logits = slope * x + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = loss + slope_reg * (log_slope**2) + bias_reg * (bias**2)
        loss.backward()
        return loss

    optimizer.step(closure)
    slope_value = float(torch.exp(log_slope).detach().cpu())
    bias_value = float(bias.detach().cpu())
    min_slope = max(1e-3, float(getattr(cfg, "min_slope", 0.25)))
    max_slope = max(min_slope, float(getattr(cfg, "max_slope", 4.0)))
    max_abs_bias = max(0.0, float(getattr(cfg, "max_abs_bias", 2.0)))
    slope_value = float(np.clip(slope_value, min_slope, max_slope))
    bias_value = float(np.clip(bias_value, -max_abs_bias, max_abs_bias))
    return slope_value, bias_value


def _fit_class_biases(
    train_examples: list[dict[str, Any]],
    *,
    num_classes: int,
    cfg: Any,
    raw_score_eps: float,
    global_slope: float,
    global_bias: float,
) -> tuple[list[float], dict[str, Any]]:
    if not bool(getattr(cfg, "use_class_bias", True)):
        return [0.0 for _ in range(num_classes)], {
            "class_example_counts": [0 for _ in range(num_classes)],
            "class_positive_rates": [0.0 for _ in range(num_classes)],
            "class_global_mean_scores": [0.0 for _ in range(num_classes)],
        }

    min_class_examples = max(1, int(getattr(cfg, "min_class_examples", 16)))
    shrinkage = max(0.0, float(getattr(cfg, "class_bias_shrinkage", 32.0)))
    max_abs_class_bias = max(0.0, float(getattr(cfg, "max_abs_class_bias", 1.0)))

    by_class: list[list[dict[str, Any]]] = [[] for _ in range(num_classes)]
    for example in train_examples:
        class_id = int(example["category_id"])
        if 0 <= class_id < num_classes:
            by_class[class_id].append(example)

    biases = [0.0 for _ in range(num_classes)]
    class_example_counts: list[int] = []
    class_positive_rates: list[float] = []
    class_global_mean_scores: list[float] = []

    for class_id, examples_for_class in enumerate(by_class):
        example_count = len(examples_for_class)
        class_example_counts.append(int(example_count))
        if example_count <= 0:
            class_positive_rates.append(0.0)
            class_global_mean_scores.append(0.0)
            continue

        labels = np.asarray([float(example["label"]) for example in examples_for_class], dtype=float)
        global_probs = np.asarray(
            [
                1.0
                / (
                    1.0
                    + np.exp(
                        -(
                            float(global_slope) * _score_logit(float(example["score"]), eps=raw_score_eps)
                            + float(global_bias)
                        )
                    )
                )
                for example in examples_for_class
            ],
            dtype=float,
        )
        label_mean = float(labels.mean())
        global_mean = float(global_probs.mean())
        class_positive_rates.append(label_mean)
        class_global_mean_scores.append(global_mean)

        if example_count < min_class_examples:
            continue

        delta = _score_logit(label_mean, eps=raw_score_eps) - _score_logit(global_mean, eps=raw_score_eps)
        shrink = float(example_count) / float(example_count + shrinkage) if shrinkage > 0.0 else 1.0
        biases[class_id] = float(np.clip(delta * shrink, -max_abs_class_bias, max_abs_class_bias))

    return biases, {
        "class_example_counts": class_example_counts,
        "class_positive_rates": class_positive_rates,
        "class_global_mean_scores": class_global_mean_scores,
    }


def fit_pseudo_score_calibrator_from_examples(
    examples: list[dict[str, Any]],
    *,
    num_classes: int,
    calibration_cfg: Any,
    seed: int = 42,
) -> tuple[PseudoScoreCalibrator, dict[str, Any]]:
    method = str(getattr(calibration_cfg, "method", "platt_class_bias")).strip().lower()
    raw_score_eps = max(1e-6, float(getattr(calibration_cfg, "raw_score_eps", 1e-4)))
    min_output_score = float(getattr(calibration_cfg, "min_output_score", 0.0))
    max_output_score = float(getattr(calibration_cfg, "max_output_score", 1.0))
    min_output_score = float(np.clip(min_output_score, 0.0, 1.0))
    max_output_score = float(np.clip(max_output_score, min_output_score, 1.0))

    if method in {"none", "identity"}:
        calibrator = _identity_calibrator(
            num_classes,
            method="identity",
            min_output_score=min_output_score,
            max_output_score=max_output_score,
            raw_score_eps=raw_score_eps,
        )
        return calibrator, {
            "method": "identity",
            "num_examples": len(examples),
            "fallback_to_identity": False,
            "calibrator": calibrator.to_dict(),
        }

    supported = {"score_calibration", "platt", "platt_scaling", "platt_class_bias"}
    if method not in supported:
        raise ValueError(f"Unsupported pseudo_score_calibration.method: {method}")

    holdout_ratio = float(getattr(calibration_cfg, "holdout_ratio", 0.2))
    train_examples, val_examples = _split_examples_by_sample(examples, holdout_ratio=holdout_ratio, seed=seed)
    min_examples = max(1, int(getattr(calibration_cfg, "min_examples", 128)))
    min_positives = max(1, int(getattr(calibration_cfg, "min_positives", 8)))
    min_negatives = max(1, int(getattr(calibration_cfg, "min_negatives", 8)))
    train_labels = [int(example["label"]) for example in train_examples]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos

    stats: dict[str, Any] = {
        "method": "platt_class_bias",
        "requested_method": method,
        "num_examples": len(examples),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "num_train_positives": int(num_pos),
        "num_train_negatives": int(num_neg),
        "holdout_ratio": holdout_ratio,
    }
    if len(train_examples) < min_examples or num_pos < min_positives or num_neg < min_negatives:
        fallback_reason = (
            "not_enough_examples"
            f"(train={len(train_examples)}, pos={num_pos}, neg={num_neg}, "
            f"min_examples={min_examples}, min_pos={min_positives}, min_neg={min_negatives})"
        )
        calibrator = _identity_calibrator(
            num_classes,
            method="identity",
            min_output_score=min_output_score,
            max_output_score=max_output_score,
            raw_score_eps=raw_score_eps,
            fallback_reason=fallback_reason,
        )
        stats.update(
            {
                "fallback_to_identity": True,
                "fallback_reason": fallback_reason,
                "calibrator": calibrator.to_dict(),
            }
        )
        return calibrator, stats

    global_slope, global_bias = _fit_global_platt(train_examples, cfg=calibration_cfg, raw_score_eps=raw_score_eps)
    class_biases, class_stats = _fit_class_biases(
        train_examples,
        num_classes=num_classes,
        cfg=calibration_cfg,
        raw_score_eps=raw_score_eps,
        global_slope=global_slope,
        global_bias=global_bias,
    )
    calibrator = PseudoScoreCalibrator(
        method="platt_class_bias",
        global_slope=global_slope,
        global_bias=global_bias,
        class_biases=class_biases,
        min_output_score=min_output_score,
        max_output_score=max_output_score,
        raw_score_eps=raw_score_eps,
        num_classes=num_classes,
    )

    raw_train_metrics = _metrics_for_probs(train_examples, _raw_probs_for_examples(train_examples, eps=raw_score_eps), eps=raw_score_eps)
    calibrated_train_metrics = _metrics_for_probs(train_examples, _probs_for_examples(train_examples, calibrator), eps=raw_score_eps)
    raw_val_metrics = _metrics_for_probs(val_examples, _raw_probs_for_examples(val_examples, eps=raw_score_eps), eps=raw_score_eps)
    calibrated_val_metrics = _metrics_for_probs(val_examples, _probs_for_examples(val_examples, calibrator), eps=raw_score_eps)
    stats.update(
        {
            "global_slope": float(global_slope),
            "global_bias": float(global_bias),
            "class_biases": [float(value) for value in class_biases],
            "train_metrics_raw": raw_train_metrics,
            "train_metrics_calibrated": calibrated_train_metrics,
            "val_metrics_raw": raw_val_metrics,
            "val_metrics_calibrated": calibrated_val_metrics,
            **class_stats,
        }
    )

    fallback_to_identity = bool(getattr(calibration_cfg, "fallback_to_identity_on_worse_val", True))
    combined_metric_margin = float(getattr(calibration_cfg, "combined_metric_margin", 1e-6))
    if fallback_to_identity and val_examples:
        raw_combined = float(raw_val_metrics["brier"]) + float(raw_val_metrics["nll"])
        calibrated_combined = float(calibrated_val_metrics["brier"]) + float(calibrated_val_metrics["nll"])
        if calibrated_combined > raw_combined - combined_metric_margin:
            fallback_reason = (
                "holdout_not_better"
                f"(raw_combined={raw_combined:.6f}, calibrated_combined={calibrated_combined:.6f})"
            )
            calibrator = _identity_calibrator(
                num_classes,
                method="identity",
                min_output_score=min_output_score,
                max_output_score=max_output_score,
                raw_score_eps=raw_score_eps,
                fallback_reason=fallback_reason,
            )
            stats.update(
                {
                    "fallback_to_identity": True,
                    "fallback_reason": fallback_reason,
                    "calibrator": calibrator.to_dict(),
                }
            )
            return calibrator, stats

    stats.update(
        {
            "fallback_to_identity": False,
            "calibrator": calibrator.to_dict(),
        }
    )
    return calibrator, stats


def fit_pseudo_score_calibrator(
    target_train_dicts: list[dict[str, Any]],
    labeled_ids: set[str],
    *,
    num_classes: int,
    calibration_cfg: Any,
    teacher_adapter,
    stage_idx: int = 0,
    seed: int = 42,
) -> tuple[PseudoScoreCalibrator, dict[str, Any]]:
    if not labeled_ids:
        calibrator = _identity_calibrator(
            num_classes,
            method="identity",
            fallback_reason="no_labeled_target_images",
        )
        return calibrator, {
            "method": "identity",
            "selected_images": 0,
            "teacher_items": 0,
            "num_examples": 0,
            "fallback_to_identity": True,
            "fallback_reason": "no_labeled_target_images",
            "calibrator": calibrator.to_dict(),
        }

    calibration_seed = int(seed) + 1009 * (int(stage_idx) + 1)
    max_images = int(getattr(calibration_cfg, "calibration_max_images", 0))
    candidate_score_floor = max(0.0, float(getattr(calibration_cfg, "candidate_score_floor", 0.01)))
    iou_thresh = float(getattr(calibration_cfg, "gt_iou_thresh", 0.5))

    items = _labeled_items(
        target_train_dicts,
        labeled_ids,
        max_images=max_images,
        seed=calibration_seed,
    )
    teacher_items = _teacher_items(teacher_adapter, items, seed=calibration_seed + 17)
    examples = _examples_from_teacher_items(
        teacher_items,
        num_classes=num_classes,
        candidate_score_floor=candidate_score_floor,
        iou_thresh=iou_thresh,
    )
    calibrator, stats = fit_pseudo_score_calibrator_from_examples(
        examples,
        num_classes=num_classes,
        calibration_cfg=calibration_cfg,
        seed=calibration_seed + 29,
    )
    stats.update(
        {
            "selected_images": len(items),
            "teacher_items": len(teacher_items),
            "stage_idx": int(stage_idx),
            "candidate_score_floor": float(candidate_score_floor),
            "gt_iou_thresh": float(iou_thresh),
            "calibration_pool": "labeled",
        }
    )
    return calibrator, stats
