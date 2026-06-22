"""Sparse-label pseudo-label selection helpers.

The selection plugin uses only the randomly selected labeled target images as a
small calibration set. It estimates conservative classwise score-threshold
offsets for teacher pseudo labels, then the baseline trainer can apply those
offsets without changing its core pseudo-labeling logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _selection_cfg(method_cfg: Any) -> Any:
    label_cfg = _cfg_get(method_cfg, "label_guided", object())
    nested = _cfg_get(label_cfg, "score_threshold_calibration", None)
    return nested if nested is not None else label_cfg


def _nested_label_cfg(method_cfg: Any, key: str) -> Any:
    label_cfg = _cfg_get(method_cfg, "label_guided", object())
    nested = _cfg_get(label_cfg, key, None)
    return nested if nested is not None else label_cfg


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


@dataclass(frozen=True)
class _ClassCandidate:
    score: float
    is_correct: bool


def _annotations_for_calibration(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Return visible sparse target annotations, never hidden oracle labels."""

    annotations = sample.get("annotations", [])
    if not isinstance(annotations, list):
        return []
    return [ann for ann in annotations if "bbox" in ann and "category_id" in ann]


def _row_is_correct(row: dict[str, Any], annotations: list[dict[str, Any]], *, match_iou: float) -> bool:
    class_id = int(row.get("category_id", -1))
    if class_id < 0:
        return False
    row_box = [float(v) for v in row.get("bbox", [])]
    if len(row_box) != 4:
        return False
    for ann in annotations:
        if int(ann.get("category_id", -1)) != class_id:
            continue
        ann_box = [float(v) for v in ann.get("bbox", [])]
        if len(ann_box) == 4 and xyxy_iou(row_box, ann_box) >= float(match_iou):
            return True
    return False


def _class_candidates(
    teacher_items: list[dict[str, Any]],
    *,
    num_classes: int,
    match_iou: float,
    min_score: float,
) -> list[list[_ClassCandidate]]:
    candidates: list[list[_ClassCandidate]] = [[] for _ in range(int(num_classes))]
    for teacher_item in teacher_items:
        sample = teacher_item.get("sample", {})
        annotations = _annotations_for_calibration(sample)
        for row in teacher_item.get("query_rows", []):
            class_id = int(row.get("category_id", -1))
            if class_id < 0 or class_id >= int(num_classes):
                continue
            score = float(row.get("score", 0.0))
            if score < float(min_score):
                continue
            candidates[class_id].append(
                _ClassCandidate(
                    score=score,
                    is_correct=_row_is_correct(row, annotations, match_iou=match_iou),
                )
            )
    return candidates


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(numerator) / float(denominator)


def _label_class_counts(teacher_items: list[dict[str, Any]], *, num_classes: int) -> list[int]:
    counts = [0 for _ in range(int(num_classes))]
    for teacher_item in teacher_items:
        sample = teacher_item.get("sample", {})
        for ann in _annotations_for_calibration(sample):
            class_id = int(ann.get("category_id", -1))
            if 0 <= class_id < int(num_classes):
                counts[class_id] += 1
    return counts


def _smoothed_prior(counts: list[int], *, smoothing: float) -> list[float]:
    if not counts:
        return []
    smoothing = max(0.0, float(smoothing))
    total = float(sum(counts)) + smoothing * float(len(counts))
    if total <= 0.0:
        return [1.0 / float(len(counts)) for _ in counts]
    return [(float(count) + smoothing) / total for count in counts]


def _candidate_scores_by_class(
    teacher_items: list[dict[str, Any]],
    *,
    num_classes: int,
    min_score: float,
) -> list[list[float]]:
    scores: list[list[float]] = [[] for _ in range(int(num_classes))]
    for teacher_item in teacher_items:
        for row in teacher_item.get("query_rows", []):
            class_id = int(row.get("category_id", -1))
            score = float(row.get("score", 0.0))
            if 0 <= class_id < int(num_classes) and score >= float(min_score):
                scores[class_id].append(score)
    return scores


def _accepted_counts(scores_by_class: list[list[float]], thresholds: list[float]) -> list[int]:
    counts = []
    for class_id, scores in enumerate(scores_by_class):
        threshold = float(thresholds[class_id]) if class_id < len(thresholds) else 1.0
        counts.append(int(sum(float(score) >= threshold for score in scores)))
    return counts


def _clip_threshold(value: float, *, min_threshold: float, max_threshold: float) -> float:
    return float(min(max(float(value), float(min_threshold)), float(max_threshold)))


def _choose_precision_floor_threshold(
    candidates: list[_ClassCandidate],
    *,
    target_precision: float,
    min_selected: int,
) -> dict[str, Any]:
    positive_total = sum(1 for item in candidates if item.is_correct)
    if positive_total <= 0:
        return {
            "found": False,
            "reason": "no_positive_candidates",
            "positive_total": 0,
        }

    best: dict[str, Any] | None = None
    thresholds = sorted({float(item.score) for item in candidates})
    for threshold in thresholds:
        selected = [item for item in candidates if float(item.score) >= threshold]
        selected_count = len(selected)
        selected_positive = sum(1 for item in selected if item.is_correct)
        precision = _safe_div(float(selected_positive), float(selected_count))
        recall = _safe_div(float(selected_positive), float(positive_total))
        if selected_count < int(min_selected) or precision < float(target_precision):
            continue
        candidate = {
            "found": True,
            "threshold": float(threshold),
            "selected_count": int(selected_count),
            "selected_positive": int(selected_positive),
            "precision": float(precision),
            "recall": float(recall),
            "positive_total": int(positive_total),
        }
        if best is None or (
            candidate["recall"],
            candidate["selected_count"],
            -candidate["threshold"],
        ) > (
            best["recall"],
            best["selected_count"],
            -best["threshold"],
        ):
            best = candidate

    if best is not None:
        return best
    return {
        "found": False,
        "reason": "precision_floor_unmet",
        "positive_total": int(positive_total),
    }


def _clamp_threshold(
    value: float,
    *,
    base: float,
    min_threshold: float,
    max_threshold: float,
    max_delta_down: float,
    max_delta_up: float,
) -> float:
    lower = max(float(min_threshold), float(base) - float(max_delta_down))
    upper = min(float(max_threshold), float(base) + float(max_delta_up))
    if upper < lower:
        upper = lower
    return float(min(max(float(value), lower), upper))


def fit_score_threshold_calibration(
    method_cfg: Any,
    *,
    teacher_items: list[dict[str, Any]],
    base_thresholds: list[float],
    num_classes: int,
) -> dict[str, Any]:
    """Fit classwise threshold offsets from sparse labeled target images."""

    cfg = _selection_cfg(method_cfg)
    match_iou = float(_cfg_get(cfg, "match_iou", 0.5))
    min_score = float(_cfg_get(cfg, "min_score", 0.01))
    target_precision = float(_cfg_get(cfg, "target_precision", 0.75))
    min_selected = max(1, int(_cfg_get(cfg, "min_selected", 2)))
    min_positives = max(0, int(_cfg_get(cfg, "min_positives", 1)))
    min_threshold = float(_cfg_get(cfg, "min_threshold", 0.25))
    max_threshold = float(_cfg_get(cfg, "max_threshold", 0.55))
    max_delta_down = float(_cfg_get(cfg, "max_delta_down", 0.10))
    max_delta_up = float(_cfg_get(cfg, "max_delta_up", 0.15))

    base = [float(value) for value in base_thresholds]
    if len(base) != int(num_classes):
        raise ValueError(
            f"Expected {num_classes} base thresholds for label-guided calibration, got {len(base)}"
        )

    class_candidates = _class_candidates(
        teacher_items,
        num_classes=int(num_classes),
        match_iou=match_iou,
        min_score=min_score,
    )
    calibrated: list[float] = []
    offsets: list[float] = []
    class_stats: list[dict[str, Any]] = []

    for class_id, candidates in enumerate(class_candidates):
        candidate_count = len(candidates)
        positive_count = sum(1 for item in candidates if item.is_correct)
        negative_count = candidate_count - positive_count
        base_value = float(base[class_id])
        chosen = _choose_precision_floor_threshold(
            candidates,
            target_precision=target_precision,
            min_selected=min_selected,
        )
        if positive_count < min_positives:
            chosen = {
                "found": False,
                "reason": "insufficient_positive_candidates",
                "positive_total": int(positive_count),
            }

        if bool(chosen.get("found", False)):
            raw_threshold = float(chosen["threshold"])
            threshold = _clamp_threshold(
                raw_threshold,
                base=base_value,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                max_delta_down=max_delta_down,
                max_delta_up=max_delta_up,
            )
            fallback_reason = ""
        else:
            raw_threshold = base_value
            threshold = base_value
            fallback_reason = str(chosen.get("reason", "not_found"))

        calibrated.append(float(threshold))
        offsets.append(float(threshold - base_value))
        selected_count = sum(1 for item in candidates if float(item.score) >= float(threshold))
        selected_positive = sum(
            1 for item in candidates if float(item.score) >= float(threshold) and item.is_correct
        )
        class_stats.append(
            {
                "class_id": int(class_id),
                "base_threshold": float(base_value),
                "raw_calibrated_threshold": float(raw_threshold),
                "calibrated_threshold": float(threshold),
                "offset": float(threshold - base_value),
                "candidate_count": int(candidate_count),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "selected_count": int(selected_count),
                "selected_positive_count": int(selected_positive),
                "precision": _safe_div(float(selected_positive), float(selected_count)),
                "recall": _safe_div(float(selected_positive), float(positive_count)),
                "fallback_reason": fallback_reason,
            }
        )

    adjusted_classes = [stat["class_id"] for stat in class_stats if abs(float(stat["offset"])) > 1e-12]
    return {
        "enabled": True,
        "method": "score_threshold_calibration",
        "fit_images": int(len(teacher_items)),
        "num_classes": int(num_classes),
        "match_iou": float(match_iou),
        "min_score": float(min_score),
        "target_precision": float(target_precision),
        "min_selected": int(min_selected),
        "min_positives": int(min_positives),
        "min_threshold": float(min_threshold),
        "max_threshold": float(max_threshold),
        "max_delta_down": float(max_delta_down),
        "max_delta_up": float(max_delta_up),
        "base_thresholds": [float(value) for value in base],
        "calibrated_thresholds": [float(value) for value in calibrated],
        "offsets": [float(value) for value in offsets],
        "adjusted_classes": [int(value) for value in adjusted_classes],
        "class_stats": class_stats,
        "aggregate": {
            "candidate_count": int(sum(len(items) for items in class_candidates)),
            "positive_count": int(sum(stat["positive_count"] for stat in class_stats)),
            "adjusted_class_count": int(len(adjusted_classes)),
            "mean_abs_offset": _safe_div(
                sum(abs(float(value)) for value in offsets),
                float(max(len(offsets), 1)),
            ),
        },
    }


def apply_threshold_offsets(
    thresholds: list[float],
    fit_result: dict[str, Any],
) -> list[float]:
    """Apply fitted offsets to a current per-class threshold vector."""

    offsets = [float(value) for value in fit_result.get("offsets", [])]
    if not offsets:
        return [float(value) for value in thresholds]
    if len(offsets) != len(thresholds):
        raise ValueError(
            f"Expected {len(thresholds)} threshold offsets, got {len(offsets)}"
        )

    min_threshold = float(fit_result.get("min_threshold", 0.0))
    max_threshold = float(fit_result.get("max_threshold", 1.0))
    adjusted = []
    for threshold, offset in zip(thresholds, offsets):
        value = float(threshold) + float(offset)
        adjusted.append(float(min(max(value, min_threshold), max_threshold)))
    return adjusted


def fit_label_prior_threshold_mapping(
    method_cfg: Any,
    *,
    teacher_items: list[dict[str, Any]],
    base_thresholds: list[float],
    num_classes: int,
) -> dict[str, Any]:
    """Fit threshold offsets that map pseudo-label class priors toward sparse GT priors."""

    cfg = _nested_label_cfg(method_cfg, "threshold_mapping")
    min_score = float(_cfg_get(cfg, "min_score", 0.01))
    smoothing = max(0.0, float(_cfg_get(cfg, "smoothing", 1.0)))
    ratio_temperature = max(0.0, float(_cfg_get(cfg, "ratio_temperature", 0.75)))
    min_threshold = float(_cfg_get(cfg, "min_threshold", 0.25))
    max_threshold = float(_cfg_get(cfg, "max_threshold", 0.55))
    max_delta_down = max(0.0, float(_cfg_get(cfg, "max_delta_down", 0.10)))
    max_delta_up = max(0.0, float(_cfg_get(cfg, "max_delta_up", 0.10)))

    base = [float(value) for value in base_thresholds]
    if len(base) != int(num_classes):
        raise ValueError(f"Expected {num_classes} base thresholds for threshold mapping, got {len(base)}")

    label_counts = _label_class_counts(teacher_items, num_classes=int(num_classes))
    scores_by_class = _candidate_scores_by_class(
        teacher_items,
        num_classes=int(num_classes),
        min_score=min_score,
    )
    base_pseudo_counts = _accepted_counts(scores_by_class, base)
    label_prior = _smoothed_prior(label_counts, smoothing=smoothing)
    pseudo_prior = _smoothed_prior(base_pseudo_counts, smoothing=smoothing)

    calibrated: list[float] = []
    offsets: list[float] = []
    class_stats: list[dict[str, Any]] = []
    for class_id in range(int(num_classes)):
        base_value = float(base[class_id])
        log_ratio = float(np_log(max(label_prior[class_id], 1e-12) / max(pseudo_prior[class_id], 1e-12)))
        direction = float(np_tanh(ratio_temperature * log_ratio))
        if direction >= 0.0:
            raw_offset = -max_delta_down * direction
        else:
            raw_offset = -max_delta_up * direction
        threshold = _clip_threshold(
            base_value + raw_offset,
            min_threshold=max(float(min_threshold), base_value - max_delta_down),
            max_threshold=min(float(max_threshold), base_value + max_delta_up),
        )
        calibrated.append(float(threshold))
        offsets.append(float(threshold - base_value))
        selected_count = sum(float(score) >= float(threshold) for score in scores_by_class[class_id])
        class_stats.append(
            {
                "class_id": int(class_id),
                "label_count": int(label_counts[class_id]),
                "base_pseudo_count": int(base_pseudo_counts[class_id]),
                "mapped_pseudo_count": int(selected_count),
                "label_prior": float(label_prior[class_id]),
                "base_pseudo_prior": float(pseudo_prior[class_id]),
                "log_prior_ratio": float(log_ratio),
                "direction": float(direction),
                "base_threshold": float(base_value),
                "calibrated_threshold": float(threshold),
                "offset": float(threshold - base_value),
            }
        )

    adjusted_classes = [stat["class_id"] for stat in class_stats if abs(float(stat["offset"])) > 1e-12]
    return {
        "enabled": True,
        "method": "threshold_mapping",
        "fit_images": int(len(teacher_items)),
        "num_classes": int(num_classes),
        "min_score": float(min_score),
        "smoothing": float(smoothing),
        "ratio_temperature": float(ratio_temperature),
        "min_threshold": float(min_threshold),
        "max_threshold": float(max_threshold),
        "max_delta_down": float(max_delta_down),
        "max_delta_up": float(max_delta_up),
        "base_thresholds": [float(value) for value in base],
        "calibrated_thresholds": [float(value) for value in calibrated],
        "offsets": [float(value) for value in offsets],
        "adjusted_classes": [int(value) for value in adjusted_classes],
        "class_counts": [int(value) for value in label_counts],
        "base_pseudo_counts": [int(value) for value in base_pseudo_counts],
        "class_stats": class_stats,
        "aggregate": {
            "label_objects": int(sum(label_counts)),
            "base_pseudo_count": int(sum(base_pseudo_counts)),
            "mapped_pseudo_count": int(sum(stat["mapped_pseudo_count"] for stat in class_stats)),
            "adjusted_class_count": int(len(adjusted_classes)),
            "mean_abs_offset": _safe_div(
                sum(abs(float(value)) for value in offsets),
                float(max(len(offsets), 1)),
            ),
        },
    }


def fit_pseudo_score_reweight(
    method_cfg: Any,
    *,
    teacher_items: list[dict[str, Any]],
    base_thresholds: list[float],
    num_classes: int,
) -> dict[str, Any]:
    """Fit classwise score weights from sparse-label pseudo precision."""

    cfg = _nested_label_cfg(method_cfg, "pseudo_score_reweight")
    match_iou = float(_cfg_get(cfg, "match_iou", 0.5))
    min_score = float(_cfg_get(cfg, "min_score", 0.01))
    target_precision = max(1e-6, float(_cfg_get(cfg, "target_precision", 0.75)))
    min_candidates = max(0, int(_cfg_get(cfg, "min_candidates", 3)))
    min_positives = max(0, int(_cfg_get(cfg, "min_positives", 1)))
    min_weight = float(_cfg_get(cfg, "min_weight", 0.50))
    max_weight = float(_cfg_get(cfg, "max_weight", 1.00))
    power = max(1e-6, float(_cfg_get(cfg, "power", 1.0)))

    base = [float(value) for value in base_thresholds]
    if len(base) != int(num_classes):
        raise ValueError(f"Expected {num_classes} base thresholds for score reweighting, got {len(base)}")

    class_candidates = _class_candidates(
        teacher_items,
        num_classes=int(num_classes),
        match_iou=match_iou,
        min_score=min_score,
    )
    weights: list[float] = []
    class_stats: list[dict[str, Any]] = []
    for class_id, candidates in enumerate(class_candidates):
        accepted = [item for item in candidates if float(item.score) >= float(base[class_id])]
        accepted_count = len(accepted)
        positive_count = sum(1 for item in accepted if item.is_correct)
        if accepted_count < min_candidates or positive_count < min_positives:
            precision = None
            weight = 1.0
            fallback_reason = "insufficient_sparse_evidence"
        else:
            precision = _safe_div(float(positive_count), float(accepted_count))
            reliability = float(min(max(float(precision) / target_precision, 0.0), 1.0)) ** power
            weight = float(min(max(min_weight + (max_weight - min_weight) * reliability, min_weight), max_weight))
            fallback_reason = ""
        weights.append(float(weight))
        class_stats.append(
            {
                "class_id": int(class_id),
                "candidate_count": int(len(candidates)),
                "accepted_count": int(accepted_count),
                "accepted_positive_count": int(positive_count),
                "precision_at_base": None if precision is None else float(precision),
                "weight": float(weight),
                "fallback_reason": fallback_reason,
            }
        )

    adjusted_classes = [stat["class_id"] for stat in class_stats if abs(float(stat["weight"]) - 1.0) > 1e-12]
    return {
        "enabled": True,
        "method": "pseudo_score_reweight",
        "fit_images": int(len(teacher_items)),
        "num_classes": int(num_classes),
        "match_iou": float(match_iou),
        "min_score": float(min_score),
        "target_precision": float(target_precision),
        "min_candidates": int(min_candidates),
        "min_positives": int(min_positives),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "power": float(power),
        "class_weights": [float(value) for value in weights],
        "adjusted_classes": [int(value) for value in adjusted_classes],
        "class_stats": class_stats,
        "aggregate": {
            "candidate_count": int(sum(len(items) for items in class_candidates)),
            "accepted_count": int(sum(stat["accepted_count"] for stat in class_stats)),
            "accepted_positive_count": int(sum(stat["accepted_positive_count"] for stat in class_stats)),
            "adjusted_class_count": int(len(adjusted_classes)),
            "mean_weight": _safe_div(sum(weights), float(max(len(weights), 1))),
        },
    }


def apply_class_score_weights(
    teacher_items: list[dict[str, Any]],
    fit_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply fitted classwise score weights before baseline pseudo filtering."""

    weights = [float(value) for value in fit_result.get("class_weights", [])]
    if not weights:
        return teacher_items
    weighted_items: list[dict[str, Any]] = []
    for teacher_item in teacher_items:
        updated_item = dict(teacher_item)
        weighted_rows: list[dict[str, Any]] = []
        for row in teacher_item.get("query_rows", []):
            class_id = int(row.get("category_id", -1))
            weight = weights[class_id] if 0 <= class_id < len(weights) else 1.0
            weighted_row = dict(row)
            raw_score = float(weighted_row.get("raw_score", weighted_row.get("score", 0.0)))
            weighted_row["raw_score"] = raw_score
            weighted_row["score"] = float(min(max(float(row.get("score", 0.0)) * float(weight), 0.0), 1.0))
            weighted_row["label_guided_score_weight"] = float(weight)
            weighted_rows.append(weighted_row)
        updated_item["query_rows"] = weighted_rows
        weighted_items.append(updated_item)
    return weighted_items


# Avoid importing numpy globally in projects that only use the simple helpers.
def np_log(value: float) -> float:
    import numpy as np

    return float(np.log(value))


def np_tanh(value: float) -> float:
    import numpy as np

    return float(np.tanh(value))
