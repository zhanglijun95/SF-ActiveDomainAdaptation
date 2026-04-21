"""Pseudo-label recalibration utilities for stepwise DAOD.

The trainer calls this module at each label-injection point.  The module returns
per-class hard pseudo-label thresholds plus logging stats.  The existing
count-rarity rule is kept as a baseline, while the newer methods use the
currently selected human labels to rebalance or calibrate the pseudo labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engine.daod_round_trainer import _teacher_outputs_for_unlabeled, _xyxy_iou


def label_class_counts(target_train_dicts: list[dict[str, Any]], labeled_ids: set[str], num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for sample in target_train_dicts:
        if sample["sample_id"] not in labeled_ids:
            continue
        for ann in sample.get("annotations", []):
            category_id = int(ann.get("category_id", -1))
            if 0 <= category_id < num_classes:
                counts[category_id] += 1
    return counts


def _bounds(base_score_min: float, cfg: Any) -> tuple[float, float]:
    min_score_min = float(getattr(cfg, "min_score_min", max(0.0, base_score_min - float(getattr(cfg, "max_delta", 0.10)))))
    min_score_min = min(base_score_min, max(0.0, min_score_min))
    max_score_min = float(getattr(cfg, "max_score_min", base_score_min))
    max_score_min = max(base_score_min, min(1.0, max_score_min))
    return min_score_min, max_score_min


def _clip_threshold(value: float, *, min_score_min: float, max_score_min: float) -> float:
    return float(np.clip(float(value), min_score_min, max_score_min))


def _smoothed_prior(counts: list[int], *, smoothing: float) -> list[float]:
    if not counts:
        return []
    smoothing = max(0.0, float(smoothing))
    total = float(sum(counts)) + smoothing * float(len(counts))
    if total <= 0.0:
        return [1.0 / float(len(counts)) for _ in counts]
    return [(float(count) + smoothing) / total for count in counts]


def _count_rarity_thresholds(
    counts: list[int],
    *,
    base_score_min: float,
    cfg: Any,
) -> dict[int, float]:
    smoothing = max(0.0, float(getattr(cfg, "smoothing", 1.0)))
    max_delta = max(0.0, float(getattr(cfg, "max_delta", 0.10)))
    min_score_min, _ = _bounds(base_score_min, cfg)
    max_count = max(counts) if counts else 0
    if max_count <= 0:
        return {idx: base_score_min for idx in range(len(counts))}

    thresholds: dict[int, float] = {}
    denom = float(max_count) + smoothing
    for idx, count in enumerate(counts):
        rarity = float(max_count - count) / denom
        threshold = base_score_min - max_delta * float(np.clip(rarity, 0.0, 1.0))
        thresholds[idx] = float(np.clip(threshold, min_score_min, base_score_min))
    return thresholds


def _stage_strength_scale(cfg: Any, stage_idx: int) -> float:
    stage_scales = getattr(cfg, "stage_scales", None)
    if stage_scales is not None:
        scales = [float(value) for value in stage_scales]
        if not scales:
            return 1.0
        idx = min(max(int(stage_idx), 0), len(scales) - 1)
        return max(0.0, scales[idx])

    num_stages = max(1, int(getattr(cfg, "num_stages", 2)))
    start_scale = max(0.0, float(getattr(cfg, "stage_start_scale", 0.5)))
    final_scale = max(0.0, float(getattr(cfg, "stage_final_scale", 1.0)))
    if num_stages <= 1:
        return final_scale
    progress = float(min(max(int(stage_idx), 0), num_stages - 1)) / float(num_stages - 1)
    return start_scale + (final_scale - start_scale) * progress


def _rarity_exp_curve(rarity: float, gamma: float) -> float:
    rarity = float(np.clip(rarity, 0.0, 1.0))
    gamma = max(1e-6, float(gamma))
    denom = float(1.0 - np.exp(-gamma))
    if denom <= 1e-12:
        return rarity
    return float((1.0 - np.exp(-gamma * rarity)) / denom)


def _label_rarity_function_thresholds(
    counts: list[int],
    *,
    base_score_min: float,
    cfg: Any,
    method_name: str,
    stage_idx: int,
    use_exp: bool,
    use_stage_scale: bool,
) -> tuple[dict[int, float], dict[str, Any]]:
    smoothing = max(0.0, float(getattr(cfg, "smoothing", 1.0)))
    max_delta = max(0.0, float(getattr(cfg, "max_delta", 0.10)))
    strength_scale = _stage_strength_scale(cfg, stage_idx) if use_stage_scale else 1.0
    effective_max_delta = max_delta * strength_scale
    min_score_min, _ = _bounds(base_score_min, cfg)
    max_count = max(counts) if counts else 0
    if max_count <= 0:
        return {idx: base_score_min for idx in range(len(counts))}, {
            "method": method_name,
            "class_counts": counts,
            "rarity": [0.0 for _ in counts],
            "rarity_strength": [0.0 for _ in counts],
            "stage_strength_scale": strength_scale,
            "effective_max_delta": effective_max_delta,
        }

    thresholds: dict[int, float] = {}
    rarity_values: list[float] = []
    strength_values: list[float] = []
    denom = float(max_count) + smoothing
    exp_gamma = max(1e-6, float(getattr(cfg, "exp_gamma", 2.0)))
    for idx, count in enumerate(counts):
        rarity = float(np.clip(float(max_count - count) / denom, 0.0, 1.0))
        rarity_strength = _rarity_exp_curve(rarity, exp_gamma) if use_exp else rarity
        threshold = base_score_min - effective_max_delta * rarity_strength
        thresholds[idx] = float(np.clip(threshold, min_score_min, base_score_min))
        rarity_values.append(rarity)
        strength_values.append(rarity_strength)
    return thresholds, {
        "method": method_name,
        "class_counts": counts,
        "rarity": rarity_values,
        "rarity_strength": strength_values,
        "stage_strength_scale": strength_scale,
        "effective_max_delta": effective_max_delta,
        "exp_gamma": exp_gamma if use_exp else None,
    }


def _calibration_items(
    target_train_dicts: list[dict[str, Any]],
    labeled_ids: set[str],
    *,
    pool: str,
    max_images: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool = str(pool).strip().lower()
    if pool == "labeled":
        items = [sample for sample in target_train_dicts if sample["sample_id"] in labeled_ids]
    elif pool == "all":
        items = list(target_train_dicts)
    else:
        items = [sample for sample in target_train_dicts if sample["sample_id"] not in labeled_ids]

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


def _scores_by_class(
    teacher_items: list[dict[str, Any]],
    *,
    num_classes: int,
    candidate_score_floor: float,
) -> list[list[float]]:
    scores = [[] for _ in range(num_classes)]
    for teacher_item in teacher_items:
        for row in teacher_item["query_rows"]:
            category_id = int(row["category_id"])
            score = float(row["score"])
            if 0 <= category_id < num_classes and score >= candidate_score_floor:
                scores[category_id].append(score)
    return scores


def _base_counts(scores_by_class: list[list[float]], *, base_score_min: float) -> list[int]:
    return [int(sum(float(score) >= base_score_min for score in scores)) for scores in scores_by_class]


def _thresholds_from_target_counts(
    scores_by_class: list[list[float]],
    target_counts: list[int],
    *,
    base_score_min: float,
    min_score_min: float,
    max_score_min: float,
) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for category_id, scores in enumerate(scores_by_class):
        sorted_scores = sorted((float(score) for score in scores), reverse=True)
        target_count = int(round(float(target_counts[category_id]))) if category_id < len(target_counts) else 0
        if not sorted_scores:
            thresholds[category_id] = base_score_min
        elif target_count <= 0:
            thresholds[category_id] = max_score_min
        elif target_count >= len(sorted_scores):
            thresholds[category_id] = min(sorted_scores[-1], base_score_min)
        else:
            thresholds[category_id] = sorted_scores[target_count - 1]
        thresholds[category_id] = _clip_threshold(
            thresholds[category_id],
            min_score_min=min_score_min,
            max_score_min=max_score_min,
        )
    return thresholds


def _gt_boxes_by_class(sample: dict[str, Any], num_classes: int) -> list[list[list[float]]]:
    boxes = [[] for _ in range(num_classes)]
    for ann in sample.get("annotations", []):
        category_id = int(ann.get("category_id", -1))
        bbox = ann.get("bbox", [])
        if 0 <= category_id < num_classes and len(bbox) == 4:
            boxes[category_id].append([float(v) for v in bbox])
    return boxes


def _class_eval_curves(
    teacher_items: list[dict[str, Any]],
    *,
    num_classes: int,
    candidate_score_floor: float,
    iou_thresh: float,
) -> tuple[list[int], list[int], list[list[tuple[float, int, int]]]]:
    gt_by_sample: dict[str, list[list[list[float]]]] = {}
    gt_counts = [0 for _ in range(num_classes)]
    rows_by_class: list[list[tuple[str, float, list[float]]]] = [[] for _ in range(num_classes)]

    for teacher_item in teacher_items:
        sample_id = str(teacher_item["sample"]["sample_id"])
        gt_by_class = _gt_boxes_by_class(teacher_item["sample"], num_classes)
        gt_by_sample[sample_id] = gt_by_class
        for category_id, boxes in enumerate(gt_by_class):
            gt_counts[category_id] += len(boxes)
        for row in teacher_item["query_rows"]:
            category_id = int(row["category_id"])
            score = float(row["score"])
            if 0 <= category_id < num_classes and score >= candidate_score_floor:
                rows_by_class[category_id].append((sample_id, score, [float(v) for v in row["bbox"]]))

    curves: list[list[tuple[float, int, int]]] = []
    for category_id in range(num_classes):
        rows = sorted(rows_by_class[category_id], key=lambda item: item[1], reverse=True)
        matched_gt: set[tuple[str, int]] = set()
        cumulative_tp = 0
        cumulative_fp = 0
        grouped: list[tuple[float, int, int]] = []
        current_score: float | None = None
        group_tp = 0
        group_fp = 0

        for sample_id, score, pred_box in rows:
            best_gt_idx = -1
            best_iou = 0.0
            for gt_idx, gt_box in enumerate(gt_by_sample.get(sample_id, [[] for _ in range(num_classes)])[category_id]):
                key = (sample_id, gt_idx)
                if key in matched_gt:
                    continue
                iou = _xyxy_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            is_tp = best_gt_idx >= 0 and best_iou >= iou_thresh
            if is_tp:
                matched_gt.add((sample_id, best_gt_idx))

            if current_score is None:
                current_score = score
            if score != current_score:
                cumulative_tp += group_tp
                cumulative_fp += group_fp
                grouped.append((current_score, cumulative_tp, cumulative_fp))
                current_score = score
                group_tp = 0
                group_fp = 0
            if is_tp:
                group_tp += 1
            else:
                group_fp += 1

        if current_score is not None:
            cumulative_tp += group_tp
            cumulative_fp += group_fp
            grouped.append((current_score, cumulative_tp, cumulative_fp))
        curves.append(grouped)
    return gt_counts, [len(rows) for rows in rows_by_class], curves


def _metrics_at_base(
    gt_counts: list[int],
    curves: list[list[tuple[float, int, int]]],
    *,
    base_score_min: float,
) -> tuple[list[float], list[float], list[int]]:
    precisions: list[float] = []
    recalls: list[float] = []
    pred_counts: list[int] = []
    for total_gt, curve in zip(gt_counts, curves):
        tp = 0
        fp = 0
        for threshold, curve_tp, curve_fp in curve:
            if threshold >= base_score_min:
                tp, fp = curve_tp, curve_fp
            else:
                break
        pred_count = tp + fp
        precision = 1.0 if pred_count <= 0 else float(tp) / float(pred_count)
        recall = 1.0 if total_gt <= 0 else float(tp) / float(total_gt)
        precisions.append(float(np.clip(precision, 0.0, 1.0)))
        recalls.append(float(np.clip(recall, 0.0, 1.0)))
        pred_counts.append(int(pred_count))
    return precisions, recalls, pred_counts


def _label_prior_ratio(
    counts: list[int],
    scores_by_class: list[list[float]],
    *,
    base_score_min: float,
    cfg: Any,
) -> tuple[dict[int, float], dict[str, Any]]:
    min_score_min, max_score_min = _bounds(base_score_min, cfg)
    smoothing = max(0.0, float(getattr(cfg, "smoothing", 1.0)))
    max_delta = max(0.0, float(getattr(cfg, "max_delta", 0.12)))
    temperature = max(0.0, float(getattr(cfg, "ratio_temperature", 0.75)))

    label_prior = _smoothed_prior(counts, smoothing=smoothing)
    accepted_counts = _base_counts(scores_by_class, base_score_min=base_score_min)
    pseudo_prior = _smoothed_prior(accepted_counts, smoothing=smoothing)
    thresholds: dict[int, float] = {}
    shifts: list[float] = []
    for category_id, (label_p, pseudo_p) in enumerate(zip(label_prior, pseudo_prior)):
        log_ratio = float(np.log(max(label_p, 1e-12) / max(pseudo_p, 1e-12)))
        shift = max_delta * float(np.tanh(temperature * log_ratio))
        threshold = base_score_min - shift
        thresholds[category_id] = _clip_threshold(threshold, min_score_min=min_score_min, max_score_min=max_score_min)
        shifts.append(shift)
    return thresholds, {
        "method": "label_prior_ratio",
        "class_counts": counts,
        "label_prior": label_prior,
        "base_pseudo_counts": accepted_counts,
        "base_pseudo_prior": pseudo_prior,
        "threshold_shifts": shifts,
    }


def _label_prior_quota(
    counts: list[int],
    scores_by_class: list[list[float]],
    *,
    base_score_min: float,
    cfg: Any,
) -> tuple[dict[int, float], dict[str, Any]]:
    min_score_min, max_score_min = _bounds(base_score_min, cfg)
    smoothing = max(0.0, float(getattr(cfg, "smoothing", 1.0)))
    total_scale = max(0.0, float(getattr(cfg, "target_total_scale", 1.0)))
    min_target_per_seen_class = int(getattr(cfg, "min_target_per_seen_class", 1))

    label_prior = _smoothed_prior(counts, smoothing=smoothing)
    accepted_counts = _base_counts(scores_by_class, base_score_min=base_score_min)
    target_total = max(1, int(round(float(sum(accepted_counts)) * total_scale)))
    target_counts = [int(round(target_total * prior)) for prior in label_prior]
    for idx, count in enumerate(counts):
        if count > 0:
            target_counts[idx] = max(target_counts[idx], min_target_per_seen_class)

    thresholds = _thresholds_from_target_counts(
        scores_by_class,
        target_counts,
        base_score_min=base_score_min,
        min_score_min=min_score_min,
        max_score_min=max_score_min,
    )
    return thresholds, {
        "method": "label_prior_quota",
        "class_counts": counts,
        "label_prior": label_prior,
        "base_pseudo_counts": accepted_counts,
        "target_pseudo_counts": target_counts,
    }


def _coverage_precision(
    counts: list[int],
    teacher_items: list[dict[str, Any]],
    *,
    num_classes: int,
    base_score_min: float,
    cfg: Any,
) -> tuple[dict[int, float], dict[str, Any]]:
    min_score_min, max_score_min = _bounds(base_score_min, cfg)
    candidate_score_floor = max(0.0, float(getattr(cfg, "candidate_score_floor", 0.01)))
    iou_thresh = float(getattr(cfg, "gt_iou_thresh", 0.50))
    lower_delta = max(0.0, float(getattr(cfg, "lower_delta", getattr(cfg, "max_delta", 0.12))))
    raise_delta = max(0.0, float(getattr(cfg, "raise_delta", 0.12)))
    precision_target = float(getattr(cfg, "precision_target", 0.70))

    gt_counts, candidate_counts, curves = _class_eval_curves(
        teacher_items,
        num_classes=num_classes,
        candidate_score_floor=candidate_score_floor,
        iou_thresh=iou_thresh,
    )
    precisions, recalls, pred_counts = _metrics_at_base(gt_counts, curves, base_score_min=base_score_min)
    max_count = max(counts) if counts else 0

    thresholds: dict[int, float] = {}
    shifts: list[float] = []
    for category_id in range(num_classes):
        if gt_counts[category_id] <= 0:
            thresholds[category_id] = base_score_min
            shifts.append(0.0)
            continue
        rarity = 0.0 if max_count <= 0 else float(max_count - counts[category_id]) / float(max_count + 1.0)
        coverage_gap = 1.0 - recalls[category_id]
        precision_gap = max(0.0, precision_target - precisions[category_id])
        lower_shift = lower_delta * coverage_gap * (0.5 + 0.5 * rarity)
        raise_shift = raise_delta * precision_gap if pred_counts[category_id] > 0 else 0.0
        shift = lower_shift - raise_shift
        threshold = base_score_min - shift
        thresholds[category_id] = _clip_threshold(threshold, min_score_min=min_score_min, max_score_min=max_score_min)
        shifts.append(shift)

    return thresholds, {
        "method": "coverage_precision",
        "class_counts": counts,
        "gt_counts": gt_counts,
        "candidate_counts": candidate_counts,
        "base_pred_counts": pred_counts,
        "base_precisions": precisions,
        "base_recalls": recalls,
        "threshold_shifts": shifts,
    }


def _selected_fbeta(
    counts: list[int],
    teacher_items: list[dict[str, Any]],
    *,
    num_classes: int,
    base_score_min: float,
    cfg: Any,
) -> tuple[dict[int, float], dict[str, Any]]:
    min_score_min, max_score_min = _bounds(base_score_min, cfg)
    candidate_score_floor = max(0.0, float(getattr(cfg, "candidate_score_floor", 0.01)))
    iou_thresh = float(getattr(cfg, "gt_iou_thresh", 0.50))
    base_beta = max(0.05, float(getattr(cfg, "beta", 1.5)))
    rarity_beta_gain = max(0.0, float(getattr(cfg, "rarity_beta_gain", 0.75)))
    precision_floor = float(getattr(cfg, "precision_floor", 0.50))
    precision_penalty = max(0.0, float(getattr(cfg, "precision_penalty", 0.15)))

    gt_counts, candidate_counts, curves = _class_eval_curves(
        teacher_items,
        num_classes=num_classes,
        candidate_score_floor=candidate_score_floor,
        iou_thresh=iou_thresh,
    )
    max_count = max(counts) if counts else 0
    thresholds: dict[int, float] = {}
    best_precisions: list[float] = []
    best_recalls: list[float] = []
    best_utilities: list[float] = []

    for category_id, curve in enumerate(curves):
        total_gt = gt_counts[category_id]
        if total_gt <= 0 or not curve:
            thresholds[category_id] = base_score_min
            best_precisions.append(0.0)
            best_recalls.append(0.0)
            best_utilities.append(0.0)
            continue

        rarity = 0.0 if max_count <= 0 else float(max_count - counts[category_id]) / float(max_count + 1.0)
        beta = base_beta * (1.0 + rarity_beta_gain * rarity)
        beta2 = beta * beta
        best_threshold = base_score_min
        best_precision = 0.0
        best_recall = 0.0
        best_utility = -1e9
        for threshold, tp, fp in curve:
            precision = float(tp) / float(max(1, tp + fp))
            recall = float(tp) / float(max(1, total_gt))
            f_beta = 0.0
            if precision > 0.0 and recall > 0.0:
                f_beta = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
            utility = f_beta - precision_penalty * max(0.0, precision_floor - precision)
            if utility > best_utility:
                best_utility = utility
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        thresholds[category_id] = _clip_threshold(best_threshold, min_score_min=min_score_min, max_score_min=max_score_min)
        best_precisions.append(float(best_precision))
        best_recalls.append(float(best_recall))
        best_utilities.append(float(best_utility))

    return thresholds, {
        "method": "selected_fbeta",
        "class_counts": counts,
        "gt_counts": gt_counts,
        "candidate_counts": candidate_counts,
        "best_precisions": best_precisions,
        "best_recalls": best_recalls,
        "best_utilities": best_utilities,
    }


def compute_pseudo_recalibration(
    target_train_dicts: list[dict[str, Any]],
    labeled_ids: set[str],
    *,
    num_classes: int,
    base_score_min: float,
    recalibration_cfg: Any,
    teacher_adapter=None,
    stage_idx: int = 0,
    seed: int = 42,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Return class-specific hard pseudo-label thresholds and diagnostics."""

    counts = label_class_counts(target_train_dicts, labeled_ids, num_classes)
    method = str(getattr(recalibration_cfg, "method", "label_rarity")).strip().lower()
    if method in {"count", "count_rarity", "class_count", "class_rarity", "label_rarity"}:
        thresholds = _count_rarity_thresholds(counts, base_score_min=base_score_min, cfg=recalibration_cfg)
        return thresholds, {"method": "label_rarity", "class_counts": counts}
    if method in {"label_rarity_stage_scaled", "class_rarity_stage_scaled", "count_rarity_stage_scaled"}:
        return _label_rarity_function_thresholds(
            counts,
            base_score_min=base_score_min,
            cfg=recalibration_cfg,
            method_name="label_rarity_stage_scaled",
            stage_idx=stage_idx,
            use_exp=False,
            use_stage_scale=True,
        )
    if method in {"label_rarity_exp", "class_rarity_exp", "count_rarity_exp"}:
        return _label_rarity_function_thresholds(
            counts,
            base_score_min=base_score_min,
            cfg=recalibration_cfg,
            method_name="label_rarity_exp",
            stage_idx=stage_idx,
            use_exp=True,
            use_stage_scale=False,
        )
    if method in {
        "label_rarity_exp_stage_scaled",
        "label_rarity_stage_scaled_exp",
        "class_rarity_exp_stage_scaled",
        "count_rarity_exp_stage_scaled",
    }:
        return _label_rarity_function_thresholds(
            counts,
            base_score_min=base_score_min,
            cfg=recalibration_cfg,
            method_name="label_rarity_exp_stage_scaled",
            stage_idx=stage_idx,
            use_exp=True,
            use_stage_scale=True,
        )

    calibration_seed = int(seed) + 1009 * (int(stage_idx) + 1)
    candidate_score_floor = max(0.0, float(getattr(recalibration_cfg, "candidate_score_floor", 0.01)))

    if method in {"label_prior_ratio", "prior_ratio", "label_prior_quota", "prior_quota"}:
        pool = str(getattr(recalibration_cfg, "calibration_pool", "unlabeled"))
        max_images = int(getattr(recalibration_cfg, "calibration_max_images", 512))
        items = _calibration_items(
            target_train_dicts,
            labeled_ids,
            pool=pool,
            max_images=max_images,
            seed=calibration_seed,
        )
        teacher_items = _teacher_items(teacher_adapter, items, seed=calibration_seed + 17)
        scores = _scores_by_class(
            teacher_items,
            num_classes=num_classes,
            candidate_score_floor=candidate_score_floor,
        )
        if method in {"label_prior_ratio", "prior_ratio"}:
            return _label_prior_ratio(counts, scores, base_score_min=base_score_min, cfg=recalibration_cfg)
        return _label_prior_quota(counts, scores, base_score_min=base_score_min, cfg=recalibration_cfg)

    if method in {"coverage_precision", "coverage_precision_function", "selected_fbeta", "fbeta"}:
        pool = str(getattr(recalibration_cfg, "calibration_pool", "labeled"))
        max_images = int(getattr(recalibration_cfg, "calibration_max_images", 0))
        items = _calibration_items(
            target_train_dicts,
            labeled_ids,
            pool=pool,
            max_images=max_images,
            seed=calibration_seed,
        )
        teacher_items = _teacher_items(teacher_adapter, items, seed=calibration_seed + 17)
        if method in {"coverage_precision", "coverage_precision_function"}:
            return _coverage_precision(
                counts,
                teacher_items,
                num_classes=num_classes,
                base_score_min=base_score_min,
                cfg=recalibration_cfg,
            )
        return _selected_fbeta(
            counts,
            teacher_items,
            num_classes=num_classes,
            base_score_min=base_score_min,
            cfg=recalibration_cfg,
        )

    raise ValueError(f"Unsupported pseudo_recalibration.method: {method}")
