"""Soft latent-query activation losses for DINO-style SFDA.

The selector is fitted elsewhere from sparse target labels. This module only
turns selected below-threshold teacher queries into a small query-level
classification regularizer, without adding pseudo boxes or detector parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data.daod.analysis import raw_output_to_query_rows


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


@dataclass
class SoftQueryActivationLossStats:
    targets: int = 0
    matched: int = 0
    weight_sum: float = 0.0
    risk_weight_sum: float = 0.0
    match_iou_sum: float = 0.0

    def update(
        self,
        *,
        matched: bool,
        weight: float,
        match_iou: float | None,
        risk_weight: float = 1.0,
    ) -> None:
        self.targets += 1
        if not matched:
            return
        self.matched += 1
        self.weight_sum += float(weight)
        self.risk_weight_sum += float(risk_weight)
        if match_iou is not None:
            self.match_iou_sum += float(match_iou)

    def as_dict(self) -> dict[str, Any]:
        return {
            "targets": int(self.targets),
            "matched": int(self.matched),
            "mean_weight": float(self.weight_sum / self.matched) if self.matched > 0 else None,
            "mean_risk_weight": float(self.risk_weight_sum / self.matched) if self.matched > 0 else None,
            "mean_match_iou": float(self.match_iou_sum / self.matched) if self.matched > 0 else None,
        }


@dataclass
class BenefitRiskClassGate:
    gates: list[float]
    budgets: list[float] | None
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.summary,
            "gates": [float(value) for value in self.gates],
            "budgets": None if self.budgets is None else [float(value) for value in self.budgets],
        }


def soft_query_activation_loss(
    soft_items: list[dict[str, Any]],
    *,
    objective: str,
    loss_weight: float,
    match_mode: str,
    min_match_iou: float,
    match_class_aware: bool,
    positive_target: float,
    margin: float,
    activation_weight_power: float,
    min_activation_weight: float,
    distill_temperature: float,
    distill_negative_weight: float,
    distill_boost_selected: bool,
    class_gates: list[float] | dict[int, float] | None = None,
    class_budgets: list[float] | dict[int, float] | None = None,
    query_risk_cfg: Any | None = None,
) -> tuple[torch.Tensor, SoftQueryActivationLossStats]:
    """Compute a soft activation loss on selected teacher query rows.

    Each `soft_item` must contain `sample`, `teacher_raw`, `teacher_rows`, and
    `student_raw`. The selected teacher rows are produced by
    `LatentQueryActivator.select`, but they are used only as soft query-level
    targets here.
    """

    objective = str(objective).strip().lower()
    if objective not in {"class_bce", "margin", "distill_bce"}:
        raise ValueError(
            "method.soft_query_activation.objective must be one of "
            "class_bce/margin/distill_bce, "
            f"got {objective!r}"
        )
    match_mode = str(match_mode).strip().lower()
    if match_mode not in {"query_index", "box_iou"}:
        raise ValueError(
            "method.soft_query_activation.match_mode must be query_index or box_iou, "
            f"got {match_mode!r}"
        )

    device = _resolve_loss_device(soft_items)
    stats = SoftQueryActivationLossStats()
    loss_records: list[dict[str, Any]] = []
    for item in soft_items:
        student_raw = item["student_raw"]
        student_logits = student_raw["pred_logits"]
        teacher_logits_all = item["teacher_raw"]["pred_logits"]
        student_rows = None
        if match_mode == "box_iou":
            sample = item["sample"]
            student_rows = raw_output_to_query_rows(
                student_raw,
                image_size=(int(sample["height"]), int(sample["width"])),
            )
        for teacher_row in item.get("teacher_rows", []):
            class_id = int(teacher_row.get("category_id", -1))
            if class_id < 0 or class_id >= student_logits.shape[-1]:
                stats.update(matched=False, weight=0.0, match_iou=None)
                continue
            student_query_index, match_iou = _match_student_query(
                teacher_row,
                student_logits=student_logits,
                student_rows=student_rows,
                match_mode=match_mode,
                min_match_iou=min_match_iou,
                match_class_aware=match_class_aware,
            )
            if student_query_index is None:
                stats.update(matched=False, weight=0.0, match_iou=match_iou)
                continue

            activation_score = float(teacher_row.get("_latent_activation_score", teacher_row.get("score", 0.0)))
            activation_score = float(np.clip(activation_score, 0.0, 1.0))
            query_weight = max(
                float(min_activation_weight),
                float(activation_score ** max(float(activation_weight_power), 0.0)),
            )
            class_gate = _class_value(class_gates, class_id, default=1.0)
            query_risk_weight = _query_risk_weight(teacher_row, query_risk_cfg)
            raw_weight = query_weight * class_gate * query_risk_weight
            loss_records.append(
                {
                    "class_id": class_id,
                    "raw_weight": float(raw_weight),
                    "risk_weight": float(query_risk_weight),
                    "match_iou": match_iou,
                    "loss": _single_query_loss(
                        objective=objective,
                        student_logits=student_logits[int(student_query_index)],
                        teacher_logits=teacher_logits_all[int(teacher_row["query_index"])].to(device),
                        class_id=class_id,
                        positive_target=positive_target,
                        margin=margin,
                        distill_temperature=distill_temperature,
                        distill_negative_weight=distill_negative_weight,
                        distill_boost_selected=distill_boost_selected,
                    ),
                }
            )

    loss_terms: list[torch.Tensor] = []
    class_weight_sums: dict[int, float] = {}
    for record in loss_records:
        class_id = int(record["class_id"])
        class_weight_sums[class_id] = class_weight_sums.get(class_id, 0.0) + float(record["raw_weight"])
    for record in loss_records:
        class_id = int(record["class_id"])
        budget = _class_value(class_budgets, class_id, default=float("inf"))
        scale = 1.0
        if np.isfinite(budget):
            scale = min(1.0, max(float(budget), 0.0) / max(class_weight_sums.get(class_id, 0.0), 1e-12))
        final_weight = float(record["raw_weight"]) * float(scale)
        stats.update(
            matched=True,
            weight=final_weight,
            match_iou=record["match_iou"],
            risk_weight=float(record["risk_weight"]),
        )
        if final_weight > 0.0:
            loss_terms.append(final_weight * record["loss"])
    if not loss_terms:
        return torch.tensor(0.0, device=device), stats
    return float(loss_weight) * torch.stack(loss_terms).mean(), stats


def fit_benefit_risk_class_gate(
    teacher_items: list[dict[str, Any]],
    *,
    activator: Any,
    thresholds: list[float],
    num_classes: int,
    gate_cfg: Any,
    dedup_iou_thresh: float,
) -> BenefitRiskClassGate:
    """Fit a sparse-GT safety gate for soft query activation.

    The audit estimates, per class, whether selected below-threshold queries
    recover GT objects missed by ordinary DDT pseudo labels, or mostly add
    risky positives. The output is a fixed vector of class loss multipliers.
    """

    match_iou = float(_cfg_get(gate_cfg, "match_iou", 0.5))
    z_value = float(_cfg_get(gate_cfg, "confidence_z", 1.0))
    precision_power = float(_cfg_get(gate_cfg, "precision_power", 1.0))
    recovery_power = float(_cfg_get(gate_cfg, "recovery_power", 1.0))
    risk_power = float(_cfg_get(gate_cfg, "risk_power", 1.0))
    normalize = bool(_cfg_get(gate_cfg, "normalize", True))
    gate_floor = float(_cfg_get(gate_cfg, "gate_floor", 0.0))
    gate_max = float(_cfg_get(gate_cfg, "gate_max", 1.0))
    budget_cfg = _cfg_get(gate_cfg, "budget", {})
    budget_enabled = bool(_cfg_get(budget_cfg, "enabled", False))
    budget_scale = float(_cfg_get(budget_cfg, "scale", 2.0))
    budget_power = float(_cfg_get(budget_cfg, "need_power", 0.5))
    budget_min = float(_cfg_get(budget_cfg, "min_budget", 0.0))

    counts = {
        "gt": [0 for _ in range(num_classes)],
        "base_covered": [0 for _ in range(num_classes)],
        "base_missed": [0 for _ in range(num_classes)],
        "soft_selected": [0 for _ in range(num_classes)],
        "soft_matched": [0 for _ in range(num_classes)],
        "soft_recovered": [0 for _ in range(num_classes)],
        "soft_false": [0 for _ in range(num_classes)],
    }
    for item in teacher_items:
        sample = item["sample"]
        annotations = [ann for ann in sample.get("annotations", []) if 0 <= int(ann["category_id"]) < num_classes]
        pseudo_rows = _filter_rows(
            item.get("query_rows", []),
            thresholds=thresholds,
            dedup_iou_thresh=dedup_iou_thresh,
        )
        soft_rows, _ = activator.select(
            item.get("query_rows", []),
            thresholds=thresholds,
            dedup_iou_thresh=dedup_iou_thresh,
            sample=sample,
            existing_rows=pseudo_rows,
        )
        _accumulate_audit_counts(
            counts,
            annotations=annotations,
            pseudo_rows=pseudo_rows,
            soft_rows=soft_rows,
            match_iou=match_iou,
        )

    raw_gates = []
    class_stats: dict[str, Any] = {}
    for class_id in range(num_classes):
        selected = counts["soft_selected"][class_id]
        matched = counts["soft_matched"][class_id]
        recovered = counts["soft_recovered"][class_id]
        base_missed = counts["base_missed"][class_id]
        false_count = counts["soft_false"][class_id]
        precision_lcb = _wilson_lower(matched, selected, z=z_value)
        recovery_lcb = _wilson_lower(recovered, base_missed, z=z_value)
        risk_ucb = _wilson_upper(false_count, selected, z=z_value)
        raw_gate = (
            (precision_lcb ** max(precision_power, 0.0))
            * (recovery_lcb ** max(recovery_power, 0.0))
            * (max(0.0, 1.0 - risk_ucb) ** max(risk_power, 0.0))
        )
        raw_gates.append(float(raw_gate))
        class_stats[str(class_id)] = {
            "gt": int(counts["gt"][class_id]),
            "base_covered": int(counts["base_covered"][class_id]),
            "base_missed": int(base_missed),
            "soft_selected": int(selected),
            "soft_matched": int(matched),
            "soft_recovered": int(recovered),
            "soft_false": int(false_count),
            "precision_lcb": float(precision_lcb),
            "recovery_lcb": float(recovery_lcb),
            "risk_ucb": float(risk_ucb),
            "raw_gate": float(raw_gate),
        }

    max_raw_gate = max(raw_gates) if raw_gates else 0.0
    if normalize and max_raw_gate > 0.0:
        gates = [value / max_raw_gate for value in raw_gates]
    else:
        gates = list(raw_gates)
    gates = [float(np.clip(max(value, gate_floor), 0.0, gate_max)) for value in gates]

    budgets = None
    if budget_enabled:
        max_missed = max(counts["base_missed"]) if counts["base_missed"] else 0
        budgets = []
        for class_id, gate in enumerate(gates):
            if gate <= 0.0 or max_missed <= 0:
                budgets.append(0.0)
                continue
            need = counts["base_missed"][class_id] / max(float(max_missed), 1.0)
            budget = budget_scale * gate * (max(float(need), 0.0) ** max(budget_power, 0.0))
            budgets.append(float(max(budget, budget_min if gate > 0.0 else 0.0)))

    for class_id, gate in enumerate(gates):
        class_stats[str(class_id)]["gate"] = float(gate)
        class_stats[str(class_id)]["budget"] = None if budgets is None else float(budgets[class_id])

    summary = {
        "enabled": True,
        "method": "benefit_risk_class_gate",
        "match_iou": float(match_iou),
        "confidence_z": float(z_value),
        "normalize": bool(normalize),
        "gate_floor": float(gate_floor),
        "gate_max": float(gate_max),
        "budget_enabled": bool(budget_enabled),
        "budget_scale": float(budget_scale) if budget_enabled else None,
        "budget_need_power": float(budget_power) if budget_enabled else None,
        "class_stats": class_stats,
    }
    return BenefitRiskClassGate(gates=gates, budgets=budgets, summary=summary)


def _resolve_loss_device(soft_items: list[dict[str, Any]]) -> torch.device:
    for item in soft_items:
        student_raw = item.get("student_raw", {})
        pred_logits = student_raw.get("pred_logits")
        if torch.is_tensor(pred_logits):
            return pred_logits.device
    return torch.device("cpu")


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _class_value(values: list[float] | dict[int, float] | None, class_id: int, *, default: float) -> float:
    if values is None:
        return float(default)
    if isinstance(values, dict):
        return float(values.get(int(class_id), default))
    if 0 <= int(class_id) < len(values):
        return float(values[int(class_id)])
    return float(default)


def _query_risk_weight(row: dict[str, Any], cfg: Any | None) -> float:
    if cfg is None or not bool(_cfg_get(cfg, "enabled", False)):
        return 1.0

    min_weight = float(np.clip(float(_cfg_get(cfg, "min_weight", 0.35)), 0.0, 1.0))
    max_weight = float(np.clip(float(_cfg_get(cfg, "max_weight", 1.0)), min_weight, 1.0))
    power = max(float(_cfg_get(cfg, "power", 1.0)), 0.0)
    aggregate = str(_cfg_get(cfg, "aggregate", "weighted_mean")).strip().lower()
    weight_cfg = _cfg_get(cfg, "weights", {})
    weights = _query_risk_weights(weight_cfg)

    score = float(row.get("score", 0.0))
    pseudo_threshold = float(row.get("_pseudo_threshold", 0.0))
    if pseudo_threshold > 1e-6:
        score_proximity = _clip01(score / pseudo_threshold)
    else:
        score_proximity = _clip01(score)
    signals = {
        "activation": _clip01(float(row.get("_latent_activation_score", score))),
        "score_proximity": score_proximity,
        "margin": _clip01(float(row.get("softmax_margin", 0.0))),
        "confidence": 1.0 - _clip01(float(row.get("softmax_entropy", 1.0))),
        "box_stability": 1.0 - _clip01(float(row.get("decoder_box_iou_gap", 1.0))),
        "center_stability": 1.0 - _clip01(float(row.get("decoder_center_shift", 1.0))),
    }
    weighted_values = [
        (float(weights[name]), float(value))
        for name, value in signals.items()
        if float(weights.get(name, 0.0)) > 0.0
    ]
    if not weighted_values:
        quality = 1.0
    elif aggregate == "geometric":
        weight_sum = sum(weight for weight, _ in weighted_values)
        quality = np.exp(
            sum(weight * np.log(max(value, 1e-6)) for weight, value in weighted_values) / max(weight_sum, 1e-12)
        )
    else:
        weight_sum = sum(weight for weight, _ in weighted_values)
        quality = sum(weight * value for weight, value in weighted_values) / max(weight_sum, 1e-12)
    quality = _clip01(float(quality) ** power)
    return float(min_weight + (max_weight - min_weight) * quality)


def _query_risk_weights(cfg: Any) -> dict[str, float]:
    defaults = {
        "activation": 0.35,
        "score_proximity": 0.25,
        "margin": 0.10,
        "confidence": 0.10,
        "box_stability": 0.15,
        "center_stability": 0.05,
    }
    if isinstance(cfg, dict):
        defaults.update({str(key): max(float(value), 0.0) for key, value in cfg.items()})
    else:
        for name in list(defaults):
            if hasattr(cfg, name):
                defaults[name] = max(float(getattr(cfg, name)), 0.0)
    return defaults


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _filter_rows(
    query_rows: list[dict[str, Any]],
    *,
    thresholds: list[float],
    dedup_iou_thresh: float,
) -> list[dict[str, Any]]:
    rows = []
    for row in query_rows:
        class_id = int(row.get("category_id", -1))
        if class_id < 0 or class_id >= len(thresholds):
            continue
        if float(row.get("score", 0.0)) >= float(thresholds[class_id]):
            rows.append(dict(row))
    return _deduplicate_rows(rows, iou_thresh=dedup_iou_thresh)


def _deduplicate_rows(rows: list[dict[str, Any]], *, iou_thresh: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        suppress = False
        for other in kept:
            if int(other["category_id"]) != int(row["category_id"]):
                continue
            if _xyxy_iou(other["bbox"], row["bbox"]) >= float(iou_thresh):
                suppress = True
                break
        if not suppress:
            kept.append(row)
    return kept


def _accumulate_audit_counts(
    counts: dict[str, list[int]],
    *,
    annotations: list[dict[str, Any]],
    pseudo_rows: list[dict[str, Any]],
    soft_rows: list[dict[str, Any]],
    match_iou: float,
) -> None:
    missed_gt_indices: dict[int, set[int]] = {}
    for gt_index, ann in enumerate(annotations):
        class_id = int(ann["category_id"])
        counts["gt"][class_id] += 1
        covered = any(
            int(row["category_id"]) == class_id
            and _xyxy_iou([float(v) for v in row["bbox"]], [float(v) for v in ann["bbox"]]) >= float(match_iou)
            for row in pseudo_rows
        )
        if covered:
            counts["base_covered"][class_id] += 1
        else:
            counts["base_missed"][class_id] += 1
            missed_gt_indices.setdefault(class_id, set()).add(gt_index)

    recovered_gt_indices: dict[int, set[int]] = {class_id: set() for class_id in missed_gt_indices}
    for row in soft_rows:
        class_id = int(row.get("category_id", -1))
        if class_id < 0 or class_id >= len(counts["soft_selected"]):
            continue
        counts["soft_selected"][class_id] += 1
        best_gt_index = None
        best_iou = 0.0
        for gt_index, ann in enumerate(annotations):
            if int(ann["category_id"]) != class_id:
                continue
            iou = _xyxy_iou([float(v) for v in row["bbox"]], [float(v) for v in ann["bbox"]])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is None or best_iou < float(match_iou):
            counts["soft_false"][class_id] += 1
            continue
        counts["soft_matched"][class_id] += 1
        if (
            best_gt_index in missed_gt_indices.get(class_id, set())
            and best_gt_index not in recovered_gt_indices.setdefault(class_id, set())
        ):
            recovered_gt_indices[class_id].add(best_gt_index)
            counts["soft_recovered"][class_id] += 1


def _wilson_lower(successes: int, total: int, *, z: float) -> float:
    if total <= 0:
        return 0.0
    center, radius, denom = _wilson_parts(successes, total, z=z)
    return float(max(0.0, (center - radius) / denom))


def _wilson_upper(successes: int, total: int, *, z: float) -> float:
    if total <= 0:
        return 1.0
    center, radius, denom = _wilson_parts(successes, total, z=z)
    return float(min(1.0, (center + radius) / denom))


def _wilson_parts(successes: int, total: int, *, z: float) -> tuple[float, float, float]:
    z = max(float(z), 0.0)
    n = max(float(total), 1.0)
    p = float(successes) / n
    denom = 1.0 + (z * z / n)
    center = p + (z * z / (2.0 * n))
    radius = z * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
    return float(center), float(radius), float(denom)


def _match_student_query(
    teacher_row: dict[str, Any],
    *,
    student_logits: torch.Tensor,
    student_rows: list[dict[str, Any]] | None,
    match_mode: str,
    min_match_iou: float,
    match_class_aware: bool,
) -> tuple[int | None, float | None]:
    if match_mode == "query_index":
        query_index = int(teacher_row["query_index"])
        if query_index < 0 or query_index >= student_logits.shape[0]:
            return None, None
        return query_index, None

    if not student_rows:
        return None, None
    best_query_index: int | None = None
    best_iou = -1.0
    teacher_class = int(teacher_row["category_id"])
    teacher_box = [float(v) for v in teacher_row["bbox"]]
    for student_row in student_rows:
        if match_class_aware and int(student_row["category_id"]) != teacher_class:
            continue
        iou = _xyxy_iou(teacher_box, [float(v) for v in student_row["bbox"]])
        if iou > best_iou:
            best_iou = iou
            best_query_index = int(student_row["query_index"])
    if best_query_index is None or best_iou < float(min_match_iou):
        return None, max(best_iou, 0.0)
    return best_query_index, float(best_iou)


def _single_query_loss(
    *,
    objective: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    class_id: int,
    positive_target: float,
    margin: float,
    distill_temperature: float,
    distill_negative_weight: float,
    distill_boost_selected: bool,
) -> torch.Tensor:
    if objective == "class_bce":
        target = torch.full_like(student_logits[class_id], fill_value=float(positive_target))
        return F.binary_cross_entropy_with_logits(student_logits[class_id], target)

    if objective == "margin":
        if student_logits.numel() <= 1:
            return torch.relu(torch.as_tensor(float(margin), device=student_logits.device) - student_logits[class_id])
        other_mask = torch.ones(student_logits.shape[-1], dtype=torch.bool, device=student_logits.device)
        other_mask[class_id] = False
        other_max = student_logits[other_mask].max()
        return F.relu(float(margin) - (student_logits[class_id] - other_max))

    temperature = max(float(distill_temperature), 1e-6)
    target_probs = torch.sigmoid(teacher_logits / temperature).detach()
    if distill_boost_selected:
        target_probs = target_probs.clone()
        target_probs[class_id] = torch.maximum(
            target_probs[class_id],
            torch.as_tensor(float(positive_target), dtype=target_probs.dtype, device=target_probs.device),
        )
    losses = F.binary_cross_entropy_with_logits(student_logits, target_probs, reduction="none")
    weights = torch.full_like(losses, fill_value=max(float(distill_negative_weight), 0.0))
    weights[class_id] = 1.0
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
