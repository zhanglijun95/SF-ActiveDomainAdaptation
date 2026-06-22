"""Soft query-revival loss for sparse-label SFDA object detection.

This module uses missed-query recovery candidates as weak objectness hints,
without appending them as full pseudo labels. The intent is to test whether the
oracle-recovery headroom can be approached more safely than hard pseudo-box
injection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
class QueryRevivalLossStats:
    targets: int = 0
    matched: int = 0
    weight_sum: float = 0.0
    score_sum: float = 0.0
    gate_sum: float = 0.0
    match_iou_sum: float = 0.0

    def add(self, other: "QueryRevivalLossStats") -> None:
        self.targets += int(other.targets)
        self.matched += int(other.matched)
        self.weight_sum += float(other.weight_sum)
        self.score_sum += float(other.score_sum)
        self.gate_sum += float(other.gate_sum)
        self.match_iou_sum += float(other.match_iou_sum)

    def update(
        self,
        *,
        matched: bool,
        weight: float,
        score: float,
        gate: float,
        match_iou: float | None,
    ) -> None:
        self.targets += 1
        if not matched:
            return
        self.matched += 1
        self.weight_sum += float(weight)
        self.score_sum += float(score)
        self.gate_sum += float(gate)
        if match_iou is not None:
            self.match_iou_sum += float(match_iou)

    def as_dict(self) -> dict[str, Any]:
        return {
            "targets": int(self.targets),
            "matched": int(self.matched),
            "mean_weight": float(self.weight_sum / self.matched) if self.matched > 0 else None,
            "mean_score": float(self.score_sum / self.matched) if self.matched > 0 else None,
            "mean_gate": float(self.gate_sum / self.matched) if self.matched > 0 else None,
            "mean_match_iou": float(self.match_iou_sum / self.matched) if self.matched > 0 else None,
            "match_rate": float(self.matched / max(self.targets, 1)),
        }


def query_revival_loss(
    revival_items: list[dict[str, Any]],
    *,
    loss_weight: float,
    match_mode: str,
    min_match_iou: float,
    match_class_aware: bool,
    positive_target: float,
    foreground_pool: str,
    foreground_temperature: float,
    recovery_weight_power: float,
    min_candidate_weight: float,
    class_budgets: list[float] | dict[int, float] | None = None,
) -> tuple[torch.Tensor, QueryRevivalLossStats]:
    """Compute a low-weight foreground loss on selected recovery candidates."""

    match_mode = str(match_mode).strip().lower()
    if match_mode not in {"query_index", "box_iou"}:
        raise ValueError(f"method.query_recovery.revival_loss.match_mode must be query_index or box_iou, got {match_mode!r}")
    foreground_pool = str(foreground_pool).strip().lower()
    if foreground_pool not in {"mean_logsumexp", "max", "top2_mean"}:
        raise ValueError(
            "method.query_recovery.revival_loss.foreground_pool must be "
            f"mean_logsumexp, max, or top2_mean, got {foreground_pool!r}"
        )

    device = _resolve_loss_device(revival_items)
    stats = QueryRevivalLossStats()
    records: list[dict[str, Any]] = []
    for item in revival_items:
        sample = item["sample"]
        student_raw = item["student_raw"]
        student_logits = student_raw["pred_logits"]
        student_rows = None
        if match_mode == "box_iou":
            student_rows = raw_output_to_query_rows(
                student_raw,
                image_size=(int(sample["height"]), int(sample["width"])),
            )
        for teacher_row in item.get("teacher_rows", []):
            class_id = int(teacher_row.get("category_id", -1))
            if class_id < 0:
                stats.update(matched=False, weight=0.0, score=0.0, gate=0.0, match_iou=None)
                continue
            query_index, match_iou = _match_student_query(
                teacher_row,
                student_logits=student_logits,
                student_rows=student_rows,
                match_mode=match_mode,
                min_match_iou=min_match_iou,
                match_class_aware=match_class_aware,
            )
            if query_index is None:
                stats.update(
                    matched=False,
                    weight=0.0,
                    score=float(teacher_row.get("_query_recovery_score", teacher_row.get("score", 0.0))),
                    gate=float(teacher_row.get("_query_recovery_gate", 1.0)),
                    match_iou=match_iou,
                )
                continue

            recovery_score = float(np.clip(float(teacher_row.get("_query_recovery_score", teacher_row.get("score", 0.0))), 0.0, 1.0))
            recovery_gate = float(np.clip(float(teacher_row.get("_query_recovery_gate", 1.0)), 0.0, 1.0))
            candidate_weight = max(
                float(min_candidate_weight),
                recovery_score ** max(float(recovery_weight_power), 0.0),
            )
            raw_weight = float(candidate_weight * recovery_gate)
            foreground_logit = _foreground_logit(
                student_logits[int(query_index)],
                pool=foreground_pool,
                temperature=foreground_temperature,
            )
            target = torch.as_tensor(
                float(np.clip(float(positive_target), 0.0, 1.0)),
                dtype=foreground_logit.dtype,
                device=foreground_logit.device,
            )
            records.append(
                {
                    "class_id": int(class_id),
                    "raw_weight": float(raw_weight),
                    "score": float(recovery_score),
                    "gate": float(recovery_gate),
                    "match_iou": match_iou,
                    "loss": F.binary_cross_entropy_with_logits(foreground_logit, target),
                }
            )

    class_weight_sums: dict[int, float] = {}
    for record in records:
        class_id = int(record["class_id"])
        class_weight_sums[class_id] = class_weight_sums.get(class_id, 0.0) + float(record["raw_weight"])

    loss_terms: list[torch.Tensor] = []
    for record in records:
        class_id = int(record["class_id"])
        budget = _class_value(class_budgets, class_id, default=float("inf"))
        scale = 1.0
        if np.isfinite(budget):
            scale = min(1.0, max(float(budget), 0.0) / max(class_weight_sums.get(class_id, 0.0), 1e-12))
        final_weight = float(record["raw_weight"]) * float(scale)
        stats.update(
            matched=True,
            weight=final_weight,
            score=float(record["score"]),
            gate=float(record["gate"]),
            match_iou=record["match_iou"],
        )
        if final_weight > 0.0:
            loss_terms.append(float(final_weight) * record["loss"])

    if not loss_terms:
        return torch.tensor(0.0, device=device), stats
    return float(loss_weight) * torch.stack(loss_terms).mean(), stats


def _resolve_loss_device(items: list[dict[str, Any]]) -> torch.device:
    for item in items:
        pred_logits = item.get("student_raw", {}).get("pred_logits")
        if torch.is_tensor(pred_logits):
            return pred_logits.device
    return torch.device("cpu")


def _class_value(values: list[float] | dict[int, float] | None, class_id: int, *, default: float) -> float:
    if values is None:
        return float(default)
    if isinstance(values, dict):
        return float(values.get(int(class_id), default))
    if 0 <= int(class_id) < len(values):
        return float(values[int(class_id)])
    return float(default)


def _foreground_logit(
    logits: torch.Tensor,
    *,
    pool: str,
    temperature: float,
) -> torch.Tensor:
    if pool == "max":
        return logits.max()
    if pool == "top2_mean":
        top_k = min(2, int(logits.numel()))
        return logits.topk(top_k).values.mean()
    temperature = max(float(temperature), 1e-6)
    return temperature * torch.logsumexp(logits / temperature, dim=-1) - temperature * math.log(max(int(logits.numel()), 1))


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
        query_index = int(teacher_row.get("query_index", -1))
        if query_index < 0 or query_index >= student_logits.shape[0]:
            return None, None
        return query_index, None

    if not student_rows:
        return None, None
    best_query_index = None
    best_iou = -1.0
    teacher_class = int(teacher_row.get("category_id", -1))
    teacher_box = [float(v) for v in teacher_row["bbox"]]
    for student_row in student_rows:
        if match_class_aware and int(student_row.get("category_id", -2)) != teacher_class:
            continue
        iou = _xyxy_iou(teacher_box, [float(v) for v in student_row["bbox"]])
        if iou > best_iou:
            best_iou = float(iou)
            best_query_index = int(student_row["query_index"])
    if best_query_index is None or best_iou < float(min_match_iou):
        return None, max(best_iou, 0.0)
    return int(best_query_index), float(best_iou)
