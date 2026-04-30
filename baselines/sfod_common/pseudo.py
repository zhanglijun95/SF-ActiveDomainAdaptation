"""Pseudo-label utilities shared by isolated SFOD baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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


def filter_pseudo_rows(
    query_rows: list[dict[str, Any]],
    *,
    threshold: float,
    dedup_iou_thresh: float,
) -> list[dict[str, Any]]:
    filtered = [
        dict(row)
        for row in query_rows
        if int(row.get("category_id", -1)) >= 0 and float(row.get("score", 0.0)) >= float(threshold)
    ]
    return deduplicate_rows(filtered, iou_thresh=dedup_iou_thresh)


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


def _weighted_score(specs: list[tuple[str, float]], values: dict[str, float]) -> tuple[float, dict[str, float]]:
    total = 0.0
    parts: dict[str, float] = {}
    for name, weight in specs:
        if name not in values:
            raise KeyError(f"Unknown SFOD soft-routing signal: {name}")
        value = float(values[name])
        total += float(weight) * value
        parts[name] = value
    return float(total), parts


def signal_specs(section: Any, default_specs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    signal_cfg = getattr(section, "signals", None)
    if signal_cfg is None:
        return default_specs
    return [(str(item.name), float(getattr(item, "weight", 1.0))) for item in signal_cfg]


def routing_signal_values(
    teacher_row: dict[str, Any],
    *,
    student_query_rows: list[dict[str, Any]],
) -> dict[str, float]:
    entropy = float(np.clip(teacher_row.get("softmax_entropy", 1.0), 0.0, 1.0))
    margin = float(np.clip(teacher_row.get("softmax_margin", 0.0), 0.0, 1.0))
    logit_sharpness = float(np.clip((1.0 - entropy) * margin, 0.0, 1.0))
    decoder_box_stability = float(np.clip(1.0 - teacher_row.get("decoder_box_iou_gap", 1.0), 0.0, 1.0))
    decoder_class_stability = float(np.clip(1.0 - teacher_row.get("decoder_top_class_flip", 1.0), 0.0, 1.0))

    teacher_student_agreement = 0.0
    teacher_box = [float(v) for v in teacher_row["bbox"]]
    for student_row in student_query_rows:
        if int(student_row["category_id"]) != int(teacher_row["category_id"]):
            continue
        student_box = [float(v) for v in student_row["bbox"]]
        iou = xyxy_iou(teacher_box, student_box)
        score_gap = abs(float(teacher_row["score"]) - float(student_row["score"]))
        pair_score = float(np.clip(iou, 0.0, 1.0)) * float(np.clip(1.0 - score_gap, 0.0, 1.0))
        teacher_student_agreement = max(teacher_student_agreement, pair_score)

    return {
        "score": float(np.clip(float(teacher_row["score"]), 0.0, 1.0)),
        "logit_margin": float(np.clip(margin, 0.0, 1.0)),
        "logit_sharpness": logit_sharpness,
        "decoder_box_stability": decoder_box_stability,
        "decoder_class_stability": decoder_class_stability,
        "teacher_student_agreement": float(np.clip(teacher_student_agreement, 0.0, 1.0)),
    }


def build_low_confidence_targets(
    teacher_item: dict[str, Any],
    student_item: dict[str, Any],
    *,
    hard_rows: list[dict[str, Any]],
    score_min: float,
    score_max: float,
    routing_specs: list[tuple[str, float]],
    routing_threshold: float,
    hard_exclusion_iou_max: float,
    pre_routing_topk: int = 256,
    max_targets: int = 128,
) -> list[dict[str, Any]]:
    teacher_logits = teacher_item["raw_output"]["pred_logits"]
    candidates: list[dict[str, Any]] = []
    for teacher_row in teacher_item["query_rows"]:
        score = float(teacher_row["score"])
        if not (float(score_min) <= score < float(score_max)):
            continue
        if hard_rows:
            max_hard_iou = max(xyxy_iou(teacher_row["bbox"], hard_row["bbox"]) for hard_row in hard_rows)
            if max_hard_iou > float(hard_exclusion_iou_max):
                continue
        candidates.append(teacher_row)

    candidates.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    if int(pre_routing_topk) > 0:
        candidates = candidates[: int(pre_routing_topk)]

    low_targets: list[dict[str, Any]] = []
    for teacher_row in candidates:
        signal_values = routing_signal_values(
            teacher_row,
            student_query_rows=student_item["student_query_rows"],
        )
        routing_score, routing_parts = _weighted_score(routing_specs, signal_values)
        if routing_score < float(routing_threshold):
            continue
        low_targets.append(
            {
                "teacher_row": teacher_row,
                "teacher_logits": teacher_logits[int(teacher_row["query_index"])].detach().cpu(),
                "routing_score": float(routing_score),
                "routing_signals": dict(routing_parts),
            }
        )
    low_targets.sort(key=lambda item: float(item.get("routing_score", 0.0)), reverse=True)
    if int(max_targets) > 0:
        low_targets = low_targets[: int(max_targets)]
    return low_targets


def _match_student_row(
    teacher_row: dict[str, Any],
    student_query_rows: list[dict[str, Any]],
    *,
    match_iou_min: float,
) -> tuple[dict[str, Any] | None, float]:
    best_student = None
    best_iou = -1.0
    teacher_box = [float(v) for v in teacher_row["bbox"]]
    for student_row in student_query_rows:
        if int(student_row["category_id"]) != int(teacher_row["category_id"]):
            continue
        iou = xyxy_iou(teacher_box, [float(v) for v in student_row["bbox"]])
        if iou > best_iou:
            best_iou = iou
            best_student = student_row
    if best_student is None or best_iou < float(match_iou_min):
        return None, best_iou
    return best_student, best_iou


def lpld_distillation_loss(
    low_items: list[dict[str, Any]],
    *,
    weight: float,
    match_iou_min: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not low_items:
        return torch.tensor(0.0, device=device), {"low_targets": 0, "matched_targets": 0}

    losses = []
    low_count = 0
    matched_count = 0
    for low_item in low_items:
        student_raw = low_item["student_raw"]
        student_query_rows = low_item["student_query_rows"]
        for low_target in low_item["low_targets"]:
            low_count += 1
            teacher_row = low_target["teacher_row"]
            student_row, _ = _match_student_row(
                teacher_row,
                student_query_rows,
                match_iou_min=match_iou_min,
            )
            if student_row is None:
                continue
            matched_count += 1
            teacher_logits = low_target["teacher_logits"].to(device)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            student_logits = student_raw["pred_logits"][int(student_row["query_index"])]
            kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="batchmean")
            adaptive_weight = float(np.clip(low_target["routing_score"], 0.0, 1.0))
            losses.append(kl_loss * adaptive_weight)

    if not losses:
        return torch.tensor(0.0, device=device), {"low_targets": low_count, "matched_targets": matched_count}
    return float(weight) * torch.stack(losses).mean(), {"low_targets": low_count, "matched_targets": matched_count}


def lpu_low_confidence_loss(
    low_items: list[dict[str, Any]],
    *,
    pst_weight: float,
    lscl_weight: float,
    match_iou_min: float,
    positive_iou: float,
    negative_iou: float,
    contrastive_margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not low_items:
        return torch.tensor(0.0, device=device), {"low_targets": 0, "matched_targets": 0, "pst_pairs": 0, "lscl_pairs": 0}

    pst_losses = []
    lscl_losses = []
    low_count = 0
    matched_count = 0
    for low_item in low_items:
        student_raw = low_item["student_raw"]
        student_query_rows = low_item["student_query_rows"]
        matched: list[dict[str, Any]] = []
        for low_target in low_item["low_targets"]:
            low_count += 1
            teacher_row = low_target["teacher_row"]
            student_row, match_iou = _match_student_row(
                teacher_row,
                student_query_rows,
                match_iou_min=match_iou_min,
            )
            if student_row is None:
                continue
            matched_count += 1
            teacher_logits = low_target["teacher_logits"].to(device)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            student_logits = student_raw["pred_logits"][int(student_row["query_index"])]
            pst = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="batchmean")
            confidence_weight = float(np.clip(low_target["routing_score"], 0.0, 1.0))
            pst_losses.append(pst * confidence_weight)
            matched.append(
                {
                    "teacher_row": teacher_row,
                    "student_logits": student_logits,
                    "match_iou": float(match_iou),
                }
            )

        for left_idx in range(len(matched)):
            for right_idx in range(left_idx + 1, len(matched)):
                left = matched[left_idx]
                right = matched[right_idx]
                teacher_iou = xyxy_iou(left["teacher_row"]["bbox"], right["teacher_row"]["bbox"])
                same_class = int(left["teacher_row"]["category_id"]) == int(right["teacher_row"]["category_id"])
                left_probs = torch.softmax(left["student_logits"], dim=-1)
                right_probs = torch.softmax(right["student_logits"], dim=-1)
                cosine = F.cosine_similarity(left_probs.unsqueeze(0), right_probs.unsqueeze(0)).squeeze(0)
                if same_class and teacher_iou >= float(positive_iou):
                    lscl_losses.append(1.0 - cosine)
                elif (not same_class) and teacher_iou <= float(negative_iou):
                    lscl_losses.append(torch.relu(cosine - float(contrastive_margin)))

    loss = torch.tensor(0.0, device=device)
    if pst_losses:
        loss = loss + float(pst_weight) * torch.stack(pst_losses).mean()
    if lscl_losses:
        loss = loss + float(lscl_weight) * torch.stack(lscl_losses).mean()
    return loss, {
        "low_targets": low_count,
        "matched_targets": matched_count,
        "pst_pairs": len(pst_losses),
        "lscl_pairs": len(lscl_losses),
    }


def consensus_query_rows(
    dynamic_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    *,
    consensus_iou: float,
    include_single_teacher: bool,
    single_teacher_threshold: float,
    score_merge: str,
    dedup_iou_thresh: float,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    used_static: set[int] = set()
    for dyn_row in dynamic_rows:
        best_idx = None
        best_iou = -1.0
        for static_idx, static_row in enumerate(static_rows):
            if int(static_row["category_id"]) != int(dyn_row["category_id"]):
                continue
            iou = xyxy_iou(dyn_row["bbox"], static_row["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = static_idx
        if best_idx is None or best_iou < float(consensus_iou):
            continue
        static_row = static_rows[best_idx]
        used_static.add(best_idx)
        dyn_score = float(dyn_row["score"])
        static_score = float(static_row["score"])
        if score_merge == "max":
            score = max(dyn_score, static_score)
        else:
            score = 0.5 * (dyn_score + static_score)
        dyn_box = np.asarray(dyn_row["bbox"], dtype=float)
        static_box = np.asarray(static_row["bbox"], dtype=float)
        row = dict(dyn_row)
        row["bbox"] = [float(v) for v in (0.5 * (dyn_box + static_box)).tolist()]
        row["score"] = float(score)
        row["consensus_iou"] = float(best_iou)
        row["dynamic_score"] = dyn_score
        row["static_score"] = static_score
        row["pseudo_source"] = "dynamic_static_consensus"
        merged.append(row)

    if include_single_teacher:
        for source_name, rows in (("dynamic", dynamic_rows), ("static", static_rows)):
            for row_idx, row in enumerate(rows):
                if source_name == "static" and row_idx in used_static:
                    continue
                if float(row.get("score", 0.0)) < float(single_teacher_threshold):
                    continue
                single = dict(row)
                single["pseudo_source"] = f"{source_name}_single_high_confidence"
                merged.append(single)

    return deduplicate_rows(merged, iou_thresh=dedup_iou_thresh)
