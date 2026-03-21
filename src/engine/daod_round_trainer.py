"""DAOD round trainer with fixed teacher, trainable student, and EMA updates.

This module implements the round update described in the DAOD method note:

- newly queried target images are trained with supervised loss
- remaining target images stay unlabeled for the round
- a fixed teacher produces pseudo targets on weak views
- the student is optimized on strong views
- hard-vs-soft pseudo routing happens online from teacher outputs
- the teacher is updated from the student by EMA at a configurable interval
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.daod import (
    DAODListDataset,
    build_strong_view_sample,
    build_weak_view_sample,
    collate_daod_batch,
    cycle_daod_loader,
    map_boxes_to_original_view,
)
from src.data.daod.analysis import raw_output_to_query_rows
from src.data.daod.detectron2 import materialize_daod_dicts
from src.engine.utils import save_json
from src.models import build_daod_model, run_daod_raw_outputs


@dataclass
class DAODRoundTrainSummary:
    student_checkpoint: str
    teacher_checkpoint: str
    train_history: list[dict[str, float]]
    source_val_metrics: dict[str, Any]
    target_val_metrics: dict[str, Any]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _limit_samples(dataset_dicts: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or int(limit) <= 0:
        return dataset_dicts
    return dataset_dicts[: int(limit)]


def _log_eval_metrics(output_dir: Path, step: int, split_name: str, metrics: dict[str, Any]) -> None:
    bbox_metrics = metrics.get("bbox", {})
    if not bbox_metrics:
        return
    writer = SummaryWriter(log_dir=str(output_dir))
    try:
        for key, value in bbox_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{split_name}/{key}", float(value), step)
    finally:
        writer.close()


def _signal_specs(section: Any, default_specs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    signal_cfg = getattr(section, "signals", None)
    if signal_cfg is None:
        return default_specs
    specs: list[tuple[str, float]] = []
    for item in signal_cfg:
        specs.append((str(item.name), float(getattr(item, "weight", 1.0))))
    return specs


def _signal_value(name: str, values: dict[str, float]) -> float:
    if name not in values:
        raise KeyError(f"Unknown routing signal: {name}")
    return float(values[name])


def _weighted_score(specs: list[tuple[str, float]], values: dict[str, float]) -> tuple[float, dict[str, float]]:
    parts: dict[str, float] = {}
    total = 0.0
    for name, weight in specs:
        value = _signal_value(name, values)
        parts[name] = value
        total += float(weight) * value
    return float(total), parts


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
    return 0.0 if union <= 0.0 else inter / union


def _resize_shape(height: int, width: int, short_edge: int, max_size: int) -> tuple[int, int]:
    scale = float(short_edge) / float(min(height, width))
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    if max(new_h, new_w) > max_size:
        scale = float(max_size) / float(max(new_h, new_w))
        new_h = int(round(new_h * scale))
        new_w = int(round(new_w * scale))
    return max(new_h, 1), max(new_w, 1)


def _resize_pil_and_boxes(image: Image.Image, boxes: list[list[float]], short_edge: int, max_size: int):
    width, height = image.size
    new_h, new_w = _resize_shape(height, width, short_edge, max_size)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    scale_x = float(new_w) / float(width)
    scale_y = float(new_h) / float(height)
    scaled_boxes = []
    for x0, y0, x1, y1 in boxes:
        scaled_boxes.append([x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y])
    return resized, scaled_boxes, new_h, new_w


def _annotations_to_instances(
    annotations: list[dict[str, Any]],
    image_size: tuple[int, int],
    *,
    device: torch.device,
) -> Instances:
    instances = Instances(image_size)
    if not annotations:
        instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32, device=device))
        instances.gt_classes = torch.zeros((0,), dtype=torch.int64, device=device)
        return instances
    boxes = torch.tensor([ann["bbox"] for ann in annotations], dtype=torch.float32, device=device)
    classes = torch.tensor([int(ann["category_id"]) for ann in annotations], dtype=torch.int64, device=device)
    instances.gt_boxes = Boxes(boxes)
    instances.gt_classes = classes
    return instances


def _make_supervised_inputs(
    adapter,
    batch: list[dict[str, Any]],
    *,
    strong_short_edge: int,
    max_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for sample in batch:
        strong_sample = build_strong_view_sample(sample, suffix="supervised_strong")
        image = strong_sample["image"]
        boxes = [ann["bbox"] for ann in sample["annotations"]]
        image, boxes, new_h, new_w = _resize_pil_and_boxes(image, boxes, strong_short_edge, max_size)
        resized_annotations = []
        for ann, box in zip(sample["annotations"], boxes):
            resized_ann = dict(ann)
            resized_ann["bbox"] = box
            resized_annotations.append(resized_ann)
        image_np = np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))
        inputs.append(
            {
                "image": torch.as_tensor(image_np, dtype=torch.float32, device=device),
                "height": new_h,
                "width": new_w,
                "file_name": sample["file_name"],
                "sample_id": sample["sample_id"],
                "image_id": sample.get("image_id", sample["sample_id"]),
                "instances": _annotations_to_instances(resized_annotations, (new_h, new_w), device=device),
            }
        )
    return inputs


def _teacher_outputs_for_unlabeled(
    teacher_adapter,
    batch: list[dict[str, Any]],
    *,
    weak_view_rng,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for sample in batch:
        image = Image.open(sample["file_name"]).convert("RGB")
        weak_sample = build_weak_view_sample(
            {**sample, "image": image},
            rng=weak_view_rng,
            suffix="teacher_weak",
        )
        weak_meta = weak_sample["view_meta"]
        raw_output = run_daod_raw_outputs(teacher_adapter, weak_sample, with_grad=False)[0]
        query_rows = raw_output_to_query_rows(
            raw_output,
            image_size=(int(sample["height"]), int(sample["width"])),
        )
        if weak_meta.get("hflip", False):
            mapped = map_boxes_to_original_view([row["bbox"] for row in query_rows], weak_meta)
            for row, box in zip(query_rows, mapped):
                row["bbox"] = box
        outputs.append(
            {
                "sample": sample,
                "raw_output": raw_output,
                "query_rows": query_rows,
            }
        )
    return outputs


def _student_outputs_for_unlabeled(
    student_adapter,
    batch: list[dict[str, Any]],
    *,
    strong_short_edge: int,
    max_size: int,
) -> list[dict[str, Any]]:
    """Run student strong-view inference for unlabeled samples once per step.

    The returned raw outputs keep gradients so the soft distillation branch can
    reuse them directly instead of rerunning student inference later.
    """

    outputs: list[dict[str, Any]] = []
    for sample in batch:
        strong_sample = build_strong_view_sample(sample, suffix="student_strong")
        resized_image, _, new_h, new_w = _resize_pil_and_boxes(strong_sample["image"], [], strong_short_edge, max_size)
        student_input_sample = dict(sample)
        student_input_sample["image"] = resized_image
        student_input_sample["height"] = new_h
        student_input_sample["width"] = new_w
        student_input_sample["sample_id"] = strong_sample["sample_id"]
        student_raw = run_daod_raw_outputs(student_adapter, student_input_sample, with_grad=True)[0]
        student_query_rows = raw_output_to_query_rows(
            student_raw,
            image_size=(int(sample["height"]), int(sample["width"])),
        )
        outputs.append(
            {
                "sample": sample,
                "student_raw": student_raw,
                "student_query_rows": student_query_rows,
            }
        )
    return outputs


def _routing_signal_values(
    teacher_row: dict[str, Any],
    *,
    student_query_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute bounded per-detection routing signals in `[0, 1]` where possible."""

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
        iou = _xyxy_iou(teacher_box, student_box)
        score_gap = abs(float(teacher_row["score"]) - float(student_row["score"]))
        pair_score = float(np.clip(iou, 0.0, 1.0)) * float(np.clip(1.0 - score_gap, 0.0, 1.0))
        teacher_student_agreement = max(teacher_student_agreement, pair_score)

    return {
        "score": float(np.clip(float(teacher_row["score"]), 0.0, 1.0)),
        "logit_margin": float(np.clip(margin, 0.0, 1.0)),
        "logit_sharpness": logit_sharpness,
        "decoder_box_stability": float(np.clip(decoder_box_stability, 0.0, 1.0)),
        "decoder_class_stability": float(np.clip(decoder_class_stability, 0.0, 1.0)),
        "teacher_student_agreement": float(np.clip(teacher_student_agreement, 0.0, 1.0)),
    }


def _split_teacher_detections(
    teacher_item: dict[str, Any],
    student_item: dict[str, Any],
    *,
    hard_score_min: float,
    soft_score_min: float,
    soft_score_max: float,
    hard_specs: list[tuple[str, float]],
    soft_specs: list[tuple[str, float]],
    hard_threshold: float,
    soft_threshold: float,
) -> dict[str, Any]:
    """Split teacher query rows within one image into hard/soft/ignored sets."""

    hard_rows: list[dict[str, Any]] = []
    soft_targets: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    teacher_logits = teacher_item["raw_output"]["pred_logits"]
    for teacher_row in teacher_item["query_rows"]:
        score = float(teacher_row["score"])
        signal_values = _routing_signal_values(
            teacher_row,
            student_query_rows=student_item["student_query_rows"],
        )
        if score >= hard_score_min:
            routing_score, routing_parts = _weighted_score(hard_specs, signal_values)
            route = "hard" if routing_score >= hard_threshold else "ignore"
            if route == "hard":
                hard_rows.append(teacher_row)
        elif soft_score_min <= score < soft_score_max:
            routing_score, routing_parts = _weighted_score(soft_specs, signal_values)
            route = "soft" if routing_score >= soft_threshold else "ignore"
            if route == "soft":
                soft_targets.append(
                    {
                        "teacher_row": teacher_row,
                        "teacher_logits": teacher_logits[int(teacher_row["query_index"])].detach().cpu(),
                        "routing_score": float(routing_score),
                        "routing_signals": dict(routing_parts),
                    }
                )
        else:
            routing_score = 0.0
            routing_parts = {}
            route = "ignore"
        diagnostics.append(
            {
                "query_index": int(teacher_row["query_index"]),
                "class_index": int(teacher_row["class_index"]),
                "score": score,
                "route": route,
                "routing_score": float(routing_score),
                "routing_signals": routing_parts,
            }
        )

    return {
        "hard_rows": hard_rows,
        "soft_targets": soft_targets,
        "diagnostics": diagnostics,
    }


def _pseudo_annotations_from_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build hard pseudo annotations from routed query rows in original coordinates."""

    rows = sorted(rows, key=lambda row: row["score"], reverse=True)
    annotations = []
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


def _student_soft_loss(
    soft_items: list[dict[str, Any]],
    *,
    soft_loss_weight: float,
) -> torch.Tensor:
    if not soft_items:
        return torch.tensor(0.0)
    device = soft_items[0]["student_raw"]["pred_logits"].device

    loss_terms = []
    for soft_item in soft_items:
        soft_targets = soft_item["soft_targets"]
        student_raw = soft_item["student_raw"]
        student_query_rows = soft_item["student_query_rows"]
        if not soft_targets or not student_query_rows:
            continue

        matched_losses = []
        for soft_target in soft_targets:
            teacher_row = soft_target["teacher_row"]
            best_student = None
            best_iou = -1.0
            teacher_box = [float(v) for v in teacher_row["bbox"]]
            for student_row in student_query_rows:
                if int(student_row["category_id"]) != int(teacher_row["category_id"]):
                    continue
                student_box = [float(v) for v in student_row["bbox"]]
                tx0, ty0, tx1, ty1 = teacher_box
                sx0, sy0, sx1, sy1 = student_box
                ix0, iy0 = max(tx0, sx0), max(ty0, sy0)
                ix1, iy1 = min(tx1, sx1), min(ty1, sy1)
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                area_t = max(0.0, tx1 - tx0) * max(0.0, ty1 - ty0)
                area_s = max(0.0, sx1 - sx0) * max(0.0, sy1 - sy0)
                union = area_t + area_s - inter
                iou = 0.0 if union <= 0.0 else inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_student = student_row
            if best_student is None:
                continue

            teacher_logits = soft_target["teacher_logits"].to(device)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            student_logits = student_raw["pred_logits"][best_student["query_index"]]
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction="batchmean",
            )
            # Use the soft routing score itself as the adaptive weight:
            # better low-confidence pseudo labels contribute more, while weak
            # or noisy ones are naturally downweighted.
            adaptive_weight = float(np.clip(soft_target["routing_score"], 0.0, 1.0))
            matched_losses.append(kl_loss * adaptive_weight)
        if matched_losses:
            loss_terms.append(torch.stack(matched_losses).mean())
    if not loss_terms:
        return torch.tensor(0.0, device=device)
    return soft_loss_weight * torch.stack(loss_terms).mean()


def _update_ema(teacher_model: torch.nn.Module, student_model: torch.nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


def _evaluate_split(cfg: Any, model: torch.nn.Module, split_name: str, dataset_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    from src.engine.daod_train_source import _evaluate_split as _eval_impl

    return _eval_impl(cfg, model, split_name, dataset_dicts)


class DAODMeanTeacherRoundTrainer:
    """Round trainer for fixed-teacher, trainable-student DAOD updates."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def fit_round(
        self,
        *,
        cfg: Any,
        round_dir: Path,
        state_in: Any,
        plan: Any,
    ) -> dict[str, Any]:
        round_dir.mkdir(parents=True, exist_ok=True)

        student_adapter = build_daod_model(cfg, load_weights=False)
        teacher_adapter = build_daod_model(cfg, load_weights=False)
        student_model = student_adapter.model.to(self.device)
        teacher_model = teacher_adapter.model.to(self.device)
        DetectionCheckpointer(student_model).load(str(state_in.student_checkpoint))
        DetectionCheckpointer(teacher_model).load(str(state_in.teacher_checkpoint))
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        target_train_dicts = materialize_daod_dicts(cfg, "target_train")
        labeled_ids = set(state_in.queried_ids).union(plan.queried_ids)
        labeled_items = [sample for sample in target_train_dicts if sample["sample_id"] in labeled_ids]
        unlabeled_items = [sample for sample in target_train_dicts if sample["sample_id"] not in labeled_ids]

        method_train_cfg = getattr(cfg.method, "train", object())
        method_eval_cfg = getattr(cfg.method, "eval", object())
        labeled_batch_size = int(getattr(method_train_cfg, "labeled_batch_size", 4))
        unlabeled_batch_size = int(getattr(method_train_cfg, "unlabeled_batch_size", 4))
        loader_num_workers = int(getattr(method_train_cfg, "num_workers", 0))
        labeled_loader = DataLoader(
            DAODListDataset(labeled_items),
            batch_size=labeled_batch_size,
            shuffle=bool(labeled_items),
            collate_fn=collate_daod_batch,
            num_workers=loader_num_workers,
        )
        unlabeled_loader = DataLoader(
            DAODListDataset(unlabeled_items),
            batch_size=unlabeled_batch_size,
            shuffle=bool(unlabeled_items),
            collate_fn=collate_daod_batch,
            num_workers=loader_num_workers,
        )
        labeled_iter = cycle_daod_loader(labeled_loader)
        unlabeled_iter = cycle_daod_loader(unlabeled_loader)

        optimizer = torch.optim.AdamW(
            [p for p in student_model.parameters() if p.requires_grad],
            lr=float(getattr(method_train_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(method_train_cfg, "weight_decay", 1e-4)),
        )
        steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader), 1)
        max_epochs = int(getattr(cfg.method, "round_epochs", 1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * steps_per_epoch)

        strong_short_edge = int(getattr(cfg.detector, "min_size_test", 800))
        max_size = int(getattr(cfg.detector, "max_size_test", 1333))
        losses_cfg = getattr(method_train_cfg, "losses", object())
        ema_cfg = getattr(method_train_cfg, "ema", object())
        routing_cfg = getattr(method_train_cfg, "routing", object())
        hard_loss_weight = float(getattr(losses_cfg, "hard_pseudo_weight", 1.0))
        soft_loss_weight = float(getattr(losses_cfg, "soft_distill_weight", 1.0))
        ema_momentum = float(getattr(ema_cfg, "momentum", 0.999))
        ema_update_interval = int(getattr(ema_cfg, "update_interval", 1))
        weak_view_rng = np.random.default_rng(int(getattr(cfg, "seed", 42)) + int(state_in.round_idx))
        hard_routing_cfg = getattr(routing_cfg, "hard", object())
        soft_routing_cfg = getattr(routing_cfg, "soft", object())
        hard_specs = _signal_specs(
            hard_routing_cfg,
            default_specs=[("score", 0.7), ("decoder_box_stability", 0.3)],
        )
        soft_specs = _signal_specs(
            soft_routing_cfg,
            default_specs=[
                ("logit_sharpness", 0.4),
                ("decoder_box_stability", 0.3),
                ("teacher_student_agreement", 0.3),
            ],
        )
        hard_score_min = float(getattr(hard_routing_cfg, "score_min", 0.5))
        soft_score_min = float(getattr(soft_routing_cfg, "score_min", 0.05))
        soft_score_max = float(getattr(soft_routing_cfg, "score_max", hard_score_min))
        hard_threshold = float(getattr(hard_routing_cfg, "threshold", 0.6))
        soft_threshold = float(getattr(soft_routing_cfg, "threshold", 0.3))
        log_period = int(getattr(method_train_cfg, "log_period", 100))
        eval_period = int(getattr(method_train_cfg, "eval_period", 0))
        checkpoint_period = int(getattr(method_train_cfg, "checkpoint_period", 0))

        source_val_dicts = _limit_samples(
            materialize_daod_dicts(cfg, "source_val"),
            getattr(method_eval_cfg, "source_val_limit", 0),
        )
        target_val_dicts = _limit_samples(
            materialize_daod_dicts(cfg, "target_val"),
            getattr(method_eval_cfg, "target_val_limit", 0),
        )
        train_log_path = round_dir / "train_log.jsonl"
        eval_log_path = round_dir / "eval_log.jsonl"
        global_step = 0

        history: list[dict[str, float]] = []
        for epoch_idx in range(1, max_epochs + 1):
            student_model.train()
            epoch_stats = {"loss_total": 0.0, "loss_sup": 0.0, "loss_hard": 0.0, "loss_soft": 0.0, "steps": 0.0}
            for step_idx in range(steps_per_epoch):
                labeled_batch = next(labeled_iter, [])
                unlabeled_batch = next(unlabeled_iter, [])
                if not labeled_batch and not unlabeled_batch:
                    continue

                loss = torch.tensor(0.0, device=self.device)
                loss_sup = torch.tensor(0.0, device=self.device)
                loss_hard = torch.tensor(0.0, device=self.device)
                loss_soft = torch.tensor(0.0, device=self.device)

                if labeled_batch:
                    labeled_inputs = _make_supervised_inputs(
                        student_adapter,
                        labeled_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                    )
                    loss_dict = student_model(labeled_inputs)
                    loss_sup = sum(loss_dict.values())
                    loss = loss + loss_sup

                teacher_items = []
                if unlabeled_batch:
                    teacher_items = _teacher_outputs_for_unlabeled(
                        teacher_adapter,
                        unlabeled_batch,
                        weak_view_rng=weak_view_rng,
                    )
                    student_items = _student_outputs_for_unlabeled(
                        student_adapter,
                        unlabeled_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                    )
                    hard_batch = []
                    soft_batch = []
                    for teacher_item, student_item in zip(teacher_items, student_items):
                        if teacher_item["sample"]["sample_id"] != student_item["sample"]["sample_id"]:
                            raise RuntimeError("Teacher/student unlabeled batch alignment broke.")
                        routed = _split_teacher_detections(
                            teacher_item,
                            student_item,
                            hard_score_min=hard_score_min,
                            soft_score_min=soft_score_min,
                            soft_score_max=soft_score_max,
                            hard_specs=hard_specs,
                            soft_specs=soft_specs,
                            hard_threshold=hard_threshold,
                            soft_threshold=soft_threshold,
                        )
                        if routed["hard_rows"]:
                            pseudo_annotations = _pseudo_annotations_from_rows(routed["hard_rows"])
                            if pseudo_annotations:
                                pseudo_sample = dict(teacher_item["sample"])
                                pseudo_sample["annotations"] = pseudo_annotations
                                hard_batch.append(pseudo_sample)
                        if routed["soft_targets"]:
                            soft_batch.append(
                                {
                                    "sample": teacher_item["sample"],
                                    "soft_targets": routed["soft_targets"],
                                    "student_raw": student_item["student_raw"],
                                    "student_query_rows": student_item["student_query_rows"],
                                }
                            )

                    if hard_batch:
                        hard_inputs = _make_supervised_inputs(
                            student_adapter,
                            hard_batch,
                            strong_short_edge=strong_short_edge,
                            max_size=max_size,
                            device=self.device,
                        )
                        hard_loss_dict = student_model(hard_inputs)
                        loss_hard = hard_loss_weight * sum(hard_loss_dict.values())
                        loss = loss + loss_hard
                    if soft_batch:
                        loss_soft = _student_soft_loss(
                            soft_batch,
                            soft_loss_weight=soft_loss_weight,
                        )
                        loss = loss + loss_soft

                if loss.requires_grad and float(loss.detach().cpu()) > 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if (step_idx + 1) % ema_update_interval == 0:
                        _update_ema(teacher_model, student_model, ema_momentum)
                scheduler.step()
                global_step += 1

                epoch_stats["loss_total"] += float(loss.detach().cpu())
                epoch_stats["loss_sup"] += float(loss_sup.detach().cpu())
                epoch_stats["loss_hard"] += float(loss_hard.detach().cpu())
                epoch_stats["loss_soft"] += float(loss_soft.detach().cpu())
                epoch_stats["steps"] += 1.0

                if log_period > 0 and global_step % log_period == 0:
                    log_payload = {
                        "epoch": int(epoch_idx),
                        "step": int(global_step),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "loss_total": float(loss.detach().cpu()),
                        "loss_sup": float(loss_sup.detach().cpu()),
                        "loss_hard": float(loss_hard.detach().cpu()),
                        "loss_soft": float(loss_soft.detach().cpu()),
                        "num_labeled_images": int(len(labeled_batch)),
                        "num_unlabeled_images": int(len(unlabeled_batch)),
                    }
                    _append_jsonl(train_log_path, log_payload)

                if eval_period > 0 and global_step % eval_period == 0:
                    student_model.eval()
                    source_val_metrics = _evaluate_split(cfg, student_model, "source_val", source_val_dicts)
                    target_val_metrics = _evaluate_split(cfg, student_model, "target_val", target_val_dicts)
                    save_json(round_dir / "source_val_metrics.json", source_val_metrics)
                    save_json(round_dir / "target_val_metrics.json", target_val_metrics)
                    _log_eval_metrics(round_dir, global_step, "source_val", source_val_metrics)
                    _log_eval_metrics(round_dir, global_step, "target_val", target_val_metrics)
                    _append_jsonl(
                        eval_log_path,
                        {
                            "epoch": int(epoch_idx),
                            "step": int(global_step),
                            "source_val_metrics": source_val_metrics,
                            "target_val_metrics": target_val_metrics,
                        },
                    )
                    student_model.train()

                if checkpoint_period > 0 and global_step % checkpoint_period == 0:
                    DetectionCheckpointer(student_model, save_dir=str(round_dir)).save(f"student_step_{global_step}")
                    DetectionCheckpointer(teacher_model, save_dir=str(round_dir)).save(f"teacher_step_{global_step}")

            denom = max(epoch_stats["steps"], 1.0)
            for key in ("loss_total", "loss_sup", "loss_hard", "loss_soft"):
                epoch_stats[key] /= denom
            history.append(epoch_stats)

        student_ckpt = round_dir / "student_last.pth"
        teacher_ckpt = round_dir / "teacher_last.pth"
        DetectionCheckpointer(student_model, save_dir=str(round_dir)).save("student_last")
        DetectionCheckpointer(teacher_model, save_dir=str(round_dir)).save("teacher_last")

        student_model.eval()
        source_val_metrics = _evaluate_split(cfg, student_model, "source_val", source_val_dicts)
        target_val_metrics = _evaluate_split(cfg, student_model, "target_val", target_val_dicts)
        save_json(round_dir / "train_history.json", {"history": history})
        save_json(round_dir / "source_val_metrics.json", source_val_metrics)
        save_json(round_dir / "target_val_metrics.json", target_val_metrics)
        _log_eval_metrics(round_dir, global_step, "source_val", source_val_metrics)
        _log_eval_metrics(round_dir, global_step, "target_val", target_val_metrics)

        return {
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt),
            "train_history": history,
            "source_val_metrics": source_val_metrics,
            "target_val_metrics": target_val_metrics,
        }
