"""DINO-native trainers for isolated SFOD baselines.

The original public LPLD implementation is Faster R-CNN/RPN based. PETS and
LPU do not have local project code here. This trainer keeps the shared protocol
fixed to the repository's DINO detector while implementing DINO-query analogues
of each paper's source-free adaptation logic.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import random
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

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
from src.models import build_daod_model, run_daod_raw_outputs

from .active import build_sparse_target_split
from .pseudo import (
    build_low_confidence_targets,
    consensus_query_rows,
    filter_pseudo_rows,
    lpld_distillation_loss,
    lpu_low_confidence_loss,
    rows_to_annotations,
    signal_specs,
)
from .utils import append_jsonl, maybe_empty_cuda_cache, save_json


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


def _resolve_aux_device(train_cfg: Any, primary_device: torch.device, field_name: str) -> torch.device:
    if primary_device.type != "cuda" or not torch.cuda.is_available():
        return primary_device

    raw_value = getattr(train_cfg, field_name, None)
    if raw_value is None:
        return primary_device
    raw = str(raw_value).strip().lower()
    if raw in {"", "same", "student"}:
        return primary_device
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "auto":
        visible_count = torch.cuda.device_count()
        if visible_count <= 1:
            return primary_device
        primary_index = primary_device.index if primary_device.index is not None else torch.cuda.current_device()
        for idx in range(visible_count):
            if idx != primary_index:
                return torch.device(f"cuda:{idx}")
        return primary_device
    return torch.device(str(raw_value))


def _limit_samples(dataset_dicts: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or int(limit) <= 0:
        return dataset_dicts
    return dataset_dicts[: int(limit)]


def _make_daod_loader(
    items: list[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        DAODListDataset(items),
        batch_size=batch_size,
        shuffle=shuffle and bool(items),
        collate_fn=collate_daod_batch,
        num_workers=num_workers,
    )


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
    adapter: Any,
    batch: list[dict[str, Any]],
    *,
    strong_short_edge: int,
    max_size: int,
    device: torch.device,
    strong_view_rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for sample in batch:
        strong_sample = build_strong_view_sample(sample, rng=strong_view_rng, suffix="sfod_supervised_strong")
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
    teacher_adapter: Any,
    batch: list[dict[str, Any]],
    *,
    weak_view_rng: random.Random,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for sample in batch:
        image = Image.open(sample["file_name"]).convert("RGB")
        weak_sample = build_weak_view_sample(
            {**sample, "image": image},
            rng=weak_view_rng,
            suffix="sfod_teacher_weak",
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
        outputs.append({"sample": sample, "raw_output": raw_output, "query_rows": query_rows})
    return outputs


def _student_outputs_for_unlabeled(
    student_adapter: Any,
    batch: list[dict[str, Any]],
    *,
    strong_short_edge: int,
    max_size: int,
    strong_view_rng: random.Random,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for sample in batch:
        strong_sample = build_strong_view_sample(sample, rng=strong_view_rng, suffix="sfod_student_strong")
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
        outputs.append({"sample": sample, "student_raw": student_raw, "student_query_rows": student_query_rows})
    return outputs


def _set_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)


def _update_ema(teacher_model: torch.nn.Module, student_model: torch.nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            student_value = student_param.data
            if student_value.device != teacher_param.data.device:
                student_value = student_value.to(teacher_param.data.device)
            teacher_param.data.mul_(momentum).add_(student_value, alpha=1.0 - float(momentum))


def _clone_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _swap_model_weights(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state = _clone_state_cpu(left)
    right_state = _clone_state_cpu(right)
    left.load_state_dict(right_state, strict=True)
    right.load_state_dict(left_state, strict=True)


def _sum_loss_dict(loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(loss_dict.values())


def _evaluate_split(cfg: Any, model: torch.nn.Module, split_name: str, dataset_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    from src.engine.daod_train_source import _evaluate_split as _eval_impl

    if not dataset_dicts:
        return {}
    return _eval_impl(cfg, model, split_name, dataset_dicts)


class SFODBaselineTrainer:
    """Single DINO trainer shared by LPLD, PETS, and LPU baseline packages."""

    def __init__(self, cfg: Any, device: torch.device, *, algorithm: str, log_prefix: str) -> None:
        self.cfg = cfg
        self.device = device
        self.algorithm = algorithm
        self.log_prefix = log_prefix

    def _build_optimizer(self, model: torch.nn.Module, train_cfg: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(getattr(train_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(train_cfg, "weight_decay", 1e-4)),
        )

    def _build_pets_consensus_items(
        self,
        dynamic_items: list[dict[str, Any]],
        static_items: list[dict[str, Any]],
        *,
        pseudo_cfg: Any,
        pets_cfg: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        consensus_items: list[dict[str, Any]] = []
        stats = {"dynamic_rows": 0, "static_rows": 0, "consensus_rows": 0}
        candidate_threshold = float(getattr(pets_cfg, "consensus_candidate_threshold", 0.2))
        for dynamic_item, static_item in zip(dynamic_items, static_items):
            dynamic_rows = [
                row for row in dynamic_item["query_rows"] if float(row.get("score", 0.0)) >= candidate_threshold
            ]
            static_rows = [
                row for row in static_item["query_rows"] if float(row.get("score", 0.0)) >= candidate_threshold
            ]
            rows = consensus_query_rows(
                dynamic_rows,
                static_rows,
                consensus_iou=float(getattr(pets_cfg, "consensus_iou", 0.5)),
                include_single_teacher=bool(getattr(pets_cfg, "include_single_teacher", True)),
                single_teacher_threshold=float(getattr(pets_cfg, "single_teacher_threshold", 0.55)),
                score_merge=str(getattr(pets_cfg, "score_merge", "mean")).strip().lower(),
                dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
            )
            stats["dynamic_rows"] += len(dynamic_rows)
            stats["static_rows"] += len(static_rows)
            stats["consensus_rows"] += len(rows)
            consensus_items.append(
                {
                    "sample": dynamic_item["sample"],
                    "raw_output": dynamic_item["raw_output"],
                    "query_rows": rows,
                }
            )
        return consensus_items, stats

    def fit(self, *, run_dir: Path, source_checkpoint: str) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        method_cfg = getattr(self.cfg, "method", object())
        train_cfg = getattr(method_cfg, "train", object())
        pseudo_cfg = getattr(method_cfg, "pseudo", object())
        active_cfg = getattr(method_cfg, "active", object())
        eval_cfg = getattr(method_cfg, "eval", object())
        seed = int(getattr(self.cfg, "seed", 42))

        teacher_device = _resolve_aux_device(train_cfg, self.device, "teacher_device")
        student_adapter = build_daod_model(self.cfg, load_weights=False, device=self.device)
        student_model = student_adapter.model.to(self.device)
        DetectionCheckpointer(student_model).load(str(source_checkpoint))
        _set_trainable(student_model, True)

        teacher_adapter = None
        teacher_model = None
        static_teacher_adapter = None
        static_teacher_model = None
        dynamic_teacher_adapter = None
        dynamic_teacher_model = None

        if self.algorithm == "pets":
            static_device = _resolve_aux_device(train_cfg, self.device, "static_teacher_device")
            dynamic_device = _resolve_aux_device(train_cfg, self.device, "dynamic_teacher_device")
            static_teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=static_device)
            dynamic_teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=dynamic_device)
            static_teacher_model = static_teacher_adapter.model.to(static_device)
            dynamic_teacher_model = dynamic_teacher_adapter.model.to(dynamic_device)
            DetectionCheckpointer(static_teacher_model).load(str(source_checkpoint))
            DetectionCheckpointer(dynamic_teacher_model).load(str(source_checkpoint))
            _set_trainable(static_teacher_model, False)
            _set_trainable(dynamic_teacher_model, False)
            static_teacher_model.eval()
            dynamic_teacher_model.eval()
        else:
            teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=teacher_device)
            teacher_model = teacher_adapter.model.to(teacher_device)
            DetectionCheckpointer(teacher_model).load(str(source_checkpoint))
            _set_trainable(teacher_model, False)
            teacher_model.eval()

        target_train = materialize_daod_dicts(self.cfg, "target_train")
        target_train = _limit_samples(target_train, int(getattr(train_cfg, "max_target_samples", 0)))
        target_labeled, target_unlabeled, selected_ids, active_plan = build_sparse_target_split(
            target_train,
            active_cfg,
            seed=seed,
        )
        if bool(active_plan.get("enabled", False)):
            save_json(run_dir / "active_plan.json", active_plan)
            save_json(
                run_dir / "selected_target_ids.json",
                {
                    "selected_ids": list(active_plan.get("selected_ids", [])),
                    "selected_count": len(selected_ids),
                    "target_total": len(target_train),
                },
            )

        batch_size = int(getattr(train_cfg, "batch_size", 1))
        num_workers = int(getattr(train_cfg, "num_workers", 0))
        target_loader = _make_daod_loader(
            target_unlabeled,
            batch_size=batch_size,
            shuffle=bool(target_unlabeled),
            num_workers=num_workers,
        )
        labeled_loader = _make_daod_loader(
            target_labeled,
            batch_size=batch_size,
            shuffle=bool(target_labeled),
            num_workers=num_workers,
        )
        target_iter = cycle_daod_loader(target_loader)
        labeled_iter = cycle_daod_loader(labeled_loader)

        optimizer = self._build_optimizer(student_model, train_cfg)
        epochs = int(getattr(method_cfg, "epochs", 2))
        steps_per_epoch = max(len(target_loader), len(labeled_loader), 1)
        max_steps_per_epoch = int(getattr(train_cfg, "max_steps_per_epoch", 0))
        if max_steps_per_epoch > 0:
            steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
        total_steps = max(epochs * steps_per_epoch, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        strong_short_edge = int(getattr(self.cfg.detector, "min_size_test", 800))
        max_size = int(getattr(self.cfg.detector, "max_size_test", 1333))
        threshold = float(getattr(pseudo_cfg, "threshold", 0.4))
        dedup_iou_thresh = float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7))
        pseudo_weight = float(getattr(pseudo_cfg, "weight", 1.0))
        supervised_weight = float(getattr(active_cfg, "supervised_weight", 1.0))
        ema_cfg = getattr(method_cfg, "ema", object())
        ema_momentum = float(getattr(ema_cfg, "momentum", 0.999))
        ema_update_interval = int(getattr(ema_cfg, "update_interval", 1))
        log_period = int(getattr(train_cfg, "log_period", 100))
        checkpoint_period = int(getattr(train_cfg, "checkpoint_period", 0))
        train_log_path = run_dir / "train_log.jsonl"
        weak_rng = random.Random(seed + 17)
        strong_rng = random.Random(seed + 31)

        lpld_cfg = getattr(method_cfg, "lpld", object())
        lpu_cfg = getattr(method_cfg, "lpu", object())
        pets_cfg = getattr(method_cfg, "pets", object())
        pets_exchange_period = int(getattr(pets_cfg, "exchange_period_steps", 1000))
        pets_ema_momentum = float(getattr(pets_cfg, "dynamic_ema_momentum", ema_momentum))

        print(
            f"[{self.log_prefix}][train] "
            f"algorithm={self.algorithm} epochs={epochs} target_train={len(target_train)} "
            f"labeled_target={len(target_labeled)} unlabeled_target={len(target_unlabeled)} "
            f"student_device={self.device} teacher_device={teacher_device} source_ckpt={source_checkpoint}"
        )

        global_step = 0
        exchange_count = 0
        history: list[dict[str, Any]] = []
        for epoch_idx in range(1, epochs + 1):
            student_model.train()
            if teacher_model is not None:
                teacher_model.eval()
            if static_teacher_model is not None:
                static_teacher_model.eval()
            if dynamic_teacher_model is not None:
                dynamic_teacher_model.eval()

            epoch_stats = {
                "loss_total": 0.0,
                "loss_pseudo": 0.0,
                "loss_low": 0.0,
                "loss_supervised": 0.0,
                "pseudo_boxes": 0,
                "low_targets": 0,
                "matched_low_targets": 0,
                "pets_consensus_rows": 0,
            }

            for _ in range(steps_per_epoch):
                batch = next(target_iter, [])
                labeled_batch = next(labeled_iter, [])
                pseudo_batch: list[dict[str, Any]] = []
                low_items: list[dict[str, Any]] = []
                pseudo_box_count = 0
                low_stats = {"low_targets": 0, "matched_targets": 0, "pst_pairs": 0, "lscl_pairs": 0}
                pets_consensus_stats = {"consensus_rows": 0}

                if batch:
                    if self.algorithm == "pets":
                        assert dynamic_teacher_adapter is not None
                        assert static_teacher_adapter is not None
                        with torch.no_grad():
                            dynamic_items = _teacher_outputs_for_unlabeled(
                                dynamic_teacher_adapter,
                                batch,
                                weak_view_rng=weak_rng,
                            )
                            static_items = _teacher_outputs_for_unlabeled(
                                static_teacher_adapter,
                                batch,
                                weak_view_rng=weak_rng,
                            )
                        teacher_items, pets_consensus_stats = self._build_pets_consensus_items(
                            dynamic_items,
                            static_items,
                            pseudo_cfg=pseudo_cfg,
                            pets_cfg=pets_cfg,
                        )
                    else:
                        assert teacher_adapter is not None
                        with torch.no_grad():
                            teacher_items = _teacher_outputs_for_unlabeled(
                                teacher_adapter,
                                batch,
                                weak_view_rng=weak_rng,
                            )

                    need_low_branch = self.algorithm in {"lpld", "lpu"}
                    student_items = (
                        _student_outputs_for_unlabeled(
                            student_adapter,
                            batch,
                            strong_short_edge=strong_short_edge,
                            max_size=max_size,
                            strong_view_rng=strong_rng,
                        )
                        if need_low_branch
                        else []
                    )

                    for item_idx, teacher_item in enumerate(teacher_items):
                        hard_rows = filter_pseudo_rows(
                            teacher_item["query_rows"],
                            threshold=threshold,
                            dedup_iou_thresh=dedup_iou_thresh,
                        )
                        annotations = rows_to_annotations(hard_rows)
                        if annotations:
                            pseudo_sample = dict(teacher_item["sample"])
                            pseudo_sample["annotations"] = annotations
                            pseudo_batch.append(pseudo_sample)
                            pseudo_box_count += len(annotations)

                        if need_low_branch:
                            student_item = student_items[item_idx]
                            low_cfg = lpld_cfg if self.algorithm == "lpld" else lpu_cfg
                            default_specs = [
                                ("score", 0.35),
                                ("logit_sharpness", 0.25),
                                ("decoder_box_stability", 0.20),
                                ("teacher_student_agreement", 0.20),
                            ]
                            low_targets = build_low_confidence_targets(
                                teacher_item,
                                student_item,
                                hard_rows=hard_rows,
                                score_min=float(getattr(low_cfg, "low_score_min", 0.05)),
                                score_max=float(getattr(low_cfg, "high_score_min", threshold)),
                                routing_specs=signal_specs(low_cfg, default_specs),
                                routing_threshold=float(getattr(low_cfg, "routing_threshold", 0.0)),
                                hard_exclusion_iou_max=float(getattr(low_cfg, "hard_exclusion_iou_max", 0.4)),
                                pre_routing_topk=int(getattr(low_cfg, "pre_routing_topk", 256)),
                                max_targets=int(getattr(low_cfg, "max_low_targets_per_image", 128)),
                            )
                            if low_targets:
                                low_items.append(
                                    {
                                        "low_targets": low_targets,
                                        "student_raw": student_item["student_raw"],
                                        "student_query_rows": student_item["student_query_rows"],
                                    }
                                )

                loss_terms: list[torch.Tensor] = []
                loss_pseudo_value = 0.0
                loss_low_value = 0.0
                loss_supervised_value = 0.0

                if pseudo_batch:
                    pseudo_inputs = _make_supervised_inputs(
                        student_adapter,
                        pseudo_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                        strong_view_rng=strong_rng,
                    )
                    loss_pseudo = pseudo_weight * _sum_loss_dict(student_model(pseudo_inputs))
                    loss_terms.append(loss_pseudo)
                    loss_pseudo_value = float(loss_pseudo.detach().cpu())

                if low_items:
                    if self.algorithm == "lpld":
                        loss_low, low_stats = lpld_distillation_loss(
                            low_items,
                            weight=float(getattr(lpld_cfg, "soft_distill_weight", 1.0)),
                            match_iou_min=float(getattr(lpld_cfg, "match_iou_min", 0.3)),
                            device=self.device,
                        )
                    else:
                        loss_low, low_stats = lpu_low_confidence_loss(
                            low_items,
                            pst_weight=float(getattr(lpu_cfg, "pst_weight", 1.0)),
                            lscl_weight=float(getattr(lpu_cfg, "lscl_weight", 0.1)),
                            match_iou_min=float(getattr(lpu_cfg, "match_iou_min", 0.3)),
                            positive_iou=float(getattr(lpu_cfg, "positive_iou", 0.5)),
                            negative_iou=float(getattr(lpu_cfg, "negative_iou", 0.1)),
                            contrastive_margin=float(getattr(lpu_cfg, "contrastive_margin", 0.2)),
                            device=self.device,
                        )
                    if float(loss_low.detach().cpu()) != 0.0:
                        loss_terms.append(loss_low)
                    loss_low_value = float(loss_low.detach().cpu())

                if labeled_batch:
                    supervised_inputs = _make_supervised_inputs(
                        student_adapter,
                        labeled_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                        strong_view_rng=strong_rng,
                    )
                    loss_supervised = supervised_weight * _sum_loss_dict(student_model(supervised_inputs))
                    loss_terms.append(loss_supervised)
                    loss_supervised_value = float(loss_supervised.detach().cpu())

                if not loss_terms:
                    global_step += 1
                    continue

                loss = sum(loss_terms)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(getattr(train_cfg, "clip_max_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        float(getattr(train_cfg, "clip_max_norm", 0.0)),
                    )
                optimizer.step()
                scheduler.step()
                global_step += 1

                if self.algorithm == "pets":
                    assert static_teacher_model is not None
                    assert dynamic_teacher_model is not None
                    if pets_exchange_period > 0 and global_step % pets_exchange_period == 0:
                        _swap_model_weights(student_model, static_teacher_model)
                        exchange_count += 1
                    _update_ema(dynamic_teacher_model, student_model, pets_ema_momentum)
                elif teacher_model is not None and ema_update_interval > 0 and global_step % ema_update_interval == 0:
                    _update_ema(teacher_model, student_model, ema_momentum)

                epoch_stats["loss_total"] += float(loss.detach().cpu())
                epoch_stats["loss_pseudo"] += loss_pseudo_value
                epoch_stats["loss_low"] += loss_low_value
                epoch_stats["loss_supervised"] += loss_supervised_value
                epoch_stats["pseudo_boxes"] += int(pseudo_box_count)
                epoch_stats["low_targets"] += int(low_stats.get("low_targets", 0))
                epoch_stats["matched_low_targets"] += int(low_stats.get("matched_targets", 0))
                epoch_stats["pets_consensus_rows"] += int(pets_consensus_stats.get("consensus_rows", 0))

                if log_period > 0 and global_step % log_period == 0:
                    append_jsonl(
                        train_log_path,
                        {
                            "epoch": int(epoch_idx),
                            "step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_total": float(loss.detach().cpu()),
                            "loss_pseudo": loss_pseudo_value,
                            "loss_low": loss_low_value,
                            "loss_supervised": loss_supervised_value,
                            "pseudo_box_count": int(pseudo_box_count),
                            "low_stats": low_stats,
                            "pets_consensus": pets_consensus_stats,
                            "pets_exchanges": int(exchange_count),
                        },
                    )

                if checkpoint_period > 0 and global_step % checkpoint_period == 0:
                    DetectionCheckpointer(student_model, save_dir=str(run_dir)).save(f"student_step_{global_step}")

            denom = max(steps_per_epoch, 1)
            epoch_summary = {
                "epoch": int(epoch_idx),
                "loss_total": float(epoch_stats["loss_total"] / denom),
                "loss_pseudo": float(epoch_stats["loss_pseudo"] / denom),
                "loss_low": float(epoch_stats["loss_low"] / denom),
                "loss_supervised": float(epoch_stats["loss_supervised"] / denom),
                "pseudo_boxes": int(epoch_stats["pseudo_boxes"]),
                "low_targets": int(epoch_stats["low_targets"]),
                "matched_low_targets": int(epoch_stats["matched_low_targets"]),
                "pets_consensus_rows": int(epoch_stats["pets_consensus_rows"]),
                "pets_exchanges": int(exchange_count),
            }
            history.append(epoch_summary)
            append_jsonl(run_dir / "epoch_log.jsonl", epoch_summary)

        student_ckpt = run_dir / "student_last.pth"
        DetectionCheckpointer(student_model, save_dir=str(run_dir)).save("student_last")
        teacher_ckpt = None
        dynamic_teacher_ckpt = None
        static_teacher_ckpt = None
        if teacher_model is not None:
            teacher_ckpt = run_dir / "teacher_last.pth"
            DetectionCheckpointer(teacher_model, save_dir=str(run_dir)).save("teacher_last")
        if dynamic_teacher_model is not None:
            dynamic_teacher_ckpt = run_dir / "dynamic_teacher_last.pth"
            DetectionCheckpointer(dynamic_teacher_model, save_dir=str(run_dir)).save("dynamic_teacher_last")
        if static_teacher_model is not None:
            static_teacher_ckpt = run_dir / "static_teacher_last.pth"
            DetectionCheckpointer(static_teacher_model, save_dir=str(run_dir)).save("static_teacher_last")

        target_val = _limit_samples(
            materialize_daod_dicts(self.cfg, "target_val"),
            int(getattr(eval_cfg, "target_val_limit", 0)),
        )
        evaluate_teacher = bool(getattr(eval_cfg, "evaluate_teacher", False))
        final_model_name = str(getattr(eval_cfg, "final_model", "student")).strip().lower()
        student_model.eval()
        print(f"[{self.log_prefix}][eval] step={global_step} split=target_val student")
        student_target_metrics = _evaluate_split(self.cfg, student_model, "target_val", target_val)
        teacher_target_metrics: dict[str, Any] = {}
        dynamic_teacher_target_metrics: dict[str, Any] = {}
        static_teacher_target_metrics: dict[str, Any] = {}
        teacher_eval_error: str | None = None

        if evaluate_teacher:
            try:
                if teacher_model is not None:
                    teacher_model.eval()
                    with _device_context(teacher_device):
                        print(f"[{self.log_prefix}][eval] step={global_step} split=target_val teacher")
                        teacher_target_metrics = _evaluate_split(self.cfg, teacher_model, "target_val", target_val)
                if dynamic_teacher_model is not None:
                    dynamic_teacher_model.eval()
                    print(f"[{self.log_prefix}][eval] step={global_step} split=target_val dynamic_teacher")
                    dynamic_teacher_target_metrics = _evaluate_split(
                        self.cfg,
                        dynamic_teacher_model,
                        "target_val",
                        target_val,
                    )
                if static_teacher_model is not None:
                    static_teacher_model.eval()
                    print(f"[{self.log_prefix}][eval] step={global_step} split=target_val static_teacher")
                    static_teacher_target_metrics = _evaluate_split(
                        self.cfg,
                        static_teacher_model,
                        "target_val",
                        target_val,
                    )
            except RuntimeError as exc:
                teacher_eval_error = str(exc)
                print(f"[{self.log_prefix}][eval][warning] teacher eval failed; keeping student metrics: {exc}")

        final_checkpoint: Path = student_ckpt
        final_target_metrics = student_target_metrics
        if final_model_name == "teacher" and teacher_target_metrics and teacher_ckpt is not None:
            final_checkpoint = teacher_ckpt
            final_target_metrics = teacher_target_metrics
        elif final_model_name == "dynamic_teacher" and dynamic_teacher_target_metrics and dynamic_teacher_ckpt is not None:
            final_checkpoint = dynamic_teacher_ckpt
            final_target_metrics = dynamic_teacher_target_metrics
        elif final_model_name == "static_teacher" and static_teacher_target_metrics and static_teacher_ckpt is not None:
            final_checkpoint = static_teacher_ckpt
            final_target_metrics = static_teacher_target_metrics
        else:
            final_model_name = "student"

        save_json(run_dir / "target_val_metrics.json", final_target_metrics)
        save_json(run_dir / "student_target_val_metrics.json", student_target_metrics)
        if teacher_target_metrics:
            save_json(run_dir / "teacher_target_val_metrics.json", teacher_target_metrics)
        if dynamic_teacher_target_metrics:
            save_json(run_dir / "dynamic_teacher_target_val_metrics.json", dynamic_teacher_target_metrics)
        if static_teacher_target_metrics:
            save_json(run_dir / "static_teacher_target_val_metrics.json", static_teacher_target_metrics)

        summary = {
            "algorithm": self.algorithm,
            "epochs": int(epochs),
            "global_step": int(global_step),
            "source_checkpoint": str(source_checkpoint),
            "final_model": final_model_name,
            "final_checkpoint": str(final_checkpoint),
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt) if teacher_ckpt is not None else None,
            "dynamic_teacher_checkpoint": str(dynamic_teacher_ckpt) if dynamic_teacher_ckpt is not None else None,
            "static_teacher_checkpoint": str(static_teacher_ckpt) if static_teacher_ckpt is not None else None,
            "history": history,
            "active_plan": active_plan,
            "student_target_val_metrics": student_target_metrics,
            "teacher_target_val_metrics": teacher_target_metrics,
            "dynamic_teacher_target_val_metrics": dynamic_teacher_target_metrics,
            "static_teacher_target_val_metrics": static_teacher_target_metrics,
            "teacher_eval_error": teacher_eval_error,
            "final_target_val_metrics": final_target_metrics,
            "pets_exchanges": int(exchange_count),
        }
        save_json(run_dir / "summary.json", summary)
        maybe_empty_cuda_cache()
        return summary
