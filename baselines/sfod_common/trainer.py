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
from src.engine.daod_gradient_surgery import (
    PCGradStats,
    add_grads_in_place,
    assign_grads,
    clone_grad_list,
    target_anchored_pcgrad,
)
from src.engine.daod_label_guided import build_label_guided_hook, label_guided_hook_requires_teacher_fit
from src.engine.daod_query_recovery import (
    QueryRecoveryScorer,
    QueryRecoverySelectionStats,
    fit_query_recovery_scorer,
    merge_multiview_teacher_items,
)
from src.engine.daod_query_revival import QueryRevivalLossStats, query_revival_loss
from src.engine.daod_round_trainer import _update_aema as _update_aema_weighted
from src.engine.daod_teacher_guidance import collect_grad_importance, importance_map_stats, merge_importance_maps
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


def _class_loss_from_dict(loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    selected = [value for key, value in loss_dict.items() if "class" in key or "ce" in key]
    if not selected:
        selected = list(loss_dict.values())
    return sum(selected)


def _importance_loss_from_dict(loss_dict: dict[str, torch.Tensor], *, mode: str) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "class":
        return _class_loss_from_dict(loss_dict)
    if mode == "full":
        return _sum_loss_dict(loss_dict)
    raise ValueError(f"Unsupported label-guided teacher-update loss mode: {mode!r}")


def _teacher_grad_importance_from_supervised_batch(
    *,
    teacher_model: torch.nn.Module,
    teacher_adapter: Any,
    batch: list[dict[str, Any]],
    teacher_device: torch.device,
    strong_short_edge: int,
    max_size: int,
    loss_mode: str,
    loss_weight: float = 1.0,
) -> tuple[dict[str, torch.Tensor], float]:
    if not batch:
        return {}, 0.0

    with _device_context(teacher_device):
        teacher_model.train()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(True)
        teacher_inputs = _make_supervised_inputs(
            teacher_adapter,
            batch,
            strong_short_edge=strong_short_edge,
            max_size=max_size,
            device=teacher_device,
        )
        loss = float(loss_weight) * _importance_loss_from_dict(
            teacher_model(teacher_inputs),
            mode=loss_mode,
        )
        teacher_model.zero_grad(set_to_none=True)
        loss.backward()
        importance = collect_grad_importance(teacher_model)
        loss_value = float(loss.detach().cpu())
        teacher_model.zero_grad(set_to_none=True)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    return importance, loss_value


def _loss_grads(
    loss: torch.Tensor | None,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> list[torch.Tensor | None]:
    if loss is None or not loss.requires_grad:
        return [None for _ in parameters]
    return list(
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def _gradient_surgery_log_fields(
    *,
    method: str,
    pseudo_stats: PCGradStats | None,
) -> dict[str, Any]:
    return {
        "grad_surgery_method": method,
        "grad_cos_pseudo_before": pseudo_stats.cosine_before if pseudo_stats is not None else None,
        "grad_cos_pseudo_after": pseudo_stats.cosine_after if pseudo_stats is not None else None,
        "grad_adjusted_pseudo": bool(pseudo_stats.projected) if pseudo_stats is not None else None,
        "grad_weight_pseudo": pseudo_stats.weight if pseudo_stats is not None else None,
    }


def _new_gradient_surgery_epoch_stats() -> dict[str, float]:
    return {
        "steps": 0.0,
        "pseudo_steps": 0.0,
        "pseudo_adjusted": 0.0,
        "pseudo_cos_before_sum": 0.0,
        "pseudo_cos_before_count": 0.0,
        "pseudo_cos_after_sum": 0.0,
        "pseudo_cos_after_count": 0.0,
        "pseudo_weight_sum": 0.0,
        "pseudo_weight_count": 0.0,
    }


def _accumulate_gradient_surgery_stats(
    epoch_stats: dict[str, float],
    stats: PCGradStats | None,
) -> None:
    if stats is None:
        return
    epoch_stats["pseudo_steps"] += 1.0
    if stats.projected:
        epoch_stats["pseudo_adjusted"] += 1.0
    if stats.cosine_before is not None:
        epoch_stats["pseudo_cos_before_sum"] += float(stats.cosine_before)
        epoch_stats["pseudo_cos_before_count"] += 1.0
    if stats.cosine_after is not None:
        epoch_stats["pseudo_cos_after_sum"] += float(stats.cosine_after)
        epoch_stats["pseudo_cos_after_count"] += 1.0
    if stats.weight is not None:
        epoch_stats["pseudo_weight_sum"] += float(stats.weight)
        epoch_stats["pseudo_weight_count"] += 1.0


def _gradient_surgery_epoch_summary(
    epoch_stats: dict[str, float],
    *,
    enabled: bool,
    method: str,
) -> dict[str, Any]:
    def _mean(sum_key: str, count_key: str) -> float | None:
        count = float(epoch_stats[count_key])
        if count <= 0.0:
            return None
        return float(epoch_stats[sum_key] / count)

    pseudo_steps = float(epoch_stats["pseudo_steps"])
    return {
        "enabled": bool(enabled),
        "method": method if enabled else None,
        "steps": int(epoch_stats["steps"]),
        "pseudo_adjustment_rate": float(epoch_stats["pseudo_adjusted"] / pseudo_steps)
        if pseudo_steps > 0.0
        else None,
        "mean_cos_pseudo_before": _mean("pseudo_cos_before_sum", "pseudo_cos_before_count"),
        "mean_cos_pseudo_after": _mean("pseudo_cos_after_sum", "pseudo_cos_after_count"),
        "mean_pseudo_weight": _mean("pseudo_weight_sum", "pseudo_weight_count"),
    }


def _new_query_revival_epoch_stats() -> QueryRevivalLossStats:
    return QueryRevivalLossStats()


def _query_revival_log_fields(
    *,
    enabled: bool,
    loss_stats: QueryRevivalLossStats | None,
) -> dict[str, Any]:
    if not enabled or loss_stats is None:
        return {
            "query_revival_enabled": bool(enabled),
            "query_revival_targets": None,
            "query_revival_matched": None,
            "query_revival_mean_score": None,
            "query_revival_mean_gate": None,
            "query_revival_mean_weight": None,
            "query_revival_mean_match_iou": None,
        }
    loss_dict = loss_stats.as_dict()
    return {
        "query_revival_enabled": True,
        "query_revival_targets": int(loss_dict.get("targets", 0)),
        "query_revival_matched": int(loss_dict.get("matched", 0)),
        "query_revival_mean_score": loss_dict.get("mean_score"),
        "query_revival_mean_gate": loss_dict.get("mean_gate"),
        "query_revival_mean_weight": loss_dict.get("mean_weight"),
        "query_revival_mean_match_iou": loss_dict.get("mean_match_iou"),
    }


def _query_revival_epoch_summary(
    epoch_stats: QueryRevivalLossStats,
    *,
    enabled: bool,
    loss_weight: float,
    train_as: str,
    match_mode: str,
    foreground_pool: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "loss_weight": float(loss_weight) if enabled else None,
        "train_as": train_as if enabled else None,
        "match_mode": match_mode if enabled else None,
        "foreground_pool": foreground_pool if enabled else None,
        **epoch_stats.as_dict(),
    }


def _query_recovery_num_views(recovery_cfg: Any) -> int:
    if not bool(getattr(recovery_cfg, "enabled", False)):
        return 1
    multi_view_cfg = getattr(recovery_cfg, "multi_view", object())
    if not bool(getattr(multi_view_cfg, "enabled", False)):
        return 1
    return max(1, int(getattr(multi_view_cfg, "views", 2)))


def _build_query_recovery_teacher_items(
    *,
    teacher_adapter: Any,
    primary_items: list[dict[str, Any]],
    source_batch: list[dict[str, Any]],
    recovery_cfg: Any,
    teacher_device: torch.device,
    seed: int,
    step_offset: int,
) -> list[dict[str, Any]]:
    num_views = _query_recovery_num_views(recovery_cfg)
    support_iou = float(getattr(getattr(recovery_cfg, "multi_view", object()), "support_iou", 0.5))
    extra_views: list[list[dict[str, Any]]] = []
    for view_idx in range(1, num_views):
        with _device_context(teacher_device), torch.no_grad():
            extra_views.append(
                _teacher_outputs_for_unlabeled(
                    teacher_adapter,
                    source_batch,
                    weak_view_rng=random.Random(int(seed) + int(step_offset) * 1009 + view_idx * 9173),
                )
            )
    return merge_multiview_teacher_items(
        primary_items,
        extra_views,
        support_iou=support_iou,
    )


def _evaluate_split(cfg: Any, model: torch.nn.Module, split_name: str, dataset_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    from src.engine.daod_train_source import _evaluate_split as _eval_impl

    if not dataset_dicts:
        return {}
    return _eval_impl(cfg, model, split_name, dataset_dicts)


def _bbox_ap50(metrics: dict[str, Any]) -> float | None:
    bbox = metrics.get("bbox", {}) if isinstance(metrics, dict) else {}
    value = bbox.get("AP50")
    if value is None:
        return None
    return float(value)


def _compact_intermediate_eval_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep summaries small; full evaluator outputs are saved separately."""
    return {key: value for key, value in record.items() if key != "metrics"}


def _student_eval_patience_update(
    *,
    ap50: float,
    previous_ap50: float | None,
    best_ap50: float | None,
    no_improve_count: int,
    consecutive_drop_count: int,
    min_delta: float,
    mode: str,
    patience: int,
) -> dict[str, Any]:
    improved = best_ap50 is None or ap50 > float(best_ap50) + float(min_delta)
    if previous_ap50 is not None and ap50 < float(previous_ap50) - float(min_delta):
        consecutive_drop_count += 1
    else:
        consecutive_drop_count = 0
    if improved:
        no_improve_count = 0
    else:
        no_improve_count += 1
    stop_count = consecutive_drop_count if mode == "consecutive_drop" else no_improve_count
    return {
        "improved": bool(improved),
        "previous_ap50": float(ap50),
        "consecutive_drop_count": int(consecutive_drop_count),
        "no_improve_count": int(no_improve_count),
        "stop_count": int(stop_count),
        "should_stop": bool(patience > 0 and stop_count >= int(patience)),
    }


def _run_intermediate_target_eval(
    *,
    cfg: Any,
    run_dir: Path,
    model: torch.nn.Module,
    log_prefix: str,
    exp_name: str,
    split_items: list[dict[str, Any]],
    device: torch.device,
    epoch: int,
    step: int,
    target_val_limit: int,
) -> dict[str, Any]:
    model.eval()
    with _device_context(device):
        print(
            f"[{log_prefix}][intermediate_eval] "
            f"exp={exp_name} epoch={epoch} step={step} split=target_val model=student",
            flush=True,
        )
        metrics = _evaluate_split(cfg, model, "target_val", split_items)
    ap50 = _bbox_ap50(metrics)
    metrics_path = f"intermediate_target_val_student_step{step}.json"
    save_json(run_dir / metrics_path, metrics)
    if ap50 is not None:
        print(
            f"[{log_prefix}][intermediate_eval] "
            f"exp={exp_name} epoch={epoch} step={step} model=student AP50={ap50:.3f}",
            flush=True,
        )
    return {
        "exp_name": exp_name,
        "epoch": int(epoch),
        "step": int(step),
        "model": "student",
        "target_val_limit": int(target_val_limit),
        "AP50": ap50,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


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
        label_guided_hook = build_label_guided_hook(method_cfg)
        label_guided_summary = label_guided_hook.state.component_summary
        save_json(run_dir / "label_guided_components.json", label_guided_summary)
        save_json(run_dir / "label_guided_hook_state.json", label_guided_hook.state.as_dict())
        train_cfg = getattr(method_cfg, "train", object())
        pseudo_cfg = getattr(method_cfg, "pseudo", object())
        active_cfg = getattr(method_cfg, "active", object())
        eval_cfg = getattr(method_cfg, "eval", object())
        seed = int(getattr(self.cfg, "seed", 42))
        exp_name = str(getattr(method_cfg, "exp_name", f"{self.algorithm}_daod")).strip() or f"{self.algorithm}_daod"

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
        static_device = None
        dynamic_device = None

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
        num_classes = int(self.cfg.data.num_classes)
        threshold = float(getattr(pseudo_cfg, "threshold", 0.4))
        base_thresholds = [threshold] * num_classes
        dedup_iou_thresh = float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7))
        pseudo_weight = float(getattr(pseudo_cfg, "weight", 1.0))
        supervised_weight = float(getattr(active_cfg, "supervised_weight", 1.0))
        ema_cfg = getattr(method_cfg, "ema", object())
        ema_momentum = float(getattr(ema_cfg, "momentum", 0.999))
        ema_update_interval = int(getattr(ema_cfg, "update_interval", 1))
        gradient_surgery_cfg = getattr(method_cfg, "gradient_surgery", object())
        gradient_surgery_enabled = bool(getattr(gradient_surgery_cfg, "enabled", False))
        gradient_surgery_method = str(
            getattr(gradient_surgery_cfg, "method", "target_anchored_pcgrad")
        ).strip().lower()
        gradient_surgery_apply_pseudo = bool(getattr(gradient_surgery_cfg, "apply_to_pseudo", True))
        gradient_surgery_eps = max(0.0, float(getattr(gradient_surgery_cfg, "eps", 1e-12)))
        if gradient_surgery_enabled and gradient_surgery_method != "target_anchored_pcgrad":
            raise ValueError(
                "Shared SFOD gradient_surgery currently supports only "
                f"target_anchored_pcgrad, got {gradient_surgery_method!r}"
            )
        query_recovery_cfg = getattr(method_cfg, "query_recovery", object())
        query_recovery_enabled = bool(getattr(query_recovery_cfg, "enabled", False))
        query_recovery_fit_max_images = max(0, int(getattr(query_recovery_cfg, "fit_max_images", 0)))
        query_recovery_train_as = str(getattr(query_recovery_cfg, "train_as", "hard_pseudo")).strip().lower()
        query_revival_cfg = getattr(query_recovery_cfg, "revival_loss", object())
        query_revival_enabled = query_recovery_enabled and query_recovery_train_as == "revival_loss"
        query_revival_loss_weight = float(getattr(query_revival_cfg, "loss_weight", 0.02))
        query_revival_match_mode = str(getattr(query_revival_cfg, "match_mode", "box_iou")).strip().lower()
        query_revival_min_match_iou = float(getattr(query_revival_cfg, "min_match_iou", 0.40))
        query_revival_match_class_aware = bool(getattr(query_revival_cfg, "match_class_aware", False))
        query_revival_positive_target = float(getattr(query_revival_cfg, "positive_target", 0.70))
        query_revival_foreground_pool = str(
            getattr(query_revival_cfg, "foreground_pool", "mean_logsumexp")
        ).strip().lower()
        query_revival_foreground_temperature = float(getattr(query_revival_cfg, "foreground_temperature", 1.0))
        query_revival_weight_power = float(getattr(query_revival_cfg, "recovery_weight_power", 1.0))
        query_revival_min_weight = float(getattr(query_revival_cfg, "min_candidate_weight", 0.10))
        if query_recovery_enabled and query_recovery_train_as not in {"hard_pseudo", "revival_loss"}:
            raise ValueError(
                "method.query_recovery.train_as must be hard_pseudo or revival_loss, "
                f"got {query_recovery_train_as!r}"
            )
        if query_revival_enabled and query_revival_match_mode not in {"query_index", "box_iou"}:
            raise ValueError(
                "method.query_recovery.revival_loss.match_mode must be query_index or box_iou, "
                f"got {query_revival_match_mode!r}"
            )
        if query_revival_enabled and query_revival_foreground_pool not in {"mean_logsumexp", "max", "top2_mean"}:
            raise ValueError(
                "method.query_recovery.revival_loss.foreground_pool must be mean_logsumexp, max, or top2_mean, "
                f"got {query_revival_foreground_pool!r}"
            )
        label_guided_aema_cfg = getattr(method_cfg, "label_guided_aema", object())
        label_guided_aema_enabled = bool(getattr(label_guided_aema_cfg, "enabled", False))
        label_guided_aema_merge = str(getattr(label_guided_aema_cfg, "merge", "max")).strip().lower()
        label_guided_aema_weight = float(getattr(label_guided_aema_cfg, "guidance_weight", 1.0))
        label_guided_aema_normalize = bool(getattr(label_guided_aema_cfg, "normalize", True))
        label_guided_aema_loss_mode = str(getattr(label_guided_aema_cfg, "loss", "full")).strip().lower()
        label_guided_aema_loss_weight = float(getattr(label_guided_aema_cfg, "supervised_loss_weight", 1.0))
        label_guided_aema_top_fraction = float(getattr(label_guided_aema_cfg, "top_fraction", 0.1))
        label_guided_aema_adaptive_momentum = float(
            getattr(label_guided_aema_cfg, "adaptive_momentum", max(0.0, ema_momentum - 0.002))
        )
        if label_guided_aema_enabled and label_guided_aema_merge not in {"max", "add", "gt_only", "base_only"}:
            raise ValueError(
                "method.label_guided_aema.merge must be one of max/add/gt_only/base_only, "
                f"got {label_guided_aema_merge!r}"
            )
        if label_guided_aema_enabled and label_guided_aema_loss_mode not in {"full", "class"}:
            raise ValueError(
                "method.label_guided_aema.loss must be 'full' or 'class', "
                f"got {label_guided_aema_loss_mode!r}"
            )
        log_period = int(getattr(train_cfg, "log_period", 100))
        checkpoint_period = int(getattr(train_cfg, "checkpoint_period", 0))
        intermediate_eval_cfg = getattr(eval_cfg, "intermediate", object())
        intermediate_eval_enabled = bool(
            getattr(intermediate_eval_cfg, "enabled", getattr(eval_cfg, "intermediate_enabled", False))
        )
        intermediate_eval_interval_steps = max(
            1,
            int(
                getattr(
                    intermediate_eval_cfg,
                    "interval_steps",
                    getattr(eval_cfg, "intermediate_interval_steps", 1000),
                )
            ),
        )
        intermediate_eval_model = str(
            getattr(intermediate_eval_cfg, "model", getattr(eval_cfg, "intermediate_model", "student"))
        ).strip().lower()
        if intermediate_eval_model != "student":
            raise ValueError(
                "Shared SFOD intermediate eval is student-only for early stopping; "
                f"set method.eval.intermediate.model=student, got {intermediate_eval_model!r}"
            )
        intermediate_eval_limit = int(
            getattr(
                intermediate_eval_cfg,
                "target_val_limit",
                getattr(eval_cfg, "intermediate_target_val_limit", getattr(eval_cfg, "target_val_limit", 0)),
            )
        )
        early_stop_patience = max(
            0,
            int(
                getattr(
                    intermediate_eval_cfg,
                    "early_stop_patience",
                    getattr(eval_cfg, "early_stop_patience", 3),
                )
            ),
        )
        early_stop_min_delta = max(
            0.0,
            float(
                getattr(
                    intermediate_eval_cfg,
                    "early_stop_min_delta",
                    getattr(eval_cfg, "early_stop_min_delta", 0.0),
                )
            ),
        )
        early_stop_mode = str(
            getattr(
                intermediate_eval_cfg,
                "early_stop_mode",
                getattr(eval_cfg, "early_stop_mode", "consecutive_drop"),
            )
        ).strip().lower()
        if early_stop_mode not in {"consecutive_drop", "no_improve"}:
            raise ValueError(
                "method.eval.intermediate.early_stop_mode must be consecutive_drop or no_improve, "
                f"got {early_stop_mode!r}"
            )
        train_log_path = run_dir / "train_log.jsonl"
        weak_rng = random.Random(seed + 17)
        strong_rng = random.Random(seed + 31)
        trainable_params = [parameter for parameter in student_model.parameters() if parameter.requires_grad]

        if label_guided_hook_requires_teacher_fit(method_cfg):
            label_cfg = getattr(method_cfg, "label_guided", object())
            fit_max_images = max(0, int(getattr(label_cfg, "fit_max_images", 0)))
            fit_items = list(target_labeled)
            if fit_max_images > 0:
                fit_items = fit_items[:fit_max_images]
            fit_adapter = dynamic_teacher_adapter if self.algorithm == "pets" else teacher_adapter
            fit_device = dynamic_device if self.algorithm == "pets" and dynamic_device is not None else teacher_device
            print(
                f"[{self.log_prefix}][label_guided][fit] "
                f"method={getattr(label_cfg, 'method', 'unknown')} "
                f"labeled_images={len(fit_items)}",
                flush=True,
            )
            fit_teacher_items: list[dict[str, Any]] = []
            if fit_items and fit_adapter is not None:
                with _device_context(fit_device), torch.no_grad():
                    fit_teacher_items = _teacher_outputs_for_unlabeled(
                        fit_adapter,
                        fit_items,
                        weak_view_rng=random.Random(seed + 4242),
                    )
            label_guided_hook = build_label_guided_hook(
                method_cfg,
                fit_teacher_items=fit_teacher_items,
                base_thresholds=base_thresholds,
                num_classes=num_classes,
            )
            label_guided_summary = label_guided_hook.state.component_summary
            save_json(run_dir / "label_guided_components.json", label_guided_summary)
            save_json(run_dir / "label_guided_hook_state.json", label_guided_hook.state.as_dict())
            fitted_name = next(iter(label_guided_hook.state.step_stats.keys()), "none")
            calibration_state = label_guided_hook.state.step_stats.get(fitted_name, {})
            print(
                f"[{self.log_prefix}][label_guided][fit] "
                f"state={fitted_name} "
                f"adjusted_classes={calibration_state.get('adjusted_classes', [])} "
                f"mean_abs_offset={float(calibration_state.get('aggregate', {}).get('mean_abs_offset', 0.0)):.4f}",
                flush=True,
            )

        lpld_cfg = getattr(method_cfg, "lpld", object())
        lpu_cfg = getattr(method_cfg, "lpu", object())
        pets_cfg = getattr(method_cfg, "pets", object())
        pets_exchange_period = int(getattr(pets_cfg, "exchange_period_steps", 1000))
        pets_ema_momentum = float(getattr(pets_cfg, "dynamic_ema_momentum", ema_momentum))
        label_guided_aema_momentum = float(getattr(label_guided_aema_cfg, "momentum", pets_ema_momentum if self.algorithm == "pets" else ema_momentum))

        query_recovery_scorer: QueryRecoveryScorer | None = None
        query_recovery_stats: dict[str, Any] = {}
        if query_recovery_enabled:
            fit_items = list(target_labeled)
            if query_recovery_fit_max_images > 0:
                fit_items = fit_items[:query_recovery_fit_max_images]
            num_views = _query_recovery_num_views(query_recovery_cfg)
            try:
                query_recovery_cfg["_resolved_num_views"] = int(num_views)
            except TypeError:
                setattr(query_recovery_cfg, "_resolved_num_views", int(num_views))
            fit_adapter = dynamic_teacher_adapter if self.algorithm == "pets" else teacher_adapter
            fit_device = dynamic_device if self.algorithm == "pets" and dynamic_device is not None else teacher_device
            print(
                f"[{self.log_prefix}][query_recovery][fit] "
                f"train_as={query_recovery_train_as} labeled_images={len(fit_items)} "
                f"views={num_views} min_score={float(getattr(query_recovery_cfg, 'min_score', 0.01)):.3f}",
                flush=True,
            )
            recovery_fit_items: list[dict[str, Any]] = []
            if fit_items and fit_adapter is not None:
                with _device_context(fit_device), torch.no_grad():
                    if self.algorithm == "pets":
                        assert dynamic_teacher_adapter is not None
                        assert static_teacher_adapter is not None
                        dynamic_fit_items = _teacher_outputs_for_unlabeled(
                            dynamic_teacher_adapter,
                            fit_items,
                            weak_view_rng=random.Random(seed + 5252),
                        )
                        static_fit_items = _teacher_outputs_for_unlabeled(
                            static_teacher_adapter,
                            fit_items,
                            weak_view_rng=random.Random(seed + 5252),
                        )
                        primary_fit_items, _ = self._build_pets_consensus_items(
                            dynamic_fit_items,
                            static_fit_items,
                            pseudo_cfg=pseudo_cfg,
                            pets_cfg=pets_cfg,
                        )
                    else:
                        primary_fit_items = _teacher_outputs_for_unlabeled(
                            fit_adapter,
                            fit_items,
                            weak_view_rng=random.Random(seed + 5252),
                        )
                recovery_fit_items = _build_query_recovery_teacher_items(
                    teacher_adapter=fit_adapter,
                    primary_items=primary_fit_items,
                    source_batch=fit_items,
                    recovery_cfg=query_recovery_cfg,
                    teacher_device=fit_device,
                    seed=seed + 9090,
                    step_offset=0,
                )
                query_recovery_scorer = fit_query_recovery_scorer(
                    recovery_fit_items,
                    thresholds=base_thresholds,
                    num_classes=num_classes,
                    recovery_cfg=query_recovery_cfg,
                    seed=seed,
                    dedup_iou_thresh=dedup_iou_thresh,
                )
                query_recovery_stats = query_recovery_scorer.summary()
            else:
                query_recovery_stats = {
                    "fit_images": 0,
                    "fit_candidates": 0,
                    "fit_positive": 0,
                    "fit_negative": 0,
                    "reason": "no_labeled_target_images",
                }
            query_recovery_stats.update(
                {
                    "enabled": True,
                    "train_as": query_recovery_train_as,
                    "fit_max_images": int(query_recovery_fit_max_images),
                    "revival_loss": {
                        "enabled": bool(query_revival_enabled),
                        "loss_weight": float(query_revival_loss_weight) if query_revival_enabled else None,
                        "match_mode": query_revival_match_mode if query_revival_enabled else None,
                        "min_match_iou": float(query_revival_min_match_iou) if query_revival_enabled else None,
                        "match_class_aware": bool(query_revival_match_class_aware) if query_revival_enabled else None,
                        "positive_target": float(query_revival_positive_target) if query_revival_enabled else None,
                        "foreground_pool": query_revival_foreground_pool if query_revival_enabled else None,
                    },
                }
            )
            save_json(run_dir / "query_recovery.json", query_recovery_stats)

        print(
            f"[{self.log_prefix}][train] "
            f"algorithm={self.algorithm} epochs={epochs} target_train={len(target_train)} "
            f"labeled_target={len(target_labeled)} unlabeled_target={len(target_unlabeled)} "
            f"student_device={self.device} teacher_device={teacher_device} source_ckpt={source_checkpoint}"
        )
        if label_guided_summary["enabled_component_names"]:
            print(
                f"[{self.log_prefix}][label_guided] "
                f"components={','.join(label_guided_summary['enabled_component_names'])} "
                f"categories={','.join(label_guided_summary['enabled_categories'])} "
                f"legacy={','.join(label_guided_summary['legacy_live_component_names']) or 'none'}"
            )
        if gradient_surgery_enabled:
            print(
                f"[{self.log_prefix}][gradient_surgery] "
                f"method={gradient_surgery_method} apply_to_pseudo={gradient_surgery_apply_pseudo}",
                flush=True,
            )
        if label_guided_aema_enabled:
            print(
                f"[{self.log_prefix}][label_guided_aema] "
                f"merge={label_guided_aema_merge} guidance_weight={label_guided_aema_weight:.3f} "
                f"normalize={label_guided_aema_normalize} loss={label_guided_aema_loss_mode} "
                f"top_fraction={label_guided_aema_top_fraction:.3f}",
                flush=True,
            )
        if query_recovery_enabled:
            print(
                f"[{self.log_prefix}][query_recovery] "
                f"train_as={query_recovery_train_as} fit_candidates={query_recovery_stats.get('fit_candidates', 0)} "
                f"fit_positive={query_recovery_stats.get('fit_positive', 0)} "
                f"views={query_recovery_stats.get('multi_view', {}).get('views', 1)}",
                flush=True,
            )
        if intermediate_eval_enabled:
            print(
                f"[{self.log_prefix}][intermediate_eval] "
                f"exp={exp_name} enabled model=student interval_steps={intermediate_eval_interval_steps} "
                f"target_val_limit={intermediate_eval_limit} early_stop_mode={early_stop_mode} "
                f"patience={early_stop_patience} min_delta={early_stop_min_delta:.4f}",
                flush=True,
            )

        global_step = 0
        exchange_count = 0
        history: list[dict[str, Any]] = []
        intermediate_target_val: list[dict[str, Any]] | None = None
        intermediate_eval_history: list[dict[str, Any]] = []
        early_stop_triggered = False
        early_stop_record: dict[str, Any] | None = None
        best_student_ap50: float | None = None
        best_student_step: int | None = None
        best_student_epoch: int | None = None
        best_student_metrics: dict[str, Any] = {}
        best_student_ckpt = run_dir / "student_best_intermediate.pth"
        previous_student_ap50: float | None = None
        consecutive_drop_count = 0
        no_improve_count = 0

        def maybe_run_intermediate_eval(epoch_idx: int) -> bool:
            nonlocal intermediate_target_val
            nonlocal early_stop_triggered, early_stop_record
            nonlocal best_student_ap50, best_student_step, best_student_epoch, best_student_metrics
            nonlocal previous_student_ap50, consecutive_drop_count, no_improve_count
            if not intermediate_eval_enabled or global_step % intermediate_eval_interval_steps != 0:
                return False
            if intermediate_target_val is None:
                intermediate_target_val = _limit_samples(
                    materialize_daod_dicts(self.cfg, "target_val"),
                    intermediate_eval_limit,
                )
            try:
                eval_record = _run_intermediate_target_eval(
                    cfg=self.cfg,
                    run_dir=run_dir,
                    model=student_model,
                    log_prefix=self.log_prefix,
                    exp_name=exp_name,
                    split_items=intermediate_target_val,
                    device=self.device,
                    epoch=epoch_idx,
                    step=global_step,
                    target_val_limit=intermediate_eval_limit,
                )
                ap50 = eval_record.get("AP50")
                if ap50 is not None:
                    ap50_value = float(ap50)
                    patience_update = _student_eval_patience_update(
                        ap50=ap50_value,
                        previous_ap50=previous_student_ap50,
                        best_ap50=best_student_ap50,
                        no_improve_count=no_improve_count,
                        consecutive_drop_count=consecutive_drop_count,
                        min_delta=early_stop_min_delta,
                        mode=early_stop_mode,
                        patience=early_stop_patience,
                    )
                    improved = bool(patience_update["improved"])
                    previous_student_ap50 = float(patience_update["previous_ap50"])
                    consecutive_drop_count = int(patience_update["consecutive_drop_count"])
                    no_improve_count = int(patience_update["no_improve_count"])
                    if improved:
                        best_student_ap50 = ap50_value
                        best_student_step = int(global_step)
                        best_student_epoch = int(epoch_idx)
                        best_student_metrics = dict(eval_record.get("metrics", {}))
                        DetectionCheckpointer(student_model, save_dir=str(run_dir)).save(
                            "student_best_intermediate"
                        )
                        eval_record["is_best"] = True
                        print(
                            f"[{self.log_prefix}][best_student] "
                            f"exp={exp_name} epoch={epoch_idx} step={global_step} AP50={ap50_value:.3f}",
                            flush=True,
                        )
                    else:
                        eval_record["is_best"] = False
                    eval_record["best_AP50"] = best_student_ap50
                    eval_record["best_step"] = best_student_step
                    eval_record["best_epoch"] = best_student_epoch
                    eval_record["consecutive_drop_count"] = int(consecutive_drop_count)
                    eval_record["no_improve_count"] = int(no_improve_count)
                    eval_record["early_stop_mode"] = early_stop_mode
                    eval_record["early_stop_patience"] = int(early_stop_patience)
                    eval_record_compact = _compact_intermediate_eval_record(eval_record)
                    save_json(run_dir / "best_student_target_val_metrics.json", best_student_metrics)
                    save_json(run_dir / "last_student_eval.json", eval_record_compact)
                    if eval_record.get("is_best", False):
                        save_json(run_dir / "best_student_eval.json", eval_record_compact)
                    stop_count = int(patience_update["stop_count"])
                    if bool(patience_update["should_stop"]):
                        early_stop_triggered = True
                        early_stop_record = {
                            **_compact_intermediate_eval_record(eval_record),
                            "reason": f"{early_stop_mode}_patience_exhausted",
                            "patience": int(early_stop_patience),
                            "stop_count": int(stop_count),
                        }
                        print(
                            f"[{self.log_prefix}][early_stop] "
                            f"exp={exp_name} epoch={epoch_idx} step={global_step} "
                            f"AP50={ap50_value:.3f} best_AP50={float(best_student_ap50):.3f} "
                            f"mode={early_stop_mode} count={int(stop_count)} "
                            f"patience={int(early_stop_patience)}",
                            flush=True,
                        )
            except RuntimeError as exc:
                eval_record = {
                    "exp_name": exp_name,
                    "epoch": int(epoch_idx),
                    "step": int(global_step),
                    "model": "student",
                    "target_val_limit": int(intermediate_eval_limit),
                    "error": str(exc),
                }
                print(
                    f"[{self.log_prefix}][intermediate_eval][warning] "
                    f"exp={exp_name} epoch={epoch_idx} step={global_step} model=student failed: {exc}",
                    flush=True,
                )
            student_model.train()
            if teacher_model is not None:
                teacher_model.eval()
            if static_teacher_model is not None:
                static_teacher_model.eval()
            if dynamic_teacher_model is not None:
                dynamic_teacher_model.eval()
            eval_record_compact = _compact_intermediate_eval_record(eval_record)
            intermediate_eval_history.append(eval_record_compact)
            append_jsonl(run_dir / "intermediate_eval_log.jsonl", eval_record_compact)
            if early_stop_triggered:
                print(
                    f"[{self.log_prefix}][early_stop] exp={exp_name} exiting_train_loop_at_step={global_step}",
                    flush=True,
                )
            return early_stop_triggered

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
                "loss_query_revival": 0.0,
                "query_recovery_candidates": 0,
                "query_recovery_selected": 0,
                "query_revival_targets": 0,
                "query_revival_matched": 0,
            }
            epoch_gradient_surgery_stats = _new_gradient_surgery_epoch_stats()
            epoch_query_recovery_stats = QueryRecoverySelectionStats()
            epoch_query_revival_stats = _new_query_revival_epoch_stats()
            epoch_label_guided_teacher_loss = 0.0
            epoch_label_guided_teacher_steps = 0
            epoch_aema_update_steps = 0
            epoch_ema_update_steps = 0

            for _ in range(steps_per_epoch):
                batch = next(target_iter, [])
                labeled_batch = next(labeled_iter, [])
                pseudo_batch: list[dict[str, Any]] = []
                low_items: list[dict[str, Any]] = []
                query_revival_items: list[dict[str, Any]] = []
                pseudo_box_count = 0
                low_stats = {"low_targets": 0, "matched_targets": 0, "pst_pairs": 0, "lscl_pairs": 0}
                pets_consensus_stats = {"consensus_rows": 0}
                step_query_recovery_stats = QueryRecoverySelectionStats()
                query_revival_loss_stats = QueryRevivalLossStats()
                effective_thresholds = label_guided_hook.adjust_thresholds(
                    [float(value) for value in base_thresholds],
                    global_step=global_step,
                )

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
                    teacher_items = label_guided_hook.before_pseudo_filter(
                        teacher_items,
                        thresholds=effective_thresholds,
                        global_step=global_step,
                    )
                    query_recovery_items: list[dict[str, Any]] | None = None
                    if query_recovery_scorer is not None:
                        recovery_adapter = dynamic_teacher_adapter if self.algorithm == "pets" else teacher_adapter
                        recovery_device = dynamic_device if self.algorithm == "pets" and dynamic_device is not None else teacher_device
                        assert recovery_adapter is not None
                        query_recovery_items = _build_query_recovery_teacher_items(
                            teacher_adapter=recovery_adapter,
                            primary_items=teacher_items,
                            source_batch=batch,
                            recovery_cfg=query_recovery_cfg,
                            teacher_device=recovery_device,
                            seed=seed + 31337,
                            step_offset=global_step,
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
                            thresholds=effective_thresholds,
                            dedup_iou_thresh=dedup_iou_thresh,
                        )
                        if query_recovery_scorer is not None and query_recovery_items is not None:
                            recovery_item = query_recovery_items[item_idx]
                            recovered_rows, item_recovery_stats = query_recovery_scorer.select(
                                recovery_item["query_rows"],
                                thresholds=effective_thresholds,
                                dedup_iou_thresh=dedup_iou_thresh,
                                sample=recovery_item["sample"],
                                existing_rows=hard_rows,
                            )
                            if recovered_rows:
                                if query_recovery_train_as == "hard_pseudo":
                                    hard_rows = [*hard_rows, *recovered_rows]
                                elif query_revival_enabled:
                                    enriched_recovery_rows = []
                                    for row in recovered_rows:
                                        class_id = int(row.get("category_id", -1))
                                        enriched_row = dict(row)
                                        if 0 <= class_id < len(effective_thresholds):
                                            enriched_row["_pseudo_threshold"] = float(effective_thresholds[class_id])
                                        enriched_recovery_rows.append(enriched_row)
                                    query_revival_items.append(
                                        {
                                            "sample": recovery_item["sample"],
                                            "teacher_rows": enriched_recovery_rows,
                                        }
                                    )
                            step_query_recovery_stats.add(item_recovery_stats)
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

                if query_recovery_enabled:
                    epoch_query_recovery_stats.add(step_query_recovery_stats)

                loss_terms: list[torch.Tensor] = []
                loss_pseudo_value = 0.0
                loss_low_value = 0.0
                loss_query_revival_value = 0.0
                loss_supervised_value = 0.0
                core_loss_scales = {"pseudo": 1.0, "masked": 1.0, "supervised": 1.0}
                loss_pseudo: torch.Tensor | None = None
                loss_low: torch.Tensor | None = None
                loss_query_revival: torch.Tensor | None = None
                loss_supervised: torch.Tensor | None = None
                defer_query_revival_backward = bool(query_revival_items) and not bool(gradient_surgery_enabled)

                def compute_query_revival_loss_for_step() -> tuple[
                    torch.Tensor | None,
                    QueryRevivalLossStats,
                    float,
                ]:
                    stats = QueryRevivalLossStats()
                    if not query_revival_items:
                        return None, stats, 0.0
                    query_revival_strong_rng = random.Random(seed + 818181 + global_step)
                    query_revival_loss_items: list[dict[str, Any]] = []
                    with _device_context(self.device):
                        for revival_item in query_revival_items:
                            strong_sample = build_strong_view_sample(
                                revival_item["sample"],
                                rng=query_revival_strong_rng,
                                suffix="student_query_revival",
                            )
                            student_raw = run_daod_raw_outputs(
                                student_adapter,
                                strong_sample,
                                with_grad=True,
                            )[0]
                            query_revival_loss_items.append({**revival_item, "student_raw": student_raw})
                    loss_value, stats = query_revival_loss(
                        query_revival_loss_items,
                        loss_weight=query_revival_loss_weight,
                        match_mode=query_revival_match_mode,
                        min_match_iou=query_revival_min_match_iou,
                        match_class_aware=query_revival_match_class_aware,
                        positive_target=query_revival_positive_target,
                        foreground_pool=query_revival_foreground_pool,
                        foreground_temperature=query_revival_foreground_temperature,
                        recovery_weight_power=query_revival_weight_power,
                        min_candidate_weight=query_revival_min_weight,
                        class_budgets=query_recovery_scorer.class_budgets
                        if query_recovery_scorer is not None
                        else None,
                    )
                    value = float(loss_value.detach().cpu()) if loss_value.requires_grad else 0.0
                    return loss_value, stats, value

                if pseudo_batch:
                    pseudo_inputs = _make_supervised_inputs(
                        student_adapter,
                        pseudo_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                        strong_view_rng=strong_rng,
                    )
                    loss_pseudo = float(pseudo_weight) * _sum_loss_dict(student_model(pseudo_inputs))
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

                if query_revival_items and not defer_query_revival_backward:
                    (
                        loss_query_revival,
                        query_revival_loss_stats,
                        loss_query_revival_value,
                    ) = compute_query_revival_loss_for_step()
                    if loss_query_revival is not None and loss_query_revival.requires_grad:
                        loss_terms.append(loss_query_revival)

                if labeled_batch:
                    supervised_inputs = _make_supervised_inputs(
                        student_adapter,
                        labeled_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                        strong_view_rng=strong_rng,
                    )
                    loss_supervised = float(supervised_weight) * _sum_loss_dict(student_model(supervised_inputs))
                    loss_supervised_value = float(loss_supervised.detach().cpu())

                if loss_pseudo is not None or loss_supervised is not None:
                    core_loss_scales = label_guided_hook.loss_scales(
                        {
                            "pseudo": loss_pseudo_value if loss_pseudo is not None else 0.0,
                            "masked": 0.0,
                            "supervised": loss_supervised_value if loss_supervised is not None else 0.0,
                        },
                        global_step=global_step,
                    )
                    if loss_pseudo is not None:
                        loss_pseudo = float(core_loss_scales.get("pseudo", 1.0)) * loss_pseudo
                        loss_terms.append(loss_pseudo)
                        loss_pseudo_value = float(loss_pseudo.detach().cpu())
                    if loss_supervised is not None:
                        loss_supervised = float(core_loss_scales.get("supervised", 1.0)) * loss_supervised
                        loss_terms.append(loss_supervised)
                        loss_supervised_value = float(loss_supervised.detach().cpu())

                has_deferred_query_revival = bool(query_revival_items) and bool(defer_query_revival_backward)
                if not loss_terms and not has_deferred_query_revival:
                    global_step += 1
                    if maybe_run_intermediate_eval(epoch_idx):
                        break
                    continue

                loss = sum(loss_terms) if loss_terms else torch.tensor(0.0, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                pseudo_grad_stats: PCGradStats | None = None
                gradient_surgery_can_run = (
                    gradient_surgery_enabled
                    and gradient_surgery_apply_pseudo
                    and loss_supervised is not None
                    and loss_supervised.requires_grad
                    and loss_pseudo is not None
                    and loss_pseudo.requires_grad
                )
                did_backward = False
                if gradient_surgery_can_run:
                    supervised_grads = _loss_grads(
                        loss_supervised,
                        trainable_params,
                        retain_graph=False,
                    )
                    combined_grads = clone_grad_list(supervised_grads)
                    pseudo_grads = _loss_grads(
                        loss_pseudo,
                        trainable_params,
                        retain_graph=False,
                    )
                    pseudo_grads, pseudo_grad_stats = target_anchored_pcgrad(
                        anchor_grads=supervised_grads,
                        aux_grads=pseudo_grads,
                        eps=gradient_surgery_eps,
                    )
                    add_grads_in_place(combined_grads, pseudo_grads)
                    for untouched_loss in (loss_low, loss_query_revival):
                        if untouched_loss is not None and untouched_loss.requires_grad:
                            add_grads_in_place(
                                combined_grads,
                                _loss_grads(untouched_loss, trainable_params, retain_graph=False),
                            )
                    assign_grads(trainable_params, combined_grads)
                    epoch_gradient_surgery_stats["steps"] += 1.0
                    _accumulate_gradient_surgery_stats(epoch_gradient_surgery_stats, pseudo_grad_stats)
                    did_backward = True
                else:
                    if loss_terms:
                        loss.backward()
                        did_backward = True
                    if has_deferred_query_revival:
                        # Build the qrev graph only after core SFOD gradients are
                        # accumulated. This avoids holding pseudo/LPLD/LPU graphs
                        # and qrev raw-output graphs in memory at the same time.
                        loss_terms = []
                        loss_pseudo = None
                        loss_low = None
                        loss_supervised = None
                        (
                            loss_query_revival,
                            query_revival_loss_stats,
                            loss_query_revival_value,
                        ) = compute_query_revival_loss_for_step()
                        if loss_query_revival is not None and loss_query_revival.requires_grad:
                            loss_query_revival.backward()
                            loss = loss.detach() + loss_query_revival.detach()
                            did_backward = True
                if not did_backward:
                    global_step += 1
                    if maybe_run_intermediate_eval(epoch_idx):
                        break
                    continue
                if float(getattr(train_cfg, "clip_max_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        float(getattr(train_cfg, "clip_max_norm", 0.0)),
                    )
                optimizer.step()
                scheduler.step()
                global_step += 1

                label_guided_teacher_loss_value = 0.0
                pseudo_importance: dict[str, torch.Tensor] = {}
                gt_importance: dict[str, torch.Tensor] = {}
                teacher_importance: dict[str, torch.Tensor] = {}
                if self.algorithm == "pets":
                    assert static_teacher_model is not None
                    assert dynamic_teacher_model is not None
                    if pets_exchange_period > 0 and global_step % pets_exchange_period == 0:
                        _swap_model_weights(student_model, static_teacher_model)
                        exchange_count += 1
                    if label_guided_aema_enabled:
                        if pseudo_batch:
                            pseudo_importance, _ = _teacher_grad_importance_from_supervised_batch(
                                teacher_model=dynamic_teacher_model,
                                teacher_adapter=dynamic_teacher_adapter,
                                batch=pseudo_batch,
                                teacher_device=dynamic_device,
                                strong_short_edge=strong_short_edge,
                                max_size=max_size,
                                loss_mode="class",
                            )
                        if labeled_batch:
                            gt_importance, label_guided_teacher_loss_value = _teacher_grad_importance_from_supervised_batch(
                                teacher_model=dynamic_teacher_model,
                                teacher_adapter=dynamic_teacher_adapter,
                                batch=labeled_batch,
                                teacher_device=dynamic_device,
                                strong_short_edge=strong_short_edge,
                                max_size=max_size,
                                loss_mode=label_guided_aema_loss_mode,
                                loss_weight=label_guided_aema_loss_weight,
                            )
                        teacher_importance = merge_importance_maps(
                            pseudo_importance,
                            gt_importance,
                            merge=label_guided_aema_merge,
                            guidance_weight=label_guided_aema_weight,
                            normalize=label_guided_aema_normalize,
                        )
                        if teacher_importance:
                            _update_aema_weighted(
                                dynamic_teacher_model,
                                student_model,
                                teacher_importance,
                                momentum=label_guided_aema_momentum,
                                adaptive_momentum=label_guided_aema_adaptive_momentum,
                                top_fraction=label_guided_aema_top_fraction,
                            )
                            epoch_aema_update_steps += 1
                            if gt_importance:
                                epoch_label_guided_teacher_loss += float(label_guided_teacher_loss_value)
                                epoch_label_guided_teacher_steps += 1
                        else:
                            _update_ema(dynamic_teacher_model, student_model, pets_ema_momentum)
                            epoch_ema_update_steps += 1
                    else:
                        _update_ema(dynamic_teacher_model, student_model, pets_ema_momentum)
                        epoch_ema_update_steps += 1
                elif teacher_model is not None and ema_update_interval > 0 and global_step % ema_update_interval == 0:
                    if label_guided_aema_enabled:
                        if pseudo_batch:
                            pseudo_importance, _ = _teacher_grad_importance_from_supervised_batch(
                                teacher_model=teacher_model,
                                teacher_adapter=teacher_adapter,
                                batch=pseudo_batch,
                                teacher_device=teacher_device,
                                strong_short_edge=strong_short_edge,
                                max_size=max_size,
                                loss_mode="class",
                            )
                        if labeled_batch:
                            gt_importance, label_guided_teacher_loss_value = _teacher_grad_importance_from_supervised_batch(
                                teacher_model=teacher_model,
                                teacher_adapter=teacher_adapter,
                                batch=labeled_batch,
                                teacher_device=teacher_device,
                                strong_short_edge=strong_short_edge,
                                max_size=max_size,
                                loss_mode=label_guided_aema_loss_mode,
                                loss_weight=label_guided_aema_loss_weight,
                            )
                        teacher_importance = merge_importance_maps(
                            pseudo_importance,
                            gt_importance,
                            merge=label_guided_aema_merge,
                            guidance_weight=label_guided_aema_weight,
                            normalize=label_guided_aema_normalize,
                        )
                        if teacher_importance:
                            _update_aema_weighted(
                                teacher_model,
                                student_model,
                                teacher_importance,
                                momentum=label_guided_aema_momentum,
                                adaptive_momentum=label_guided_aema_adaptive_momentum,
                                top_fraction=label_guided_aema_top_fraction,
                            )
                            epoch_aema_update_steps += 1
                            if gt_importance:
                                epoch_label_guided_teacher_loss += float(label_guided_teacher_loss_value)
                                epoch_label_guided_teacher_steps += 1
                        else:
                            _update_ema(teacher_model, student_model, ema_momentum)
                            epoch_ema_update_steps += 1
                    else:
                        _update_ema(teacher_model, student_model, ema_momentum)
                        epoch_ema_update_steps += 1

                epoch_stats["loss_total"] += float(loss.detach().cpu())
                epoch_stats["loss_pseudo"] += loss_pseudo_value
                epoch_stats["loss_low"] += loss_low_value
                epoch_stats["loss_query_revival"] += loss_query_revival_value
                epoch_stats["loss_supervised"] += loss_supervised_value
                epoch_stats["pseudo_boxes"] += int(pseudo_box_count)
                epoch_stats["low_targets"] += int(low_stats.get("low_targets", 0))
                epoch_stats["matched_low_targets"] += int(low_stats.get("matched_targets", 0))
                epoch_stats["pets_consensus_rows"] += int(pets_consensus_stats.get("consensus_rows", 0))
                epoch_stats["query_recovery_candidates"] += int(step_query_recovery_stats.candidates)
                epoch_stats["query_recovery_selected"] += int(step_query_recovery_stats.selected)
                epoch_stats["query_revival_targets"] += int(query_revival_loss_stats.targets)
                epoch_stats["query_revival_matched"] += int(query_revival_loss_stats.matched)
                epoch_query_revival_stats.add(query_revival_loss_stats)

                if log_period > 0 and global_step % log_period == 0:
                    query_recovery_log_fields: dict[str, Any] = {}
                    if query_recovery_enabled:
                        query_recovery_log_fields = {
                            "query_recovery_enabled": True,
                            "query_recovery_candidates": int(step_query_recovery_stats.candidates),
                            "query_recovery_selected": int(step_query_recovery_stats.selected),
                            "query_recovery_mean_score": step_query_recovery_stats.as_dict().get("mean_score"),
                        }
                    append_jsonl(
                        train_log_path,
                        {
                            "epoch": int(epoch_idx),
                            "step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_total": float(loss.detach().cpu()),
                            "loss_pseudo": loss_pseudo_value,
                            "loss_low": loss_low_value,
                            "loss_query_revival": loss_query_revival_value,
                            "loss_supervised": loss_supervised_value,
                            "pseudo_box_count": int(pseudo_box_count),
                            "low_stats": low_stats,
                            "pets_consensus": pets_consensus_stats,
                            "pets_exchanges": int(exchange_count),
                            "effective_thresholds": [float(value) for value in effective_thresholds],
                            "loss_scales": {
                                key: float(value) for key, value in core_loss_scales.items()
                            },
                            **query_recovery_log_fields,
                            **_query_revival_log_fields(
                                enabled=query_revival_enabled,
                                loss_stats=query_revival_loss_stats if query_revival_enabled else None,
                            ),
                            "label_guided_aema_enabled": bool(label_guided_aema_enabled),
                            "label_guided_aema_merge": label_guided_aema_merge
                            if label_guided_aema_enabled
                            else None,
                            "label_guided_teacher_loss": float(label_guided_teacher_loss_value)
                            if label_guided_aema_enabled and gt_importance
                            else None,
                            "teacher_importance_has_pseudo": bool(pseudo_importance),
                            "teacher_importance_has_gt": bool(gt_importance),
                            "teacher_importance_stats": importance_map_stats(teacher_importance)
                            if teacher_importance
                            else None,
                            **(
                                _gradient_surgery_log_fields(
                                    method=gradient_surgery_method,
                                    pseudo_stats=pseudo_grad_stats,
                                )
                                if gradient_surgery_enabled
                                else {}
                            ),
                        },
                    )

                if checkpoint_period > 0 and global_step % checkpoint_period == 0:
                    DetectionCheckpointer(student_model, save_dir=str(run_dir)).save(f"student_step_{global_step}")
                if maybe_run_intermediate_eval(epoch_idx):
                    break

            denom = max(steps_per_epoch, 1)
            epoch_summary = {
                "epoch": int(epoch_idx),
                "loss_total": float(epoch_stats["loss_total"] / denom),
                "loss_pseudo": float(epoch_stats["loss_pseudo"] / denom),
                "loss_low": float(epoch_stats["loss_low"] / denom),
                "loss_query_revival": float(epoch_stats["loss_query_revival"] / denom),
                "loss_supervised": float(epoch_stats["loss_supervised"] / denom),
                "pseudo_boxes": int(epoch_stats["pseudo_boxes"]),
                "low_targets": int(epoch_stats["low_targets"]),
                "matched_low_targets": int(epoch_stats["matched_low_targets"]),
                "pets_consensus_rows": int(epoch_stats["pets_consensus_rows"]),
                "pets_exchanges": int(exchange_count),
                "gradient_surgery": _gradient_surgery_epoch_summary(
                    epoch_gradient_surgery_stats,
                    enabled=gradient_surgery_enabled,
                    method=gradient_surgery_method,
                ),
                "query_recovery": {
                    "enabled": bool(query_recovery_enabled),
                    "train_as": query_recovery_train_as if query_recovery_enabled else None,
                    "fit": query_recovery_scorer.summary() if query_recovery_scorer is not None else query_recovery_stats,
                    "selection": epoch_query_recovery_stats.as_dict()
                    if query_recovery_enabled
                    else None,
                },
                "query_revival": _query_revival_epoch_summary(
                    epoch_query_revival_stats,
                    enabled=query_revival_enabled,
                    loss_weight=query_revival_loss_weight,
                    train_as=query_recovery_train_as,
                    match_mode=query_revival_match_mode,
                    foreground_pool=query_revival_foreground_pool,
                ),
                "teacher_update": {
                    "label_guided_aema_enabled": bool(label_guided_aema_enabled),
                    "label_guided_aema_merge": label_guided_aema_merge
                    if label_guided_aema_enabled
                    else None,
                    "label_guided_teacher_loss": float(
                        epoch_label_guided_teacher_loss / max(epoch_label_guided_teacher_steps, 1)
                    )
                    if label_guided_aema_enabled
                    else None,
                    "aema_update_steps": int(epoch_aema_update_steps),
                    "ema_update_steps": int(epoch_ema_update_steps),
                },
                "base_thresholds": [float(value) for value in base_thresholds],
                "effective_thresholds": [
                    float(value)
                    for value in label_guided_hook.adjust_thresholds(
                        [float(item) for item in base_thresholds],
                        global_step=global_step,
                    )
                ],
                "label_guided_hook": label_guided_hook.state.as_dict(),
            }
            if intermediate_eval_history:
                epoch_summary["latest_intermediate_eval"] = intermediate_eval_history[-1]
            history.append(epoch_summary)
            append_jsonl(run_dir / "epoch_log.jsonl", epoch_summary)
            if early_stop_triggered:
                break

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
        final_student_checkpoint = student_ckpt
        final_student_source = "last"
        if best_student_ckpt.exists():
            DetectionCheckpointer(student_model).load(str(best_student_ckpt))
            final_student_checkpoint = best_student_ckpt
            final_student_source = "best_intermediate"
            print(
                f"[{self.log_prefix}][eval] "
                f"step={global_step} loaded_best_student checkpoint={best_student_ckpt}",
                flush=True,
            )
        student_model.eval()
        print(
            f"[{self.log_prefix}][eval] "
            f"step={global_step} split=target_val student source={final_student_source}",
            flush=True,
        )
        student_eval_error: str | None = None
        try:
            student_target_metrics = _evaluate_split(self.cfg, student_model, "target_val", target_val)
        except RuntimeError as exc:
            student_eval_error = str(exc)
            student_target_metrics = dict(best_student_metrics)
            print(
                f"[{self.log_prefix}][eval][warning] "
                f"student final eval failed; using saved best intermediate metrics: {exc}",
                flush=True,
            )
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

        final_checkpoint: Path = final_student_checkpoint
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
        if best_student_metrics:
            save_json(run_dir / "best_student_target_val_metrics.json", best_student_metrics)
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
            "best_student_checkpoint": str(best_student_ckpt) if best_student_ckpt.exists() else None,
            "final_student_checkpoint": str(final_student_checkpoint),
            "final_student_source": final_student_source,
            "teacher_checkpoint": str(teacher_ckpt) if teacher_ckpt is not None else None,
            "dynamic_teacher_checkpoint": str(dynamic_teacher_ckpt) if dynamic_teacher_ckpt is not None else None,
            "static_teacher_checkpoint": str(static_teacher_ckpt) if static_teacher_ckpt is not None else None,
            "history": history,
            "intermediate_eval": {
                "enabled": bool(intermediate_eval_enabled),
                "model": intermediate_eval_model if intermediate_eval_enabled else None,
                "interval_steps": int(intermediate_eval_interval_steps) if intermediate_eval_enabled else None,
                "target_val_limit": int(intermediate_eval_limit) if intermediate_eval_enabled else None,
                "early_stop_mode": early_stop_mode if intermediate_eval_enabled else None,
                "early_stop_patience": int(early_stop_patience) if intermediate_eval_enabled else None,
                "early_stop_min_delta": float(early_stop_min_delta) if intermediate_eval_enabled else None,
                "early_stop_triggered": bool(early_stop_triggered),
                "early_stop_record": early_stop_record,
                "best_student_AP50": best_student_ap50,
                "best_student_step": best_student_step,
                "best_student_epoch": best_student_epoch,
                "consecutive_drop_count": int(consecutive_drop_count),
                "no_improve_count": int(no_improve_count),
                "history": intermediate_eval_history,
            },
            "active_plan": active_plan,
            "label_guided_components": label_guided_summary,
            "label_guided_hook": label_guided_hook.state.as_dict(),
            "gradient_surgery": {
                "enabled": bool(gradient_surgery_enabled),
                "method": gradient_surgery_method if gradient_surgery_enabled else None,
                "apply_to_pseudo": bool(gradient_surgery_apply_pseudo) if gradient_surgery_enabled else None,
                "history": [
                    entry.get("gradient_surgery", {})
                    for entry in history
                    if bool(entry.get("gradient_surgery", {}).get("enabled", False))
                ],
            },
            "label_guided_aema": {
                "enabled": bool(label_guided_aema_enabled),
                "merge": label_guided_aema_merge if label_guided_aema_enabled else None,
                "guidance_weight": float(label_guided_aema_weight) if label_guided_aema_enabled else None,
                "normalize": bool(label_guided_aema_normalize) if label_guided_aema_enabled else None,
                "loss": label_guided_aema_loss_mode if label_guided_aema_enabled else None,
                "supervised_loss_weight": float(label_guided_aema_loss_weight)
                if label_guided_aema_enabled
                else None,
                "top_fraction": float(label_guided_aema_top_fraction) if label_guided_aema_enabled else None,
                "adaptive_momentum": float(label_guided_aema_adaptive_momentum) if label_guided_aema_enabled else None,
                "history": [
                    entry.get("teacher_update", {})
                    for entry in history
                    if bool(entry.get("teacher_update", {}).get("label_guided_aema_enabled", False))
                ],
            },
            "query_recovery": {
                "enabled": bool(query_recovery_enabled),
                "train_as": query_recovery_train_as if query_recovery_enabled else None,
                "fit_max_images": int(query_recovery_fit_max_images) if query_recovery_enabled else None,
                "fit": query_recovery_scorer.summary() if query_recovery_scorer is not None else query_recovery_stats,
                "history": [
                    entry.get("query_recovery", {})
                    for entry in history
                    if bool(entry.get("query_recovery", {}).get("enabled", False))
                ],
            },
            "query_revival": {
                "enabled": bool(query_revival_enabled),
                "loss_weight": float(query_revival_loss_weight) if query_revival_enabled else None,
                "match_mode": query_revival_match_mode if query_revival_enabled else None,
                "min_match_iou": float(query_revival_min_match_iou) if query_revival_enabled else None,
                "match_class_aware": bool(query_revival_match_class_aware) if query_revival_enabled else None,
                "positive_target": float(query_revival_positive_target) if query_revival_enabled else None,
                "foreground_pool": query_revival_foreground_pool if query_revival_enabled else None,
                "foreground_temperature": float(query_revival_foreground_temperature) if query_revival_enabled else None,
                "recovery_weight_power": float(query_revival_weight_power) if query_revival_enabled else None,
                "min_candidate_weight": float(query_revival_min_weight) if query_revival_enabled else None,
                "history": [
                    entry.get("query_revival", {})
                    for entry in history
                    if bool(entry.get("query_revival", {}).get("enabled", False))
                ],
            },
            "base_thresholds": [float(value) for value in base_thresholds],
            "effective_thresholds": [
                float(value)
                for value in label_guided_hook.adjust_thresholds(
                    [float(item) for item in base_thresholds],
                    global_step=global_step,
                )
            ],
            "student_target_val_metrics": student_target_metrics,
            "teacher_target_val_metrics": teacher_target_metrics,
            "dynamic_teacher_target_val_metrics": dynamic_teacher_target_metrics,
            "static_teacher_target_val_metrics": static_teacher_target_metrics,
            "student_eval_error": student_eval_error,
            "teacher_eval_error": teacher_eval_error,
            "final_target_val_metrics": final_target_metrics,
            "pets_exchanges": int(exchange_count),
        }
        save_json(run_dir / "summary.json", summary)
        maybe_empty_cuda_cache()
        return summary
