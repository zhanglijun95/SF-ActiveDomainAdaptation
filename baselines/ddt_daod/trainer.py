"""DINO-adapted Dual-rate Dynamic Teacher trainer.

This trainer intentionally keeps only the DDT baseline, sparse random target
supervision, and the current soft-query activation branch. Older negative
research branches were removed so the training path stays readable.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import random
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.daod import (
    DAODListDataset,
    build_strong_view_sample,
    collate_daod_batch,
    cycle_daod_loader,
    get_daod_thing_classes,
)
from src.data.daod.detectron2 import materialize_daod_dicts
from src.engine.daod_gradient_surgery import (
    PCGradStats,
    add_grads_in_place,
    assign_grads,
    clone_grad_list,
    target_anchored_pcgrad,
)
from src.engine.daod_latent_query_activation import (
    LatentActivationSelectionStats,
    LatentQueryActivator,
    fit_latent_query_activator,
)
from src.engine.daod_round_trainer import (
    _evaluate_split,
    _limit_samples,
    _make_supervised_inputs,
    _resolve_teacher_device,
    _teacher_outputs_for_unlabeled,
    _update_aema,
    _update_ema,
)
from src.engine.daod_label_guided import build_label_guided_hook, label_guided_hook_requires_teacher_fit
from src.engine.daod_oracle_pseudo import (
    ORACLE_ANNOTATION_KEY,
    OraclePseudoStats,
    apply_oracle_pseudo_intervention,
)
from src.engine.daod_query_recovery import (
    QueryRecoveryScorer,
    QueryRecoverySelectionStats,
    fit_query_recovery_scorer,
    merge_multiview_teacher_items,
)
from src.engine.daod_query_revival import QueryRevivalLossStats, query_revival_loss
from src.engine.daod_soft_query_activation import (
    BenefitRiskClassGate,
    SoftQueryActivationLossStats,
    fit_benefit_risk_class_gate,
    soft_query_activation_loss,
)
from src.engine.daod_teacher_guidance import collect_grad_importance, importance_map_stats, merge_importance_maps
from src.models import build_daod_model, run_daod_raw_outputs

from .masking import apply_block_mask_to_inputs
from .pseudo import filter_pseudo_rows, rows_to_annotations, update_dynamic_thresholds
from .utils import append_jsonl, maybe_empty_cuda_cache, save_json


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
        return sum(loss_dict.values())
    raise ValueError(f"Unsupported teacher importance loss mode: {mode!r}")


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


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


def _sample_id(sample: dict[str, Any]) -> str:
    return str(sample["sample_id"])


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
    model_name: str,
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
            "[DDT-DAOD][intermediate_eval] "
            f"exp={exp_name} epoch={epoch} step={step} split=target_val model={model_name}",
            flush=True,
        )
        metrics = _evaluate_split(cfg, model, "target_val", split_items)
    ap50 = _bbox_ap50(metrics)
    metrics_path = f"intermediate_target_val_{model_name}_step{step}.json"
    save_json(run_dir / metrics_path, metrics)
    if ap50 is not None:
        print(
            "[DDT-DAOD][intermediate_eval] "
            f"exp={exp_name} epoch={epoch} step={step} model={model_name} AP50={ap50:.3f}",
            flush=True,
        )
    return {
        "exp_name": exp_name,
        "epoch": int(epoch),
        "step": int(step),
        "model": model_name,
        "target_val_limit": int(target_val_limit),
        "AP50": ap50,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def _without_annotations(sample: dict[str, Any]) -> dict[str, Any]:
    """Return an unlabeled target view so GT cannot leak into pseudo training."""

    cloned = dict(sample)
    cloned[ORACLE_ANNOTATION_KEY] = [dict(ann) for ann in sample.get("annotations", [])]
    cloned["annotations"] = []
    return cloned


def _resolve_budget_count(budget_cfg: Any, total_count: int) -> int:
    if total_count <= 0:
        return 0
    if isinstance(budget_cfg, float) and 0.0 < float(budget_cfg) <= 1.0:
        return min(total_count, max(1, int(round(float(budget_cfg) * float(total_count)))))
    return min(total_count, max(0, int(budget_cfg)))


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


def _build_sparse_target_split(
    target_train: list[dict[str, Any]],
    active_cfg: Any,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], dict[str, Any]]:
    active_enabled = bool(getattr(active_cfg, "enabled", False))
    if not active_enabled:
        return [], list(target_train), set(), {
            "enabled": False,
            "selected_ids": [],
            "sample_plans": [],
        }

    strategy = str(getattr(active_cfg, "strategy", "random")).strip().lower()
    if strategy != "random":
        raise ValueError(f"DDT sparse active mode currently supports only strategy=random, got {strategy!r}")

    budget_cfg = getattr(active_cfg, "budget_total", 0)
    budget_k = _resolve_budget_count(budget_cfg, len(target_train))
    rng = np.random.default_rng(int(seed))
    selected_indices = (
        sorted(int(idx) for idx in rng.choice(len(target_train), size=budget_k, replace=False))
        if budget_k > 0
        else []
    )
    selected_ids = {_sample_id(target_train[idx]) for idx in selected_indices}
    selected_order = [_sample_id(target_train[idx]) for idx in selected_indices]

    target_labeled = [sample for sample in target_train if _sample_id(sample) in selected_ids]
    target_unlabeled = [_without_annotations(sample) for sample in target_train if _sample_id(sample) not in selected_ids]
    sample_plans = [
        {
            "sample_id": _sample_id(sample),
            "selected": _sample_id(sample) in selected_ids,
            "role": "labeled" if _sample_id(sample) in selected_ids else "unlabeled",
        }
        for sample in target_train
    ]
    plan = {
        "enabled": True,
        "strategy": strategy,
        "budget_total": budget_cfg,
        "budget_k": int(budget_k),
        "target_total": len(target_train),
        "selected_ids": selected_order,
        "sample_plans": sample_plans,
    }
    return target_labeled, target_unlabeled, selected_ids, plan


def _effective_thresholds(
    ddt_thresholds: list[float],
    offsets: list[float],
    *,
    pseudo_cfg: Any,
    recalibration_cfg: Any,
    base_threshold: float,
) -> list[float]:
    """Apply legacy offset bounds for tests/backward-compatible summaries.

    The cleaned DDT path no longer computes pseudo-recalibration offsets, but
    this helper is kept because existing tests and historical summaries use it.
    """

    if not offsets or all(abs(float(offset)) <= 1e-12 for offset in offsets):
        return [float(value) for value in ddt_thresholds]

    pseudo_min = float(getattr(pseudo_cfg, "min_dt", 0.0))
    recal_min = float(getattr(recalibration_cfg, "min_score_min", pseudo_min))
    lower = max(pseudo_min, recal_min)
    upper = float(getattr(pseudo_cfg, "max_dt", base_threshold))
    return [
        float(np.clip(float(threshold) - float(offset), lower, upper))
        for threshold, offset in zip(ddt_thresholds, offsets)
    ]


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
        "mean_weight_pseudo": _mean("pseudo_weight_sum", "pseudo_weight_count"),
    }


def _new_soft_query_activation_epoch_stats() -> dict[str, float]:
    return {
        "steps": 0.0,
        "candidate_queries": 0.0,
        "selected_queries": 0.0,
        "activation_score_sum": 0.0,
        "loss_targets": 0.0,
        "matched_targets": 0.0,
        "loss_weight_sum": 0.0,
        "match_iou_sum": 0.0,
        "risk_weight_sum": 0.0,
    }


def _accumulate_soft_query_activation_stats(
    epoch_stats: dict[str, float],
    *,
    selection_stats: LatentActivationSelectionStats | None,
    loss_stats: SoftQueryActivationLossStats | None,
) -> None:
    if selection_stats is None and loss_stats is None:
        return
    epoch_stats["steps"] += 1.0
    if selection_stats is not None:
        epoch_stats["candidate_queries"] += float(selection_stats.candidates)
        epoch_stats["selected_queries"] += float(selection_stats.activated)
        epoch_stats["activation_score_sum"] += float(selection_stats.score_sum)
    if loss_stats is not None:
        epoch_stats["loss_targets"] += float(loss_stats.targets)
        epoch_stats["matched_targets"] += float(loss_stats.matched)
        epoch_stats["loss_weight_sum"] += float(loss_stats.weight_sum)
        epoch_stats["match_iou_sum"] += float(loss_stats.match_iou_sum)
        epoch_stats["risk_weight_sum"] += float(loss_stats.risk_weight_sum)


def _soft_query_activation_log_fields(
    *,
    enabled: bool,
    method: str,
    objective: str,
    selection_stats: LatentActivationSelectionStats | None,
    loss_stats: SoftQueryActivationLossStats | None,
) -> dict[str, Any]:
    if not enabled:
        return {
            "soft_query_activation_enabled": False,
            "soft_query_activation_method": None,
            "soft_query_activation_objective": None,
            "soft_query_activation_candidates": None,
            "soft_query_activation_selected": None,
            "soft_query_activation_matched": None,
            "soft_query_activation_mean_score": None,
            "soft_query_activation_mean_match_iou": None,
            "soft_query_activation_mean_risk_weight": None,
        }
    selection_dict = selection_stats.as_dict() if selection_stats is not None else {}
    loss_dict = loss_stats.as_dict() if loss_stats is not None else {}
    return {
        "soft_query_activation_enabled": True,
        "soft_query_activation_method": method,
        "soft_query_activation_objective": objective,
        "soft_query_activation_candidates": int(selection_dict.get("candidates", 0)),
        "soft_query_activation_selected": int(selection_dict.get("activated", 0)),
        "soft_query_activation_matched": int(loss_dict.get("matched", 0)),
        "soft_query_activation_mean_score": selection_dict.get("mean_activation_score"),
        "soft_query_activation_mean_match_iou": loss_dict.get("mean_match_iou"),
        "soft_query_activation_mean_risk_weight": loss_dict.get("mean_risk_weight"),
    }


def _soft_query_activation_epoch_summary(
    epoch_stats: dict[str, float],
    *,
    enabled: bool,
    controller: LatentQueryActivator | None,
    method: str,
    objective: str,
    loss_weight: float,
) -> dict[str, Any]:
    selected = int(epoch_stats["selected_queries"])
    matched = int(epoch_stats["matched_targets"])
    return {
        "enabled": bool(enabled),
        "method": method if enabled else None,
        "objective": objective if enabled else None,
        "loss_weight": float(loss_weight) if enabled else None,
        "controller": controller.summary() if controller is not None else None,
        "steps": int(epoch_stats["steps"]),
        "candidate_queries": int(epoch_stats["candidate_queries"]),
        "selected_queries": selected,
        "loss_targets": int(epoch_stats["loss_targets"]),
        "matched_targets": matched,
        "mean_activation_score": float(epoch_stats["activation_score_sum"] / selected)
        if selected > 0
        else None,
        "mean_loss_weight": float(epoch_stats["loss_weight_sum"] / matched)
        if matched > 0
        else None,
        "mean_risk_weight": float(epoch_stats["risk_weight_sum"] / matched)
        if matched > 0
        else None,
        "mean_match_iou": float(epoch_stats["match_iou_sum"] / matched)
        if matched > 0
        else None,
        "selection_rate": float(epoch_stats["selected_queries"] / max(epoch_stats["candidate_queries"], 1.0))
        if enabled
        else None,
        "match_rate": float(epoch_stats["matched_targets"] / max(epoch_stats["loss_targets"], 1.0))
        if enabled
        else None,
    }


def _new_query_revival_epoch_stats() -> dict[str, float]:
    return {
        "steps": 0.0,
        "loss_targets": 0.0,
        "matched_targets": 0.0,
        "loss_weight_sum": 0.0,
        "score_sum": 0.0,
        "gate_sum": 0.0,
        "match_iou_sum": 0.0,
    }


def _accumulate_query_revival_stats(
    epoch_stats: dict[str, float],
    *,
    loss_stats: QueryRevivalLossStats | None,
) -> None:
    if loss_stats is None:
        return
    epoch_stats["steps"] += 1.0
    epoch_stats["loss_targets"] += float(loss_stats.targets)
    epoch_stats["matched_targets"] += float(loss_stats.matched)
    epoch_stats["loss_weight_sum"] += float(loss_stats.weight_sum)
    epoch_stats["score_sum"] += float(loss_stats.score_sum)
    epoch_stats["gate_sum"] += float(loss_stats.gate_sum)
    epoch_stats["match_iou_sum"] += float(loss_stats.match_iou_sum)


def _query_revival_log_fields(
    *,
    enabled: bool,
    loss_stats: QueryRevivalLossStats | None,
) -> dict[str, Any]:
    if not enabled:
        return {
            "query_revival_enabled": False,
            "query_revival_targets": None,
            "query_revival_matched": None,
            "query_revival_mean_score": None,
            "query_revival_mean_gate": None,
            "query_revival_mean_weight": None,
            "query_revival_mean_match_iou": None,
        }
    loss_dict = loss_stats.as_dict() if loss_stats is not None else {}
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
    epoch_stats: dict[str, float],
    *,
    enabled: bool,
    loss_weight: float,
    train_as: str,
    match_mode: str,
    foreground_pool: str,
) -> dict[str, Any]:
    matched = int(epoch_stats["matched_targets"])
    targets = int(epoch_stats["loss_targets"])
    return {
        "enabled": bool(enabled),
        "train_as": train_as if enabled else None,
        "loss_weight": float(loss_weight) if enabled else None,
        "match_mode": match_mode if enabled else None,
        "foreground_pool": foreground_pool if enabled else None,
        "steps": int(epoch_stats["steps"]),
        "loss_targets": targets,
        "matched_targets": matched,
        "mean_loss_weight": float(epoch_stats["loss_weight_sum"] / matched)
        if matched > 0
        else None,
        "mean_score": float(epoch_stats["score_sum"] / matched) if matched > 0 else None,
        "mean_gate": float(epoch_stats["gate_sum"] / matched) if matched > 0 else None,
        "mean_match_iou": float(epoch_stats["match_iou_sum"] / matched)
        if matched > 0
        else None,
        "match_rate": float(matched / max(targets, 1)) if enabled else None,
    }


def _soft_query_loss_ramp_multiplier(
    cfg: Any,
    *,
    global_step: int,
    total_steps: int,
    steps_per_epoch: int,
) -> float:
    if not bool(getattr(cfg, "enabled", False)):
        return 1.0
    start_multiplier = float(np.clip(float(getattr(cfg, "start_multiplier", 0.0)), 0.0, 1.0))
    start_step = _soft_query_ramp_step(
        cfg,
        step_name="start_step",
        epoch_name="start_epoch",
        fraction_name="start_fraction",
        default_fraction=0.0,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )
    end_step = _soft_query_ramp_step(
        cfg,
        step_name="end_step",
        epoch_name="end_epoch",
        fraction_name="end_fraction",
        default_fraction=1.0,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )
    if int(global_step) <= start_step:
        return start_multiplier
    if end_step <= start_step or int(global_step) >= end_step:
        return 1.0
    progress = float(int(global_step) - start_step) / max(float(end_step - start_step), 1.0)
    return float(start_multiplier + (1.0 - start_multiplier) * np.clip(progress, 0.0, 1.0))


def _soft_query_ramp_step(
    cfg: Any,
    *,
    step_name: str,
    epoch_name: str,
    fraction_name: str,
    default_fraction: float,
    total_steps: int,
    steps_per_epoch: int,
) -> int:
    if hasattr(cfg, step_name):
        return max(0, int(getattr(cfg, step_name)))
    if hasattr(cfg, epoch_name):
        return max(0, int(round((float(getattr(cfg, epoch_name)) - 1.0) * max(int(steps_per_epoch), 1))))
    fraction = float(getattr(cfg, fraction_name, default_fraction))
    return max(0, int(round(float(np.clip(fraction, 0.0, 1.0)) * max(int(total_steps), 1))))


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
                    weak_view_rng=np.random.default_rng(int(seed) + int(step_offset) * 1009 + view_idx * 9173),
                )
            )
    return merge_multiview_teacher_items(
        primary_items,
        extra_views,
        support_iou=support_iou,
    )


class DDTDAODTrainer:
    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def fit(self, *, run_dir: Path, source_checkpoint: str) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        method_cfg = getattr(self.cfg, "method", object())
        exp_name = str(getattr(method_cfg, "exp_name", "ddt_daod"))
        label_guided_hook = build_label_guided_hook(method_cfg)
        label_guided_summary = label_guided_hook.state.component_summary
        save_json(run_dir / "label_guided_components.json", label_guided_summary)
        save_json(run_dir / "label_guided_hook_state.json", label_guided_hook.state.as_dict())
        train_cfg = getattr(method_cfg, "train", object())
        pseudo_cfg = getattr(method_cfg, "pseudo", object())
        mask_cfg = getattr(method_cfg, "masking", object())
        active_cfg = getattr(method_cfg, "active", object())
        aema_cfg = getattr(method_cfg, "aema", object())
        eval_cfg = getattr(method_cfg, "eval", object())
        intermediate_eval_cfg = getattr(eval_cfg, "intermediate", object())
        label_guided_aema_cfg = getattr(method_cfg, "label_guided_aema", object())
        label_guided_aema_enabled = bool(getattr(label_guided_aema_cfg, "enabled", False))
        gradient_surgery_cfg = getattr(method_cfg, "gradient_surgery", object())
        gradient_surgery_enabled = bool(getattr(gradient_surgery_cfg, "enabled", False))
        gradient_surgery_method = str(
            getattr(gradient_surgery_cfg, "method", "target_anchored_pcgrad")
        ).strip().lower()
        gradient_surgery_apply_pseudo = bool(getattr(gradient_surgery_cfg, "apply_to_pseudo", True))
        gradient_surgery_apply_masked = bool(getattr(gradient_surgery_cfg, "apply_to_masked", False))
        gradient_surgery_eps = max(0.0, float(getattr(gradient_surgery_cfg, "eps", 1e-12)))
        oracle_pseudo_cfg = getattr(method_cfg, "oracle_pseudo", object())
        oracle_pseudo_enabled = bool(getattr(oracle_pseudo_cfg, "enabled", False))
        oracle_pseudo_mode = str(getattr(oracle_pseudo_cfg, "mode", "filter")).strip().lower()
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
        soft_query_cfg = getattr(method_cfg, "soft_query_activation", object())
        soft_query_enabled = bool(getattr(soft_query_cfg, "enabled", False))
        soft_query_method = str(getattr(soft_query_cfg, "method", "precision_rule")).strip().lower()
        soft_query_objective = str(getattr(soft_query_cfg, "objective", "class_bce")).strip().lower()
        soft_query_fit_max_images = max(0, int(getattr(soft_query_cfg, "fit_max_images", 0)))
        soft_query_loss_weight = float(getattr(soft_query_cfg, "loss_weight", 0.05))
        soft_query_match_mode = str(getattr(soft_query_cfg, "match_mode", "box_iou")).strip().lower()
        soft_query_min_match_iou = float(getattr(soft_query_cfg, "min_match_iou", 0.40))
        soft_query_match_class_aware = bool(getattr(soft_query_cfg, "match_class_aware", False))
        soft_query_positive_target = float(getattr(soft_query_cfg, "positive_target", 0.80))
        soft_query_margin = float(getattr(soft_query_cfg, "margin", 0.30))
        soft_query_weight_power = float(getattr(soft_query_cfg, "activation_weight_power", 1.0))
        soft_query_min_weight = float(getattr(soft_query_cfg, "min_activation_weight", 0.25))
        soft_query_distill_temperature = float(getattr(soft_query_cfg, "distill_temperature", 1.0))
        soft_query_distill_negative_weight = float(getattr(soft_query_cfg, "distill_negative_weight", 0.20))
        soft_query_distill_boost_selected = bool(getattr(soft_query_cfg, "distill_boost_selected", True))
        soft_query_loss_ramp_cfg = getattr(soft_query_cfg, "loss_ramp", object())
        soft_query_risk_cfg = getattr(soft_query_cfg, "query_risk", object())
        soft_query_risk_enabled = soft_query_enabled and bool(getattr(soft_query_risk_cfg, "enabled", False))
        soft_query_gate_cfg = getattr(soft_query_cfg, "class_gate", object())
        soft_query_gate_enabled = soft_query_enabled and bool(getattr(soft_query_gate_cfg, "enabled", False))
        soft_query_gate_schedule = str(getattr(soft_query_gate_cfg, "schedule", "static")).strip().lower()

        if soft_query_enabled and soft_query_method not in {"precision_rule", "reliability_model"}:
            raise ValueError(
                "method.soft_query_activation.method must be precision_rule or reliability_model, "
                f"got {soft_query_method!r}"
            )
        if soft_query_enabled and soft_query_objective not in {"class_bce", "margin", "distill_bce"}:
            raise ValueError(
                "method.soft_query_activation.objective must be one of class_bce/margin/distill_bce, "
                f"got {soft_query_objective!r}"
            )
        if soft_query_enabled and soft_query_match_mode not in {"query_index", "box_iou"}:
            raise ValueError(
                "method.soft_query_activation.match_mode must be query_index or box_iou, "
                f"got {soft_query_match_mode!r}"
            )
        if soft_query_risk_enabled and str(
            getattr(soft_query_risk_cfg, "aggregate", "weighted_mean")
        ).strip().lower() not in {"weighted_mean", "geometric"}:
            raise ValueError(
                "method.soft_query_activation.query_risk.aggregate must be weighted_mean or geometric, "
                f"got {getattr(soft_query_risk_cfg, 'aggregate')!r}"
            )
        if soft_query_gate_enabled and soft_query_gate_schedule != "static":
            raise ValueError(
                "Cleaned DDT supports only method.soft_query_activation.class_gate.schedule=static, "
                f"got {soft_query_gate_schedule!r}"
            )
        if oracle_pseudo_enabled and oracle_pseudo_mode not in {
            "filter",
            "recover",
            "recovery",
            "filter_recover",
            "filter+recover",
            "both",
            "classwise",
        }:
            raise ValueError(
                "method.oracle_pseudo.mode must be filter, recover, filter_recover, or classwise, "
                f"got {oracle_pseudo_mode!r}"
            )
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
        if query_recovery_enabled and float(getattr(query_recovery_cfg, "min_score", 0.01)) < 0.0:
            raise ValueError("method.query_recovery.min_score must be non-negative")
        if gradient_surgery_enabled and gradient_surgery_method != "target_anchored_pcgrad":
            raise ValueError(
                "Cleaned DDT currently restores target_anchored_pcgrad as the "
                "registered gradient-surgery representative; got "
                f"{gradient_surgery_method!r}"
            )
        if gradient_surgery_enabled and gradient_surgery_apply_masked:
            raise ValueError("Cleaned DDT PCGrad representative uses apply_to_masked=false.")

        seed = int(getattr(self.cfg, "seed", 42))
        teacher_device = _resolve_teacher_device(train_cfg, self.device)
        student_adapter = build_daod_model(self.cfg, load_weights=False, device=self.device)
        teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=teacher_device)
        student_model = student_adapter.model.to(self.device)
        teacher_model = teacher_adapter.model.to(teacher_device)
        DetectionCheckpointer(student_model).load(str(source_checkpoint))
        DetectionCheckpointer(teacher_model).load(str(source_checkpoint))

        target_train = materialize_daod_dicts(self.cfg, "target_train")
        target_train = _limit_samples(target_train, int(getattr(train_cfg, "max_target_samples", 0)))
        target_labeled, target_unlabeled, selected_ids, active_plan = _build_sparse_target_split(
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

        trainable_params = [parameter for parameter in student_model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(getattr(train_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(train_cfg, "weight_decay", 1e-4)),
        )
        epochs = int(getattr(method_cfg, "epochs", 2))
        steps_per_epoch = max(len(target_loader), len(labeled_loader), 1)
        total_steps = max(epochs * steps_per_epoch, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        num_classes = int(self.cfg.data.num_classes)
        class_names = tuple(get_daod_thing_classes(self.cfg))
        base_threshold = float(getattr(pseudo_cfg, "threshold", 0.4))
        thresholds = [base_threshold] * num_classes
        strong_short_edge = int(getattr(self.cfg.detector, "min_size_test", 800))
        max_size = int(getattr(self.cfg.detector, "max_size_test", 1333))
        weak_rng = np.random.default_rng(seed)
        fixed_masked_ratio = float(getattr(mask_cfg, "masked_ratio", 0.5))
        coef_masked_img = float(getattr(mask_cfg, "coef_masked_img", 1.0))
        supervised_weight = float(getattr(active_cfg, "supervised_weight", 1.0))
        alpha_ema = float(getattr(aema_cfg, "alpha_ema", 0.9996))
        alpha_aema = float(getattr(aema_cfg, "alpha_aema", 0.997))
        top_fraction = float(getattr(aema_cfg, "top_fraction", 0.1))
        update_interval = int(getattr(aema_cfg, "update_interval", 2))
        use_teacher_grad = bool(getattr(aema_cfg, "use_teacher_grad_importance", True))
        label_guided_aema_merge = str(getattr(label_guided_aema_cfg, "merge", "max")).strip().lower()
        label_guided_aema_weight = float(getattr(label_guided_aema_cfg, "guidance_weight", 1.0))
        label_guided_aema_normalize = bool(getattr(label_guided_aema_cfg, "normalize", True))
        label_guided_aema_loss_mode = str(getattr(label_guided_aema_cfg, "loss", "full")).strip().lower()
        label_guided_aema_loss_weight = float(getattr(label_guided_aema_cfg, "supervised_loss_weight", 1.0))
        label_guided_aema_top_fraction = float(getattr(label_guided_aema_cfg, "top_fraction", top_fraction))
        if label_guided_aema_merge not in {"max", "add", "gt_only", "base_only"}:
            raise ValueError(
                "method.label_guided_aema.merge must be one of max/add/gt_only/base_only, "
                f"got {label_guided_aema_merge!r}"
            )
        if label_guided_aema_loss_mode not in {"full", "class"}:
            raise ValueError(
                "method.label_guided_aema.loss must be 'full' or 'class', "
                f"got {label_guided_aema_loss_mode!r}"
            )
        use_dynamic_threshold = bool(getattr(pseudo_cfg, "dynamic", True))
        dynamic_empty_policy = str(getattr(pseudo_cfg, "empty_policy", "keep")).strip().lower()
        if dynamic_empty_policy not in {"keep", "official"}:
            raise ValueError(
                "method.pseudo.empty_policy must be 'keep' or 'official', "
                f"got {dynamic_empty_policy!r}"
            )
        log_period = int(getattr(train_cfg, "log_period", 100))
        checkpoint_period = int(getattr(train_cfg, "checkpoint_period", 0))
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
                "DDT intermediate eval is student-only for early stopping; "
                "set method.eval.intermediate.model=student, got "
                f"got {intermediate_eval_model!r}"
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
        intermediate_eval_limit = int(
            getattr(
                intermediate_eval_cfg,
                "target_val_limit",
                getattr(eval_cfg, "intermediate_target_val_limit", getattr(eval_cfg, "target_val_limit", 0)),
            )
        )
        train_log_path = run_dir / "train_log.jsonl"

        if label_guided_hook_requires_teacher_fit(method_cfg):
            label_cfg = getattr(method_cfg, "label_guided", object())
            fit_max_images = max(0, int(getattr(label_cfg, "fit_max_images", 0)))
            fit_items = list(target_labeled)
            if fit_max_images > 0:
                fit_items = fit_items[:fit_max_images]
            print(
                "[DDT-DAOD][label_guided][fit] "
                f"method={getattr(label_cfg, 'method', 'unknown')} "
                f"labeled_images={len(fit_items)}",
                flush=True,
            )
            fit_teacher_items: list[dict[str, Any]] = []
            if fit_items:
                with _device_context(teacher_device), torch.no_grad():
                    fit_teacher_items = _teacher_outputs_for_unlabeled(
                        teacher_adapter,
                        fit_items,
                        weak_view_rng=np.random.default_rng(seed + 4242),
                    )
            label_guided_hook = build_label_guided_hook(
                method_cfg,
                fit_teacher_items=fit_teacher_items,
                base_thresholds=thresholds,
                num_classes=num_classes,
            )
            label_guided_summary = label_guided_hook.state.component_summary
            save_json(run_dir / "label_guided_components.json", label_guided_summary)
            save_json(run_dir / "label_guided_hook_state.json", label_guided_hook.state.as_dict())
            fitted_name = next(iter(label_guided_hook.state.step_stats.keys()), "none")
            calibration_state = label_guided_hook.state.step_stats.get(fitted_name, {})
            print(
                "[DDT-DAOD][label_guided][fit] "
                f"state={fitted_name} "
                f"adjusted_classes={calibration_state.get('adjusted_classes', [])} "
                f"mean_abs_offset={float(calibration_state.get('aggregate', {}).get('mean_abs_offset', 0.0)):.4f}",
                flush=True,
            )

        query_recovery_scorer: QueryRecoveryScorer | None = None
        query_recovery_stats: dict[str, Any] = {}
        if query_recovery_enabled:
            fit_items = list(target_labeled)
            if query_recovery_fit_max_images > 0:
                fit_items = fit_items[:query_recovery_fit_max_images]
            num_views = _query_recovery_num_views(query_recovery_cfg)
            query_recovery_cfg["_resolved_num_views"] = int(num_views)
            print(
                "[DDT-DAOD][query_recovery][fit] "
                f"labeled_images={len(fit_items)} views={num_views} "
                f"min_score={float(getattr(query_recovery_cfg, 'min_score', 0.01)):.3f} "
                f"precision_floor={float(getattr(query_recovery_cfg, 'precision_floor', 0.55)):.3f}",
                flush=True,
            )
            if fit_items:
                with _device_context(teacher_device), torch.no_grad():
                    primary_fit_items = _teacher_outputs_for_unlabeled(
                        teacher_adapter,
                        fit_items,
                        weak_view_rng=np.random.default_rng(seed + 8181),
                    )
                recovery_fit_items = _build_query_recovery_teacher_items(
                    teacher_adapter=teacher_adapter,
                    primary_items=primary_fit_items,
                    source_batch=fit_items,
                    recovery_cfg=query_recovery_cfg,
                    teacher_device=teacher_device,
                    seed=seed + 9191,
                    step_offset=0,
                )
                query_recovery_scorer = fit_query_recovery_scorer(
                    recovery_fit_items,
                    thresholds=thresholds,
                    num_classes=num_classes,
                    recovery_cfg=query_recovery_cfg,
                    seed=seed,
                    dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                )
                query_recovery_stats = query_recovery_scorer.summary()
            else:
                query_recovery_stats = {
                    "enabled": True,
                    "fit_images": 0,
                    "fit_candidates": 0,
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
                        "foreground_temperature": float(query_revival_foreground_temperature)
                        if query_revival_enabled
                        else None,
                        "recovery_weight_power": float(query_revival_weight_power) if query_revival_enabled else None,
                        "min_candidate_weight": float(query_revival_min_weight) if query_revival_enabled else None,
                    },
                    "multi_view": {
                        "enabled": bool(getattr(getattr(query_recovery_cfg, "multi_view", object()), "enabled", False)),
                        "views": int(num_views),
                        "support_iou": float(
                            getattr(getattr(query_recovery_cfg, "multi_view", object()), "support_iou", 0.5)
                        ),
                    },
                }
            )
            save_json(run_dir / "query_recovery.json", query_recovery_stats)

        soft_query_activator: LatentQueryActivator | None = None
        soft_query_stats: dict[str, Any] = {}
        soft_query_class_gate: BenefitRiskClassGate | None = None
        soft_query_gate_history: list[dict[str, Any]] = []
        if soft_query_enabled:
            fit_items = list(target_labeled)
            if soft_query_fit_max_images > 0:
                fit_items = fit_items[:soft_query_fit_max_images]
            print(
                "[DDT-DAOD][soft_query_activation][fit] "
                f"method={soft_query_method} objective={soft_query_objective} "
                f"labeled_images={len(fit_items)} "
                f"min_score={float(getattr(soft_query_cfg, 'min_score', 0.02)):.3f} "
                f"precision_target={float(getattr(soft_query_cfg, 'precision_target', 0.95)):.3f}",
                flush=True,
            )
            if fit_items:
                with _device_context(teacher_device), torch.no_grad():
                    fit_teacher_items = _teacher_outputs_for_unlabeled(
                        teacher_adapter,
                        fit_items,
                        weak_view_rng=np.random.default_rng(seed + 6161),
                    )
                soft_query_activator = fit_latent_query_activator(
                    fit_teacher_items,
                    thresholds=thresholds,
                    num_classes=num_classes,
                    activation_cfg=soft_query_cfg,
                    seed=seed,
                )
                soft_query_stats = soft_query_activator.summary()
                if soft_query_gate_enabled:
                    soft_query_class_gate = fit_benefit_risk_class_gate(
                        fit_teacher_items,
                        activator=soft_query_activator,
                        thresholds=thresholds,
                        num_classes=num_classes,
                        gate_cfg=soft_query_gate_cfg,
                        dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                    )
                    gate_summary = {
                        **soft_query_class_gate.as_dict(),
                        "phase": "static",
                        "epoch": 0,
                        "step": 0,
                        "schedule": "static",
                        "thresholds": [float(value) for value in thresholds],
                    }
                    soft_query_gate_history.append(gate_summary)
                    save_json(run_dir / "soft_query_class_gate.json", gate_summary)
                    append_jsonl(run_dir / "soft_query_class_gate_log.jsonl", gate_summary)
            else:
                soft_query_stats = {
                    "enabled": True,
                    "method": soft_query_method,
                    "objective": soft_query_objective,
                    "fit_images": 0,
                    "fit_candidates": 0,
                    "reason": "no_labeled_target_images",
                }
            soft_query_stats.update(
                {
                    "objective": soft_query_objective,
                    "loss_weight": float(soft_query_loss_weight),
                    "match_mode": soft_query_match_mode,
                    "loss_ramp": {
                        "enabled": bool(getattr(soft_query_loss_ramp_cfg, "enabled", False)),
                        "start_fraction": getattr(soft_query_loss_ramp_cfg, "start_fraction", None),
                        "end_fraction": getattr(soft_query_loss_ramp_cfg, "end_fraction", None),
                        "start_multiplier": getattr(soft_query_loss_ramp_cfg, "start_multiplier", None),
                    },
                    "query_risk": {
                        "enabled": bool(soft_query_risk_enabled),
                        "aggregate": getattr(soft_query_risk_cfg, "aggregate", None)
                        if soft_query_risk_enabled
                        else None,
                        "min_weight": getattr(soft_query_risk_cfg, "min_weight", None)
                        if soft_query_risk_enabled
                        else None,
                        "power": getattr(soft_query_risk_cfg, "power", None) if soft_query_risk_enabled else None,
                    },
                }
            )
            save_json(run_dir / "soft_query_activation.json", soft_query_stats)

        if oracle_pseudo_enabled:
            save_json(
                run_dir / "oracle_pseudo.json",
                {
                    "enabled": True,
                    "mode": oracle_pseudo_mode,
                    "match_iou": float(getattr(oracle_pseudo_cfg, "match_iou", 0.5)),
                    "recovery_score": float(
                        getattr(oracle_pseudo_cfg, "recovery_score", getattr(oracle_pseudo_cfg, "score", 1.0))
                    ),
                    "classes": getattr(oracle_pseudo_cfg, "classes", None),
                    "default_policy": getattr(oracle_pseudo_cfg, "default_policy", None),
                    "policies": getattr(oracle_pseudo_cfg, "policies", {}),
                    "note": "Oracle diagnostic: uses hidden target GT on unlabeled images.",
                },
            )

        print(
            "[DDT-DAOD][train] "
            f"exp={exp_name} "
            f"epochs={epochs} target_train={len(target_train)} "
            f"labeled_target={len(target_labeled)} unlabeled_target={len(target_unlabeled)} "
            f"student_device={self.device} teacher_device={teacher_device} "
            f"source_ckpt={source_checkpoint}",
            flush=True,
        )
        print(
            "[DDT-DAOD][ddt] "
            f"threshold={thresholds[0]:.3f} dynamic={use_dynamic_threshold} "
            f"mask_ratio={fixed_masked_ratio:.3f} "
            f"alpha_ema={alpha_ema:.6f} alpha_aema={alpha_aema:.6f}",
            flush=True,
        )
        if label_guided_summary["enabled_component_names"]:
            print(
                "[DDT-DAOD][label_guided] "
                f"components={','.join(label_guided_summary['enabled_component_names'])} "
                f"categories={','.join(label_guided_summary['enabled_categories'])} "
                f"legacy={','.join(label_guided_summary['legacy_live_component_names']) or 'none'}",
                flush=True,
            )
        if gradient_surgery_enabled:
            print(
                "[DDT-DAOD][gradient_surgery] "
                f"method={gradient_surgery_method} apply_to_pseudo={gradient_surgery_apply_pseudo} "
                f"apply_to_masked={gradient_surgery_apply_masked}",
                flush=True,
            )
        if label_guided_aema_enabled:
            print(
                "[DDT-DAOD][label_guided_aema] "
                f"merge={label_guided_aema_merge} guidance_weight={label_guided_aema_weight:.3f} "
                f"normalize={label_guided_aema_normalize} loss={label_guided_aema_loss_mode} "
                f"top_fraction={label_guided_aema_top_fraction:.3f}",
                flush=True,
            )
        if oracle_pseudo_enabled:
            print(
                "[DDT-DAOD][oracle_pseudo] "
                f"mode={oracle_pseudo_mode} "
                f"match_iou={float(getattr(oracle_pseudo_cfg, 'match_iou', 0.5)):.3f} "
                f"recovery_score={float(getattr(oracle_pseudo_cfg, 'recovery_score', getattr(oracle_pseudo_cfg, 'score', 1.0))):.3f}",
                flush=True,
            )
        if query_recovery_enabled:
            print(
                "[DDT-DAOD][query_recovery] "
                f"train_as={query_recovery_train_as} "
                f"fit_candidates={query_recovery_stats.get('fit_candidates', 0)} "
                f"fit_positive={query_recovery_stats.get('fit_positive', 0)} "
                f"global_threshold={query_recovery_stats.get('global_threshold', query_recovery_stats.get('global_threshold_stats', {}).get('threshold'))} "
                f"views={query_recovery_stats.get('multi_view', {}).get('views', 1)}",
                flush=True,
            )
        if soft_query_enabled:
            print(
                "[DDT-DAOD][soft_query_activation] "
                f"method={soft_query_method} objective={soft_query_objective} "
                f"loss_weight={soft_query_loss_weight:.3f} "
                f"ramp={bool(getattr(soft_query_loss_ramp_cfg, 'enabled', False))} "
                f"query_risk={bool(soft_query_risk_enabled)} "
                f"class_gate={bool(soft_query_gate_enabled)} "
                f"fit_candidates={soft_query_stats.get('fit_candidates', 0)}",
                flush=True,
            )
        if intermediate_eval_enabled:
            print(
                "[DDT-DAOD][intermediate_eval] "
                f"exp={exp_name} enabled model={intermediate_eval_model} interval_steps={intermediate_eval_interval_steps}"
                f" target_val_limit={intermediate_eval_limit} early_stop_mode={early_stop_mode}"
                f" patience={early_stop_patience} min_delta={early_stop_min_delta:.4f}",
                flush=True,
            )

        global_step = 0
        history: list[dict[str, Any]] = []
        intermediate_eval_history: list[dict[str, Any]] = []
        intermediate_target_val: list[dict[str, Any]] | None = None
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
        for epoch_idx in range(1, epochs + 1):
            student_model.train()
            teacher_model.eval()
            epoch_loss = 0.0
            epoch_loss_pseudo = 0.0
            epoch_loss_mask = 0.0
            epoch_loss_soft_query = 0.0
            epoch_loss_query_revival = 0.0
            epoch_loss_supervised = 0.0
            epoch_pseudo_boxes = 0
            epoch_pseudo_importance_steps = 0
            epoch_gt_importance_steps = 0
            epoch_label_guided_teacher_loss = 0.0
            epoch_label_guided_teacher_steps = 0
            epoch_aema_update_steps = 0
            epoch_ema_update_steps = 0
            epoch_gradient_surgery_stats = _new_gradient_surgery_epoch_stats()
            epoch_soft_query_stats = _new_soft_query_activation_epoch_stats()
            epoch_query_revival_stats = _new_query_revival_epoch_stats()
            epoch_query_recovery_stats = QueryRecoverySelectionStats()
            epoch_oracle_pseudo_stats = OraclePseudoStats(num_classes)
            score_sums = [0.0] * num_classes
            score_counts = [0] * num_classes
            epoch_steps_completed = 0

            for _ in range(steps_per_epoch):
                batch = next(target_iter, [])
                labeled_batch = next(labeled_iter, [])
                effective_thresholds = label_guided_hook.adjust_thresholds(
                    [float(value) for value in thresholds],
                    global_step=global_step,
                )
                soft_query_selection_stats = LatentActivationSelectionStats()
                soft_query_loss_stats = SoftQueryActivationLossStats()
                soft_query_items: list[dict[str, Any]] = []
                query_revival_loss_stats = QueryRevivalLossStats()
                query_revival_items: list[dict[str, Any]] = []
                pseudo_batch: list[dict[str, Any]] = []
                pseudo_box_count = 0
                step_query_recovery_stats = QueryRecoverySelectionStats()
                step_oracle_pseudo_stats = OraclePseudoStats(num_classes)

                if batch:
                    with _device_context(teacher_device), torch.no_grad():
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
                        query_recovery_items = _build_query_recovery_teacher_items(
                            teacher_adapter=teacher_adapter,
                            primary_items=teacher_items,
                            source_batch=batch,
                            recovery_cfg=query_recovery_cfg,
                            teacher_device=teacher_device,
                            seed=seed + 31337,
                            step_offset=global_step,
                        )
                    for item_index, teacher_item in enumerate(teacher_items):
                        pseudo_rows = filter_pseudo_rows(
                            teacher_item["query_rows"],
                            thresholds=effective_thresholds,
                            dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                        )
                        threshold_rows = pseudo_rows
                        if oracle_pseudo_enabled:
                            oracle_result = apply_oracle_pseudo_intervention(
                                sample=teacher_item["sample"],
                                pseudo_rows=pseudo_rows,
                                cfg=oracle_pseudo_cfg,
                                num_classes=num_classes,
                                class_names=class_names,
                            )
                            pseudo_rows = oracle_result.rows
                            threshold_rows = oracle_result.threshold_rows
                            step_oracle_pseudo_stats.add(oracle_result.stats)

                        if query_recovery_scorer is not None and query_recovery_items is not None:
                            recovery_item = query_recovery_items[item_index]
                            recovered_rows, item_recovery_stats = query_recovery_scorer.select(
                                recovery_item["query_rows"],
                                thresholds=effective_thresholds,
                                dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                                sample=recovery_item["sample"],
                                existing_rows=pseudo_rows,
                            )
                            if recovered_rows:
                                if query_recovery_train_as == "hard_pseudo":
                                    pseudo_rows = [*pseudo_rows, *recovered_rows]
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

                        for row in threshold_rows:
                            class_id = int(row["category_id"])
                            score_sums[class_id] += float(row["score"])
                            score_counts[class_id] += 1

                        if soft_query_activator is not None:
                            soft_rows, item_soft_stats = soft_query_activator.select(
                                teacher_item["query_rows"],
                                thresholds=effective_thresholds,
                                dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                                sample=teacher_item["sample"],
                                existing_rows=pseudo_rows,
                            )
                            soft_query_selection_stats.candidates += item_soft_stats.candidates
                            soft_query_selection_stats.activated += item_soft_stats.activated
                            soft_query_selection_stats.score_sum += item_soft_stats.score_sum
                            enriched_soft_rows = []
                            for row in soft_rows:
                                class_id = int(row.get("category_id", -1))
                                if soft_query_class_gate is not None and not (
                                    0 <= class_id < len(soft_query_class_gate.gates)
                                    and float(soft_query_class_gate.gates[class_id]) > 0.0
                                ):
                                    continue
                                enriched_row = dict(row)
                                if 0 <= class_id < len(effective_thresholds):
                                    enriched_row["_pseudo_threshold"] = float(effective_thresholds[class_id])
                                enriched_soft_rows.append(enriched_row)
                            if enriched_soft_rows:
                                soft_query_items.append(
                                    {
                                        "sample": teacher_item["sample"],
                                        "teacher_raw": teacher_item["raw_output"],
                                        "teacher_rows": enriched_soft_rows,
                                    }
                                )

                        annotations = rows_to_annotations(pseudo_rows)
                        if not annotations:
                            continue
                        pseudo_sample = dict(teacher_item["sample"])
                        pseudo_sample["annotations"] = annotations
                        pseudo_batch.append(pseudo_sample)
                        pseudo_box_count += len(annotations)

                if oracle_pseudo_enabled:
                    epoch_oracle_pseudo_stats.add(step_oracle_pseudo_stats)
                if query_recovery_enabled:
                    epoch_query_recovery_stats.add(step_query_recovery_stats)

                loss_terms: list[torch.Tensor] = []
                loss_pseudo_value = 0.0
                loss_mask_value = 0.0
                loss_soft_query_value = 0.0
                loss_query_revival_value = 0.0
                loss_supervised_value = 0.0
                core_loss_scales = {"pseudo": 1.0, "masked": 1.0, "supervised": 1.0}
                loss_pseudo: torch.Tensor | None = None
                loss_mask: torch.Tensor | None = None
                loss_soft_query: torch.Tensor | None = None
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
                    )
                    loss_pseudo = sum(student_model(pseudo_inputs).values())
                    loss_pseudo_value = float(loss_pseudo.detach().cpu())

                    masked_inputs = apply_block_mask_to_inputs(
                        pseudo_inputs,
                        block_size=int(getattr(mask_cfg, "block_size", 64)),
                        masked_ratio=fixed_masked_ratio,
                    )
                    loss_mask = float(coef_masked_img) * sum(student_model(masked_inputs).values())
                    loss_mask_value = float(loss_mask.detach().cpu())

                if soft_query_items:
                    soft_query_strong_rng = random.Random(seed + 717171 + global_step)
                    soft_query_loss_items: list[dict[str, Any]] = []
                    with _device_context(self.device):
                        for soft_item in soft_query_items:
                            strong_sample = build_strong_view_sample(
                                soft_item["sample"],
                                rng=soft_query_strong_rng,
                                suffix="student_soft_query",
                            )
                            student_raw = run_daod_raw_outputs(
                                student_adapter,
                                strong_sample,
                                with_grad=True,
                            )[0]
                            soft_query_loss_items.append({**soft_item, "student_raw": student_raw})
                    soft_query_current_weight = soft_query_loss_weight * _soft_query_loss_ramp_multiplier(
                        soft_query_loss_ramp_cfg,
                        global_step=global_step,
                        total_steps=total_steps,
                        steps_per_epoch=steps_per_epoch,
                    )
                    loss_soft_query, soft_query_loss_stats = soft_query_activation_loss(
                        soft_query_loss_items,
                        objective=soft_query_objective,
                        loss_weight=soft_query_current_weight,
                        match_mode=soft_query_match_mode,
                        min_match_iou=soft_query_min_match_iou,
                        match_class_aware=soft_query_match_class_aware,
                        positive_target=soft_query_positive_target,
                        margin=soft_query_margin,
                        activation_weight_power=soft_query_weight_power,
                        min_activation_weight=soft_query_min_weight,
                        distill_temperature=soft_query_distill_temperature,
                        distill_negative_weight=soft_query_distill_negative_weight,
                        distill_boost_selected=soft_query_distill_boost_selected,
                        class_gates=soft_query_class_gate.gates if soft_query_class_gate is not None else None,
                        class_budgets=soft_query_class_gate.budgets if soft_query_class_gate is not None else None,
                        query_risk_cfg=soft_query_risk_cfg if soft_query_risk_enabled else None,
                    )
                    if loss_soft_query.requires_grad:
                        loss_terms.append(loss_soft_query)
                        loss_soft_query_value = float(loss_soft_query.detach().cpu())

                if query_revival_items and not defer_query_revival_backward:
                    (
                        loss_query_revival,
                        query_revival_loss_stats,
                        loss_query_revival_value,
                    ) = compute_query_revival_loss_for_step()
                    if loss_query_revival.requires_grad:
                        loss_terms.append(loss_query_revival)

                if labeled_batch:
                    supervised_inputs = _make_supervised_inputs(
                        student_adapter,
                        labeled_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                    )
                    loss_supervised = float(supervised_weight) * sum(student_model(supervised_inputs).values())
                    loss_supervised_value = float(loss_supervised.detach().cpu())

                if loss_pseudo is not None or loss_mask is not None or loss_supervised is not None:
                    core_loss_scales = label_guided_hook.loss_scales(
                        {
                            "pseudo": loss_pseudo_value if loss_pseudo is not None else 0.0,
                            "masked": loss_mask_value if loss_mask is not None else 0.0,
                            "supervised": loss_supervised_value if loss_supervised is not None else 0.0,
                        },
                        global_step=global_step,
                    )
                    if loss_pseudo is not None:
                        loss_pseudo = float(core_loss_scales.get("pseudo", 1.0)) * loss_pseudo
                        loss_terms.append(loss_pseudo)
                        loss_pseudo_value = float(loss_pseudo.detach().cpu())
                    if loss_mask is not None:
                        loss_mask = float(core_loss_scales.get("masked", 1.0)) * loss_mask
                        loss_terms.append(loss_mask)
                        loss_mask_value = float(loss_mask.detach().cpu())
                    if loss_supervised is not None:
                        loss_supervised = float(core_loss_scales.get("supervised", 1.0)) * loss_supervised
                        loss_terms.append(loss_supervised)
                        loss_supervised_value = float(loss_supervised.detach().cpu())

                _accumulate_soft_query_activation_stats(
                    epoch_soft_query_stats,
                    selection_stats=soft_query_selection_stats if soft_query_enabled else None,
                    loss_stats=soft_query_loss_stats if soft_query_enabled else None,
                )

                has_deferred_query_revival = bool(query_revival_items) and bool(defer_query_revival_backward)
                if not loss_terms and not has_deferred_query_revival:
                    global_step += 1
                    scheduler.step()
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
                    for untouched_loss in (
                        loss_mask,
                        loss_soft_query,
                        loss_query_revival,
                    ):
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
                        # DDT already holds pseudo and masked-consistency graphs. Build
                        # the qrev graph only after those gradients are accumulated.
                        loss_terms = []
                        loss_pseudo = None
                        loss_mask = None
                        loss_supervised = None
                        loss_soft_query = None
                        (
                            loss_query_revival,
                            query_revival_loss_stats,
                            loss_query_revival_value,
                        ) = compute_query_revival_loss_for_step()
                        if loss_query_revival is not None and loss_query_revival.requires_grad:
                            loss_query_revival.backward()
                            loss = loss.detach() + loss_query_revival.detach()
                            did_backward = True
                _accumulate_query_revival_stats(
                    epoch_query_revival_stats,
                    loss_stats=query_revival_loss_stats if query_revival_enabled else None,
                )
                if not did_backward:
                    global_step += 1
                    scheduler.step()
                    continue
                if float(getattr(train_cfg, "clip_max_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        float(getattr(train_cfg, "clip_max_norm", 0.0)),
                    )
                optimizer.step()

                pseudo_importance: dict[str, torch.Tensor] = {}
                gt_importance: dict[str, torch.Tensor] = {}
                grad_importance: dict[str, torch.Tensor] = {}
                label_guided_teacher_loss_value = 0.0
                teacher_update_due = update_interval > 0 and global_step % update_interval == 0
                if teacher_update_due and use_teacher_grad and pseudo_batch:
                    pseudo_importance, _ = _teacher_grad_importance_from_supervised_batch(
                        teacher_model=teacher_model,
                        teacher_adapter=teacher_adapter,
                        batch=pseudo_batch,
                        teacher_device=teacher_device,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        loss_mode="class",
                    )
                    if pseudo_importance:
                        epoch_pseudo_importance_steps += 1
                    grad_importance = pseudo_importance

                if teacher_update_due and label_guided_aema_enabled and labeled_batch:
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
                    if gt_importance:
                        epoch_gt_importance_steps += 1
                        epoch_label_guided_teacher_loss += float(label_guided_teacher_loss_value)
                        epoch_label_guided_teacher_steps += 1
                    grad_importance = merge_importance_maps(
                        pseudo_importance,
                        gt_importance,
                        merge=label_guided_aema_merge,
                        guidance_weight=label_guided_aema_weight,
                        normalize=label_guided_aema_normalize,
                    )

                if teacher_update_due:
                    if grad_importance:
                        _update_aema(
                            teacher_model,
                            student_model,
                            grad_importance,
                            momentum=alpha_ema,
                            adaptive_momentum=alpha_aema,
                            top_fraction=label_guided_aema_top_fraction if label_guided_aema_enabled else top_fraction,
                        )
                        epoch_aema_update_steps += 1
                    else:
                        _update_ema(teacher_model, student_model, alpha_ema)
                        epoch_ema_update_steps += 1

                global_step += 1
                epoch_steps_completed += 1
                scheduler.step()
                epoch_loss += float(loss.detach().cpu())
                epoch_loss_pseudo += loss_pseudo_value
                epoch_loss_mask += loss_mask_value
                epoch_loss_soft_query += loss_soft_query_value
                epoch_loss_query_revival += loss_query_revival_value
                epoch_loss_supervised += loss_supervised_value
                epoch_pseudo_boxes += int(pseudo_box_count)

                if intermediate_eval_enabled and global_step % intermediate_eval_interval_steps == 0:
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
                            model_name="student",
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
                                    "[DDT-DAOD][best_student] "
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
                            save_json(
                                run_dir / "best_student_target_val_metrics.json",
                                best_student_metrics,
                            )
                            save_json(
                                run_dir / "last_student_eval.json",
                                eval_record_compact,
                            )
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
                                    "[DDT-DAOD][early_stop] "
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
                            "[DDT-DAOD][intermediate_eval][warning] "
                            f"exp={exp_name} epoch={epoch_idx} step={global_step} "
                            f"model=student failed: {exc}",
                            flush=True,
                        )
                    student_model.train()
                    teacher_model.eval()
                    eval_record_compact = _compact_intermediate_eval_record(eval_record)
                    intermediate_eval_history.append(eval_record_compact)
                    append_jsonl(run_dir / "intermediate_eval_log.jsonl", eval_record_compact)
                    if early_stop_triggered:
                        print(
                            "[DDT-DAOD][early_stop] "
                            f"exp={exp_name} exiting_train_loop_at_step={global_step}",
                            flush=True,
                        )
                        break

                if log_period > 0 and global_step % log_period == 0:
                    soft_msg = ""
                    if soft_query_enabled:
                        soft_msg = (
                            f" softq={soft_query_loss_stats.matched}"
                            f"/{soft_query_selection_stats.activated}"
                        )
                    recovery_msg = ""
                    if query_recovery_enabled:
                        recovery_msg = (
                            f" qrec={step_query_recovery_stats.selected}"
                            f"/{step_query_recovery_stats.candidates}"
                        )
                    revival_msg = ""
                    if query_revival_enabled:
                        revival_msg = (
                            f" qrev={query_revival_loss_stats.matched}"
                            f"/{query_revival_loss_stats.targets}"
                        )
                    oracle_msg = ""
                    if oracle_pseudo_enabled:
                        oracle_msg = (
                            f" oracle_drop={step_oracle_pseudo_stats.dropped}"
                            f" oracle_rec={step_oracle_pseudo_stats.recovered}"
                        )
                    print(
                        "[DDT-DAOD][step] "
                        f"exp={exp_name} epoch={epoch_idx} step={global_step} "
                        f"loss={float(loss.detach().cpu()):.3f} "
                        f"pseudo={loss_pseudo_value:.3f} masked={loss_mask_value:.3f} "
                        f"softq={loss_soft_query_value:.3f} qrev={loss_query_revival_value:.3f} "
                        f"supervised={loss_supervised_value:.3f} "
                        f"pseudo_boxes={int(pseudo_box_count)}{soft_msg}{recovery_msg}{revival_msg}{oracle_msg}",
                        flush=True,
                    )
                    query_recovery_log_fields: dict[str, Any] = {}
                    if query_recovery_enabled:
                        query_recovery_log_fields = {
                            "query_recovery_enabled": True,
                            "query_recovery_candidates": int(step_query_recovery_stats.candidates),
                            "query_recovery_selected": int(step_query_recovery_stats.selected),
                            "query_recovery_mean_score": step_query_recovery_stats.as_dict().get("mean_score"),
                        }
                    oracle_log_fields: dict[str, Any] = {}
                    if oracle_pseudo_enabled:
                        oracle_log_fields = {
                            "oracle_pseudo_enabled": True,
                            "oracle_pseudo_mode": oracle_pseudo_mode,
                            "oracle_pseudo_input": int(step_oracle_pseudo_stats.input_pseudo),
                            "oracle_pseudo_kept": int(step_oracle_pseudo_stats.kept),
                            "oracle_pseudo_dropped": int(step_oracle_pseudo_stats.dropped),
                            "oracle_pseudo_recovered": int(step_oracle_pseudo_stats.recovered),
                            "oracle_pseudo_output": int(step_oracle_pseudo_stats.output_pseudo),
                        }
                    append_jsonl(
                        train_log_path,
                        {
                            "epoch": int(epoch_idx),
                            "step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_total": float(loss.detach().cpu()),
                            "loss_pseudo": loss_pseudo_value,
                            "loss_masked": loss_mask_value,
                            "loss_soft_query": loss_soft_query_value,
                            "loss_query_revival": loss_query_revival_value,
                            "loss_supervised": loss_supervised_value,
                            "loss_scales": {
                                key: float(value) for key, value in core_loss_scales.items()
                            },
                            "pseudo_box_count": int(pseudo_box_count),
                            "ddt_thresholds": [float(v) for v in thresholds],
                            "effective_thresholds": [float(v) for v in effective_thresholds],
                            **query_recovery_log_fields,
                            **oracle_log_fields,
                            **_soft_query_activation_log_fields(
                                enabled=soft_query_enabled,
                                method=soft_query_method,
                                objective=soft_query_objective,
                                selection_stats=soft_query_selection_stats if soft_query_enabled else None,
                                loss_stats=soft_query_loss_stats if soft_query_enabled else None,
                            ),
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
                            "teacher_importance_stats": importance_map_stats(grad_importance)
                            if grad_importance
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
                    DetectionCheckpointer(teacher_model, save_dir=str(run_dir)).save(f"teacher_step_{global_step}")

            if use_dynamic_threshold:
                thresholds = update_dynamic_thresholds(
                    thresholds,
                    score_sums,
                    score_counts,
                    alpha_dt=float(getattr(pseudo_cfg, "alpha_dt", 0.5)),
                    gamma_dt=float(getattr(pseudo_cfg, "gamma_dt", 0.9)),
                    max_dt=float(getattr(pseudo_cfg, "max_dt", 0.45)),
                    min_dt=float(getattr(pseudo_cfg, "min_dt", 0.25)),
                    empty_policy=dynamic_empty_policy,
                )
                print(f"[DDT-DAOD][thresholds] epoch={epoch_idx} values={thresholds}", flush=True)

            denom = max(epoch_steps_completed, 1)
            epoch_effective_thresholds = label_guided_hook.adjust_thresholds(
                [float(value) for value in thresholds],
                global_step=global_step,
            )
            epoch_summary = {
                "epoch": float(epoch_idx),
                "loss_total": epoch_loss / denom,
                "loss_pseudo": epoch_loss_pseudo / denom,
                "loss_masked": epoch_loss_mask / denom,
                "loss_soft_query": epoch_loss_soft_query / denom,
                "loss_query_revival": epoch_loss_query_revival / denom,
                "loss_supervised": epoch_loss_supervised / denom,
                "pseudo_boxes": float(epoch_pseudo_boxes),
                "ddt_thresholds": [float(v) for v in thresholds],
                "effective_thresholds": [float(v) for v in epoch_effective_thresholds],
                "label_guided_hook": label_guided_hook.state.as_dict(),
                "teacher_update": {
                    "pseudo_importance_steps": int(epoch_pseudo_importance_steps),
                    "gt_importance_steps": int(epoch_gt_importance_steps),
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
                "oracle_pseudo": {
                    "enabled": bool(oracle_pseudo_enabled),
                    "mode": oracle_pseudo_mode if oracle_pseudo_enabled else None,
                    "match_iou": float(getattr(oracle_pseudo_cfg, "match_iou", 0.5))
                    if oracle_pseudo_enabled
                    else None,
                    "stats": epoch_oracle_pseudo_stats.as_dict(class_names=class_names)
                    if oracle_pseudo_enabled
                    else None,
                },
                "soft_query_activation": _soft_query_activation_epoch_summary(
                    epoch_soft_query_stats,
                    enabled=soft_query_enabled,
                    controller=soft_query_activator,
                    method=soft_query_method,
                    objective=soft_query_objective,
                    loss_weight=soft_query_loss_weight,
                ),
                "soft_query_class_gate": {
                    "enabled": bool(soft_query_gate_enabled),
                    "schedule": "static" if soft_query_gate_enabled else None,
                    "update_step": 0 if soft_query_class_gate is not None else None,
                    "state": soft_query_class_gate.as_dict()
                    if soft_query_class_gate is not None and soft_query_gate_enabled
                    else None,
                },
            }
            if intermediate_eval_history:
                epoch_summary["latest_intermediate_eval"] = intermediate_eval_history[-1]
            history.append(epoch_summary)
            append_jsonl(run_dir / "epoch_log.jsonl", epoch_summary)
            if early_stop_triggered:
                break

        student_ckpt = run_dir / "student_last.pth"
        teacher_ckpt = run_dir / "teacher_last.pth"
        DetectionCheckpointer(student_model, save_dir=str(run_dir)).save("student_last")
        DetectionCheckpointer(teacher_model, save_dir=str(run_dir)).save("teacher_last")

        target_val = _limit_samples(materialize_daod_dicts(self.cfg, "target_val"), getattr(eval_cfg, "target_val_limit", 0))
        evaluate_teacher = bool(getattr(eval_cfg, "evaluate_teacher", True))
        teacher_source_metrics: dict[str, Any] = {}
        teacher_target_metrics: dict[str, Any] = {}
        teacher_eval_error: str | None = None
        student_eval_error: str | None = None
        teacher_model.eval()
        student_model.eval()
        final_student_checkpoint = student_ckpt
        final_student_source = "last"
        if best_student_ckpt.exists():
            DetectionCheckpointer(student_model).load(str(best_student_ckpt))
            final_student_checkpoint = best_student_ckpt
            final_student_source = "best_intermediate"
            print(
                "[DDT-DAOD][eval] "
                f"step={global_step} loaded_best_student checkpoint={best_student_ckpt}",
                flush=True,
            )
        print(
            "[DDT-DAOD][eval] "
            f"step={global_step} split=target_val student source={final_student_source}",
            flush=True,
        )
        try:
            student_target_metrics = _evaluate_split(self.cfg, student_model, "target_val", target_val)
        except RuntimeError as exc:
            student_eval_error = str(exc)
            student_target_metrics = dict(best_student_metrics)
            print(
                "[DDT-DAOD][eval][warning] "
                f"student final eval failed; using saved best intermediate metrics: {exc}",
                flush=True,
            )
        if evaluate_teacher:
            try:
                source_val = _limit_samples(
                    materialize_daod_dicts(self.cfg, "source_val"),
                    getattr(eval_cfg, "source_val_limit", 0),
                )
                with _device_context(teacher_device):
                    print(f"[DDT-DAOD][eval] step={global_step} split=source_val teacher", flush=True)
                    teacher_source_metrics = _evaluate_split(self.cfg, teacher_model, "source_val", source_val)
                    print(f"[DDT-DAOD][eval] step={global_step} split=target_val teacher", flush=True)
                    teacher_target_metrics = _evaluate_split(self.cfg, teacher_model, "target_val", target_val)
            except RuntimeError as exc:
                teacher_eval_error = str(exc)
                print(f"[DDT-DAOD][eval][warning] teacher eval failed; using student final metrics: {exc}", flush=True)
        final_model = "teacher" if teacher_target_metrics else "student"
        final_checkpoint = teacher_ckpt if teacher_target_metrics else final_student_checkpoint
        final_target_metrics = teacher_target_metrics if teacher_target_metrics else student_target_metrics

        save_json(run_dir / "target_val_metrics.json", final_target_metrics)
        save_json(run_dir / "student_target_val_metrics.json", student_target_metrics)
        if best_student_metrics:
            save_json(run_dir / "best_student_target_val_metrics.json", best_student_metrics)
        save_json(run_dir / "teacher_target_val_metrics.json", teacher_target_metrics)
        save_json(run_dir / "source_val_metrics.json", teacher_source_metrics)

        summary = {
            "epochs": int(epochs),
            "global_step": int(global_step),
            "source_checkpoint": str(source_checkpoint),
            "final_model": final_model,
            "final_checkpoint": str(final_checkpoint),
            "student_checkpoint": str(student_ckpt),
            "best_student_checkpoint": str(best_student_ckpt) if best_student_ckpt.exists() else None,
            "final_student_checkpoint": str(final_student_checkpoint),
            "final_student_source": final_student_source,
            "teacher_checkpoint": str(teacher_ckpt),
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
            "teacher_source_val_metrics": teacher_source_metrics,
            "teacher_target_val_metrics": teacher_target_metrics,
            "teacher_eval_error": teacher_eval_error,
            "student_eval_error": student_eval_error,
            "student_target_val_metrics": student_target_metrics,
            "final_target_val_metrics": final_target_metrics,
            "active_plan": active_plan,
            "label_guided_components": label_guided_summary,
            "label_guided_hook": label_guided_hook.state.as_dict(),
            "gradient_surgery": {
                "enabled": bool(gradient_surgery_enabled),
                "method": gradient_surgery_method if gradient_surgery_enabled else None,
                "apply_to_pseudo": bool(gradient_surgery_apply_pseudo) if gradient_surgery_enabled else None,
                "apply_to_masked": bool(gradient_surgery_apply_masked) if gradient_surgery_enabled else None,
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
                "history": [
                    entry.get("teacher_update", {})
                    for entry in history
                    if bool(entry.get("teacher_update", {}).get("label_guided_aema_enabled", False))
                ],
            },
            "thresholds": [float(v) for v in thresholds],
            "ddt_thresholds": [float(v) for v in thresholds],
            "effective_thresholds": [
                float(v)
                for v in label_guided_hook.adjust_thresholds(
                    [float(value) for value in thresholds],
                    global_step=global_step,
                )
            ],
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
            "oracle_pseudo": {
                "enabled": bool(oracle_pseudo_enabled),
                "mode": oracle_pseudo_mode if oracle_pseudo_enabled else None,
                "match_iou": float(getattr(oracle_pseudo_cfg, "match_iou", 0.5))
                if oracle_pseudo_enabled
                else None,
                "recovery_score": float(
                    getattr(oracle_pseudo_cfg, "recovery_score", getattr(oracle_pseudo_cfg, "score", 1.0))
                )
                if oracle_pseudo_enabled
                else None,
                "classes": getattr(oracle_pseudo_cfg, "classes", None) if oracle_pseudo_enabled else None,
                "default_policy": getattr(oracle_pseudo_cfg, "default_policy", None)
                if oracle_pseudo_enabled
                else None,
                "policies": getattr(oracle_pseudo_cfg, "policies", {}) if oracle_pseudo_enabled else {},
                "history": [
                    entry.get("oracle_pseudo", {})
                    for entry in history
                    if bool(entry.get("oracle_pseudo", {}).get("enabled", False))
                ],
            },
            "soft_query_activation": {
                "enabled": bool(soft_query_enabled),
                "method": soft_query_method if soft_query_enabled else None,
                "objective": soft_query_objective if soft_query_enabled else None,
                "loss_weight": float(soft_query_loss_weight) if soft_query_enabled else None,
                "match_mode": soft_query_match_mode if soft_query_enabled else None,
                "min_match_iou": float(soft_query_min_match_iou) if soft_query_enabled else None,
                "match_class_aware": bool(soft_query_match_class_aware) if soft_query_enabled else None,
                "positive_target": float(soft_query_positive_target) if soft_query_enabled else None,
                "margin": float(soft_query_margin) if soft_query_enabled else None,
                "activation_weight_power": float(soft_query_weight_power) if soft_query_enabled else None,
                "min_activation_weight": float(soft_query_min_weight) if soft_query_enabled else None,
                "fit_max_images": int(soft_query_fit_max_images) if soft_query_enabled else None,
                "controller": soft_query_activator.summary() if soft_query_activator is not None else None,
                "fit_stats": soft_query_stats if soft_query_enabled else {},
                "loss_ramp": {
                    "enabled": bool(getattr(soft_query_loss_ramp_cfg, "enabled", False)),
                    "start_fraction": getattr(soft_query_loss_ramp_cfg, "start_fraction", None),
                    "end_fraction": getattr(soft_query_loss_ramp_cfg, "end_fraction", None),
                    "start_multiplier": getattr(soft_query_loss_ramp_cfg, "start_multiplier", None),
                }
                if soft_query_enabled
                else {},
                "query_risk": {
                    "enabled": bool(soft_query_risk_enabled),
                    "aggregate": getattr(soft_query_risk_cfg, "aggregate", None) if soft_query_risk_enabled else None,
                    "min_weight": getattr(soft_query_risk_cfg, "min_weight", None) if soft_query_risk_enabled else None,
                    "power": getattr(soft_query_risk_cfg, "power", None) if soft_query_risk_enabled else None,
                }
                if soft_query_enabled
                else {},
                "class_gate": {
                    "enabled": bool(soft_query_gate_enabled),
                    "schedule": "static" if soft_query_gate_enabled else None,
                    "last_update_step": 0 if soft_query_class_gate is not None else None,
                    "state": soft_query_class_gate.as_dict()
                    if soft_query_class_gate is not None and soft_query_gate_enabled
                    else None,
                    "history": soft_query_gate_history if soft_query_gate_enabled else [],
                },
            },
        }
        save_json(run_dir / "summary.json", summary)
        if teacher_eval_error is None or "CUDA" not in teacher_eval_error:
            maybe_empty_cuda_cache()
        return summary
