"""Step-wise label injection trainer for DAOD.

This trainer implements one continuous self-training run where new human labels
are injected at scheduled global steps, instead of launching separate training
episodes per round.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.daod import DAODListDataset, collate_daod_batch, cycle_daod_loader
from src.data.daod.detectron2 import materialize_daod_dicts
from src.engine.daod_round_trainer import (
    _append_jsonl,
    _build_hard_teacher_rows,
    _build_soft_teacher_targets,
    _compute_round_budgets,
    _continuous_total_steps,
    _cosine_lr_value,
    _cuda_mem_gb,
    _evaluate_split,
    _limit_samples,
    _loader_len,
    _log_eval_metrics,
    _make_supervised_inputs,
    _mem_log_payload,
    _print_eval_summary,
    _pseudo_annotations_from_rows,
    _resolve_budget_total,
    _resolve_teacher_device,
    _set_optimizer_lr,
    _signal_specs,
    _student_outputs_for_unlabeled,
    _student_soft_loss,
    _teacher_outputs_for_unlabeled,
    _update_aema,
    _update_ema,
    _accumulate_grad_importance,
)
from src.engine.daod_pseudo_recalibration import compute_pseudo_recalibration
from src.engine.utils import save_json
from src.methods.daod_method import DAODRoundPlan, DAODRoundState
from src.models import build_daod_model


def _build_stage_loaders(
    target_train_dicts: list[dict[str, Any]],
    labeled_ids: set[str],
    *,
    labeled_batch_size: int,
    unlabeled_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, Any, Any, list[dict[str, Any]], list[dict[str, Any]]]:
    labeled_items = [sample for sample in target_train_dicts if sample["sample_id"] in labeled_ids]
    unlabeled_items = [sample for sample in target_train_dicts if sample["sample_id"] not in labeled_ids]
    labeled_dataset = DAODListDataset(labeled_items)
    unlabeled_dataset = DAODListDataset(unlabeled_items)
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=labeled_batch_size,
        shuffle=bool(labeled_items),
        collate_fn=collate_daod_batch,
        num_workers=num_workers,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=unlabeled_batch_size,
        shuffle=bool(unlabeled_items),
        collate_fn=collate_daod_batch,
        num_workers=num_workers,
    )
    return (
        labeled_loader,
        unlabeled_loader,
        cycle_daod_loader(labeled_loader),
        cycle_daod_loader(unlabeled_loader),
        labeled_items,
        unlabeled_items,
    )


def _injection_points(total_steps: int, num_stages: int) -> list[int]:
    if num_stages <= 1:
        return []
    points: list[int] = []
    for idx in range(1, num_stages):
        point = max(1, int(round(float(total_steps) * idx / num_stages)))
        if point < total_steps:
            points.append(point)
    return points


class DAODStepwiseInjectionTrainer:
    """One continuous run with scheduled label-injection events."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def fit_with_injections(
        self,
        *,
        cfg: Any,
        run_dir: Path,
        state_in: DAODRoundState,
        initial_plan: DAODRoundPlan,
        remaining_budgets: list[int],
        plan_callback: Callable[[DAODRoundState, int, str, int], DAODRoundPlan],
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        method_train_cfg = getattr(cfg.method, "train", object())
        method_eval_cfg = getattr(cfg.method, "eval", object())

        teacher_device = _resolve_teacher_device(method_train_cfg, self.device)
        teacher_init_from_student = bool(getattr(method_train_cfg, "teacher_init_from_student", False))
        continue_optimizer_across_rounds = bool(getattr(method_train_cfg, "continue_optimizer_across_rounds", True))
        use_pseudo_labels = bool(getattr(method_train_cfg, "use_pseudo_labels", True))
        base_lr = float(getattr(method_train_cfg, "lr", 1e-4))
        scheduler_name = str(getattr(method_train_cfg, "lr_scheduler", "cosine")).strip().lower()
        if scheduler_name not in {"cosine", "constant", "none"}:
            raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")

        student_adapter = build_daod_model(cfg, load_weights=False, device=self.device)
        teacher_adapter = build_daod_model(cfg, load_weights=False, device=teacher_device)
        student_model = student_adapter.model.to(self.device)
        teacher_model = teacher_adapter.model.to(teacher_device)
        DetectionCheckpointer(student_model).load(str(state_in.student_checkpoint))
        teacher_init_ckpt = str(state_in.student_checkpoint) if teacher_init_from_student else str(state_in.teacher_checkpoint)
        DetectionCheckpointer(teacher_model).load(teacher_init_ckpt)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        target_train_dicts = materialize_daod_dicts(cfg, "target_train")
        labeled_batch_size = int(getattr(method_train_cfg, "labeled_batch_size", 4))
        unlabeled_batch_size = int(getattr(method_train_cfg, "unlabeled_batch_size", 4))
        loader_num_workers = int(getattr(method_train_cfg, "num_workers", 0))
        max_epochs = int(getattr(cfg.method, "round_epochs", 1))

        budget_total = _resolve_budget_total(cfg, len(target_train_dicts))
        budget_schedule = [len(initial_plan.queried_ids), *remaining_budgets]
        if sum(budget_schedule) <= 0:
            budget_schedule = _compute_round_budgets(budget_total, int(getattr(cfg.method, "num_rounds", 1)))
        final_labeled_count = len(state_in.queried_ids) + sum(int(v) for v in budget_schedule)
        full_labeled_len = _loader_len(final_labeled_count, labeled_batch_size)
        full_unlabeled_len = _loader_len(max(0, len(target_train_dicts) - final_labeled_count), unlabeled_batch_size)
        if use_pseudo_labels:
            steps_per_epoch = max(full_labeled_len, full_unlabeled_len, 1)
        else:
            steps_per_epoch = max(full_labeled_len, 1)
        total_steps = max_epochs * steps_per_epoch

        optimizer = torch.optim.AdamW(
            [p for p in student_model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=float(getattr(method_train_cfg, "weight_decay", 1e-4)),
        )
        if continue_optimizer_across_rounds and state_in.optimizer_checkpoint:
            optimizer.load_state_dict(torch.load(str(state_in.optimizer_checkpoint), map_location="cpu", weights_only=False))
        if scheduler_name == "cosine":
            _set_optimizer_lr(optimizer, _cosine_lr_value(base_lr, int(getattr(state_in, "global_step", 0)), max(total_steps, 1)))
        else:
            _set_optimizer_lr(optimizer, base_lr)

        strong_short_edge = int(getattr(cfg.detector, "min_size_test", 800))
        max_size = int(getattr(cfg.detector, "max_size_test", 1333))
        losses_cfg = getattr(method_train_cfg, "losses", object())
        ema_cfg = getattr(method_train_cfg, "ema", object())
        routing_cfg = getattr(method_train_cfg, "routing", object())
        hard_loss_weight = float(getattr(losses_cfg, "hard_pseudo_weight", 1.0))
        soft_loss_weight = float(getattr(losses_cfg, "soft_distill_weight", 1.0))
        ema_mode = str(getattr(ema_cfg, "mode", "ema")).strip().lower()
        ema_momentum = float(getattr(ema_cfg, "momentum", 0.999))
        ema_adaptive_momentum = float(getattr(ema_cfg, "adaptive_momentum", max(0.0, ema_momentum - 0.001)))
        ema_top_fraction = float(getattr(ema_cfg, "top_fraction", 0.1))
        ema_update_interval = int(getattr(ema_cfg, "update_interval", 1))
        weak_view_rng = np.random.default_rng(int(getattr(cfg, "seed", 42)))

        hard_routing_cfg = getattr(routing_cfg, "hard", object())
        soft_routing_cfg = getattr(routing_cfg, "soft", object())
        pseudo_recalibration_cfg = getattr(method_train_cfg, "pseudo_recalibration", object())
        pseudo_recalibration_enabled = bool(getattr(pseudo_recalibration_cfg, "enabled", False))
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
        soft_threshold = float(getattr(soft_routing_cfg, "threshold", 0.3))
        hard_nms_iou = float(getattr(hard_routing_cfg, "dedup_iou_thresh", 0.7))
        soft_hard_exclusion_iou_max = float(getattr(soft_routing_cfg, "hard_exclusion_iou_max", 0.4))
        soft_match_iou_min = float(getattr(soft_routing_cfg, "match_iou_min", 0.3))
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

        train_log_path = run_dir / "train_log.jsonl"
        eval_log_path = run_dir / "eval_log.jsonl"
        memory_log_path = run_dir / "memory_log.jsonl"
        injection_log_path = run_dir / "injection_log.jsonl"
        write_tensorboard = bool(getattr(method_eval_cfg, "write_tensorboard", False))
        eval_writer = SummaryWriter(log_dir=str(run_dir)) if write_tensorboard else None

        current_stage = 0
        injection_points = _injection_points(total_steps, len(budget_schedule))
        labeled_ids = set(state_in.queried_ids).union(initial_plan.queried_ids)
        queried_ids = set(labeled_ids)
        num_classes = int(getattr(cfg.data, "num_classes", 0))
        cfg_seed = int(getattr(cfg, "seed", 42))
        hard_class_score_mins: dict[int, float] | None = None
        hard_class_counts: list[int] = []
        hard_recalibration_stats: dict[str, Any] = {}
        if pseudo_recalibration_enabled:
            hard_class_score_mins, hard_recalibration_stats = compute_pseudo_recalibration(
                target_train_dicts,
                labeled_ids,
                num_classes=num_classes,
                base_score_min=hard_score_min,
                recalibration_cfg=pseudo_recalibration_cfg,
                teacher_adapter=teacher_adapter,
                stage_idx=0,
                seed=cfg_seed,
            )
            hard_class_counts = list(hard_recalibration_stats.get("class_counts", []))

        save_json(
            run_dir / "injection_0_plan.json",
            {
                "injection_idx": 0,
                "queried_ids": list(initial_plan.queried_ids),
                "sample_plans": [plan.__dict__ for plan in initial_plan.sample_plans],
            },
        )
        _append_jsonl(
            injection_log_path,
            {
                "injection_idx": 0,
                "global_step": int(getattr(state_in, "global_step", 0)),
                "selected_count": len(initial_plan.queried_ids),
                "total_labeled": len(labeled_ids),
                "hard_pseudo_class_counts": hard_class_counts if pseudo_recalibration_enabled else None,
                "hard_pseudo_class_score_mins": hard_class_score_mins if pseudo_recalibration_enabled else None,
                "hard_pseudo_recalibration": hard_recalibration_stats if pseudo_recalibration_enabled else None,
            },
        )

        labeled_loader, unlabeled_loader, labeled_iter, unlabeled_iter, _, _ = _build_stage_loaders(
            target_train_dicts,
            labeled_ids,
            labeled_batch_size=labeled_batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
            num_workers=loader_num_workers,
        )

        global_step = int(getattr(state_in, "global_step", 0))
        history: list[dict[str, float]] = []
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        _append_jsonl(memory_log_path, _mem_log_payload(self.device, tag="after_model_build", epoch=0, step=global_step))
        grad_accum: dict[str, torch.Tensor] = {}

        def maybe_inject() -> None:
            nonlocal current_stage, labeled_ids, queried_ids, labeled_loader, unlabeled_loader, labeled_iter, unlabeled_iter
            nonlocal hard_class_score_mins, hard_class_counts, hard_recalibration_stats
            if current_stage >= len(remaining_budgets):
                return
            if global_step < injection_points[current_stage]:
                return
            budget_k = int(remaining_budgets[current_stage])
            temp_student_ckpt = run_dir / f"temp_selection_student_stage_{current_stage + 1}.pth"
            DetectionCheckpointer(student_model, save_dir=str(run_dir)).save(f"temp_selection_student_stage_{current_stage + 1}")
            plan_state = DAODRoundState(
                round_idx=current_stage + 1,
                queried_ids=set(queried_ids),
                budget_total=state_in.budget_total,
                budget_used=len(queried_ids),
                teacher_checkpoint=str(state_in.teacher_checkpoint),
                student_checkpoint=str(temp_student_ckpt),
                optimizer_checkpoint=None,
                scheduler_checkpoint=None,
                global_step=global_step,
            )
            plan = plan_callback(plan_state, budget_k, str(temp_student_ckpt), current_stage + 1)
            save_json(
                run_dir / f"injection_{current_stage + 1}_plan.json",
                {
                    "injection_idx": current_stage + 1,
                    "queried_ids": list(plan.queried_ids),
                    "sample_plans": [sample_plan.__dict__ for sample_plan in plan.sample_plans],
                },
            )
            labeled_ids = set(labeled_ids).union(plan.queried_ids)
            queried_ids = set(queried_ids).union(plan.queried_ids)
            if pseudo_recalibration_enabled:
                hard_class_score_mins, hard_recalibration_stats = compute_pseudo_recalibration(
                    target_train_dicts,
                    labeled_ids,
                    num_classes=num_classes,
                    base_score_min=hard_score_min,
                    recalibration_cfg=pseudo_recalibration_cfg,
                    teacher_adapter=teacher_adapter,
                    stage_idx=current_stage + 1,
                    seed=cfg_seed,
                )
                hard_class_counts = list(hard_recalibration_stats.get("class_counts", []))
            labeled_loader, unlabeled_loader, labeled_iter, unlabeled_iter, _, _ = _build_stage_loaders(
                target_train_dicts,
                labeled_ids,
                labeled_batch_size=labeled_batch_size,
                unlabeled_batch_size=unlabeled_batch_size,
                num_workers=loader_num_workers,
            )
            _append_jsonl(
                injection_log_path,
                {
                    "injection_idx": current_stage + 1,
                    "global_step": int(global_step),
                    "selected_count": len(plan.queried_ids),
                    "total_labeled": len(labeled_ids),
                    "hard_pseudo_class_counts": hard_class_counts if pseudo_recalibration_enabled else None,
                    "hard_pseudo_class_score_mins": hard_class_score_mins if pseudo_recalibration_enabled else None,
                    "hard_pseudo_recalibration": hard_recalibration_stats if pseudo_recalibration_enabled else None,
                },
            )
            current_stage += 1

        try:
            student_model.train()
            epoch_stats = {"loss_total": 0.0, "loss_sup": 0.0, "loss_hard": 0.0, "loss_soft": 0.0, "steps": 0.0}
            for _ in range(total_steps):
                maybe_inject()

                labeled_batch = next(labeled_iter, [])
                unlabeled_batch = next(unlabeled_iter, [])
                if not labeled_batch and not unlabeled_batch:
                    continue

                loss = torch.tensor(0.0, device=self.device)
                loss_sup = torch.tensor(0.0, device=self.device)
                loss_hard = torch.tensor(0.0, device=self.device)
                loss_soft = torch.tensor(0.0, device=self.device)
                hard_pseudo_count = 0
                soft_target_count = 0

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

                enable_hard_pseudo = hard_loss_weight > 0.0
                enable_soft_pseudo = soft_loss_weight > 0.0

                if unlabeled_batch and use_pseudo_labels and (enable_hard_pseudo or enable_soft_pseudo):
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
                    _append_jsonl(memory_log_path, _mem_log_payload(self.device, tag="after_unlabeled_forward", epoch=1, step=global_step + 1))
                    hard_batch = []
                    soft_batch = []
                    for teacher_item, student_item in zip(teacher_items, student_items):
                        hard_rows = []
                        soft_targets = []
                        if enable_hard_pseudo or enable_soft_pseudo:
                            hard_rows = _build_hard_teacher_rows(
                                teacher_item,
                                hard_score_min=hard_score_min,
                                hard_nms_iou=hard_nms_iou,
                                class_score_mins=hard_class_score_mins,
                            )
                        if enable_soft_pseudo:
                            soft_targets = _build_soft_teacher_targets(
                                teacher_item,
                                student_item,
                                hard_rows=hard_rows,
                                soft_score_min=soft_score_min,
                                soft_score_max=soft_score_max,
                                soft_specs=soft_specs,
                                soft_threshold=soft_threshold,
                                hard_exclusion_iou_max=soft_hard_exclusion_iou_max,
                            )
                        hard_pseudo_count += len(hard_rows)
                        soft_target_count += len(soft_targets)
                        if enable_hard_pseudo and hard_rows:
                            pseudo_annotations = _pseudo_annotations_from_rows(hard_rows)
                            if pseudo_annotations:
                                pseudo_sample = dict(teacher_item["sample"])
                                pseudo_sample["annotations"] = pseudo_annotations
                                hard_batch.append(pseudo_sample)
                        if soft_targets:
                            soft_batch.append(
                                {
                                    "sample": teacher_item["sample"],
                                    "soft_targets": soft_targets,
                                    "student_raw": student_item["student_raw"],
                                    "student_query_rows": student_item["student_query_rows"],
                                }
                            )

                    if enable_hard_pseudo and hard_batch:
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
                        _append_jsonl(memory_log_path, _mem_log_payload(self.device, tag="after_hard_loss", epoch=1, step=global_step + 1))
                    if enable_soft_pseudo and soft_batch:
                        loss_soft = _student_soft_loss(
                            soft_batch,
                            soft_loss_weight=soft_loss_weight,
                            match_iou_min=soft_match_iou_min,
                        )
                        loss = loss + loss_soft
                        _append_jsonl(memory_log_path, _mem_log_payload(self.device, tag="after_soft_loss", epoch=1, step=global_step + 1))

                if loss.requires_grad and float(loss.detach().cpu()) > 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if ema_mode == "aema":
                        _accumulate_grad_importance(grad_accum, student_model)
                    optimizer.step()
                    if ema_update_interval > 0 and (global_step + 1) % ema_update_interval == 0:
                        if ema_mode == "aema":
                            _update_aema(
                                teacher_model,
                                student_model,
                                grad_accum,
                                momentum=ema_momentum,
                                adaptive_momentum=ema_adaptive_momentum,
                                top_fraction=ema_top_fraction,
                            )
                            grad_accum = {}
                        else:
                            _update_ema(teacher_model, student_model, ema_momentum)
                        _append_jsonl(memory_log_path, _mem_log_payload(self.device, tag="after_optimizer_step", epoch=1, step=global_step + 1))

                global_step += 1
                if scheduler_name == "cosine":
                    _set_optimizer_lr(optimizer, _cosine_lr_value(base_lr, global_step, max(total_steps, 1)))

                epoch_stats["loss_total"] += float(loss.detach().cpu())
                epoch_stats["loss_sup"] += float(loss_sup.detach().cpu())
                epoch_stats["loss_hard"] += float(loss_hard.detach().cpu())
                epoch_stats["loss_soft"] += float(loss_soft.detach().cpu())
                epoch_stats["steps"] += 1.0

                if log_period > 0 and global_step % log_period == 0:
                    _append_jsonl(
                        train_log_path,
                        {
                            "step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_total": float(loss.detach().cpu()),
                            "loss_sup": float(loss_sup.detach().cpu()),
                            "loss_hard": float(loss_hard.detach().cpu()),
                            "loss_soft": float(loss_soft.detach().cpu()),
                            "num_labeled_images": int(len(labeled_batch)),
                            "num_unlabeled_images": int(len(unlabeled_batch)),
                            "num_hard_pseudo_boxes": int(hard_pseudo_count),
                            "num_soft_targets": int(soft_target_count),
                            "current_stage": int(current_stage),
                            "pseudo_recalibration_enabled": bool(pseudo_recalibration_enabled),
                            "pseudo_recalibration_method": hard_recalibration_stats.get("method") if pseudo_recalibration_enabled else None,
                        },
                    )

                if eval_period > 0 and global_step % eval_period == 0:
                    student_model.eval()
                    print(f"[DAOD][eval] step={global_step} split=source_val start")
                    source_val_metrics = _evaluate_split(cfg, student_model, "source_val", source_val_dicts)
                    _print_eval_summary("source_val", source_val_metrics)
                    print(f"[DAOD][eval] step={global_step} split=target_val start")
                    target_val_metrics = _evaluate_split(cfg, student_model, "target_val", target_val_dicts)
                    _print_eval_summary("target_val", target_val_metrics)
                    save_json(run_dir / "source_val_metrics.json", source_val_metrics)
                    save_json(run_dir / "target_val_metrics.json", target_val_metrics)
                    _log_eval_metrics(eval_writer, global_step, "source_val", source_val_metrics)
                    _log_eval_metrics(eval_writer, global_step, "target_val", target_val_metrics)
                    _append_jsonl(
                        eval_log_path,
                        {
                            "step": int(global_step),
                            "source_val_metrics": source_val_metrics,
                            "target_val_metrics": target_val_metrics,
                        },
                    )
                    student_model.train()

                if checkpoint_period > 0 and global_step % checkpoint_period == 0:
                    DetectionCheckpointer(student_model, save_dir=str(run_dir)).save(f"student_step_{global_step}")
                    DetectionCheckpointer(teacher_model, save_dir=str(run_dir)).save(f"teacher_step_{global_step}")

            denom = max(epoch_stats["steps"], 1.0)
            history.append(
                {
                    "loss_total": epoch_stats["loss_total"] / denom,
                    "loss_sup": epoch_stats["loss_sup"] / denom,
                    "loss_hard": epoch_stats["loss_hard"] / denom,
                    "loss_soft": epoch_stats["loss_soft"] / denom,
                    "steps": epoch_stats["steps"],
                }
            )

            student_ckpt = run_dir / "student_last.pth"
            teacher_ckpt = run_dir / "teacher_last.pth"
            optimizer_ckpt = run_dir / "optimizer_last.pt"
            DetectionCheckpointer(student_model, save_dir=str(run_dir)).save("student_last")
            DetectionCheckpointer(teacher_model, save_dir=str(run_dir)).save("teacher_last")
            torch.save(optimizer.state_dict(), optimizer_ckpt)

            student_model.eval()
            print(f"[DAOD][eval] step={global_step} split=source_val final")
            source_val_metrics = _evaluate_split(cfg, student_model, "source_val", source_val_dicts)
            _print_eval_summary("source_val", source_val_metrics)
            print(f"[DAOD][eval] step={global_step} split=target_val final")
            target_val_metrics = _evaluate_split(cfg, student_model, "target_val", target_val_dicts)
            _print_eval_summary("target_val", target_val_metrics)
            save_json(run_dir / "train_history.json", {"history": history})
            save_json(run_dir / "source_val_metrics.json", source_val_metrics)
            save_json(run_dir / "target_val_metrics.json", target_val_metrics)
            _log_eval_metrics(eval_writer, global_step, "source_val", source_val_metrics)
            _log_eval_metrics(eval_writer, global_step, "target_val", target_val_metrics)

            return {
                "student_checkpoint": str(student_ckpt),
                "teacher_checkpoint": str(teacher_ckpt),
                "optimizer_checkpoint": str(optimizer_ckpt),
                "scheduler_checkpoint": None,
                "global_step": int(global_step),
                "queried_ids": sorted(queried_ids),
                "train_history": history,
                "source_val_metrics": source_val_metrics,
                "target_val_metrics": target_val_metrics,
                "total_steps": int(total_steps),
            }
        finally:
            if eval_writer is not None:
                eval_writer.close()
