"""DINO-adapted Dual-rate Dynamic Teacher trainer."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.daod import DAODListDataset, collate_daod_batch, cycle_daod_loader
from src.data.daod.detectron2 import materialize_daod_dicts
from src.engine.daod_round_trainer import (
    _evaluate_split,
    _limit_samples,
    _make_supervised_inputs,
    _resolve_teacher_device,
    _teacher_outputs_for_unlabeled,
    _update_aema,
    _update_ema,
)
from src.models import build_daod_model

from .masking import apply_block_mask_to_inputs
from .pseudo import filter_pseudo_rows, rows_to_annotations, update_dynamic_thresholds
from .utils import append_jsonl, maybe_empty_cuda_cache, save_json


def _class_loss_from_dict(loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    selected = [value for key, value in loss_dict.items() if "class" in key or "ce" in key]
    if not selected:
        selected = list(loss_dict.values())
    return sum(selected)


def _collect_grad_importance(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    importance: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] = param.grad.detach().abs().clone()
    return importance


class DDTDAODTrainer:
    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def fit(self, *, run_dir: Path, source_checkpoint: str) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        method_cfg = getattr(self.cfg, "method", object())
        train_cfg = getattr(method_cfg, "train", object())
        pseudo_cfg = getattr(method_cfg, "pseudo", object())
        mask_cfg = getattr(method_cfg, "masking", object())
        aema_cfg = getattr(method_cfg, "aema", object())
        eval_cfg = getattr(method_cfg, "eval", object())

        teacher_device = _resolve_teacher_device(train_cfg, self.device)
        student_adapter = build_daod_model(self.cfg, load_weights=False, device=self.device)
        teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=teacher_device)
        student_model = student_adapter.model.to(self.device)
        teacher_model = teacher_adapter.model.to(teacher_device)
        DetectionCheckpointer(student_model).load(str(source_checkpoint))
        DetectionCheckpointer(teacher_model).load(str(source_checkpoint))

        target_train = materialize_daod_dicts(self.cfg, "target_train")
        max_target_samples = int(getattr(train_cfg, "max_target_samples", 0))
        target_train = _limit_samples(target_train, max_target_samples)
        target_loader = DataLoader(
            DAODListDataset(target_train),
            batch_size=int(getattr(train_cfg, "batch_size", 1)),
            shuffle=bool(target_train),
            collate_fn=collate_daod_batch,
            num_workers=int(getattr(train_cfg, "num_workers", 0)),
        )
        target_iter = cycle_daod_loader(target_loader)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in student_model.parameters() if parameter.requires_grad],
            lr=float(getattr(train_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(train_cfg, "weight_decay", 1e-4)),
        )
        epochs = int(getattr(method_cfg, "epochs", 2))
        steps_per_epoch = max(len(target_loader), 1)
        total_steps = max(epochs * steps_per_epoch, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        num_classes = int(self.cfg.data.num_classes)
        thresholds = [float(getattr(pseudo_cfg, "threshold", 0.4))] * num_classes
        weak_rng = np.random.default_rng(int(getattr(self.cfg, "seed", 42)))
        strong_short_edge = int(getattr(self.cfg.detector, "min_size_test", 800))
        max_size = int(getattr(self.cfg.detector, "max_size_test", 1333))
        coef_masked_img = float(getattr(mask_cfg, "coef_masked_img", 1.0))
        alpha_ema = float(getattr(aema_cfg, "alpha_ema", 0.9996))
        alpha_aema = float(getattr(aema_cfg, "alpha_aema", 0.997))
        top_fraction = float(getattr(aema_cfg, "top_fraction", 0.1))
        update_interval = int(getattr(aema_cfg, "update_interval", 2))
        use_teacher_grad = bool(getattr(aema_cfg, "use_teacher_grad_importance", True))
        use_dynamic_threshold = bool(getattr(pseudo_cfg, "dynamic", True))
        log_period = int(getattr(train_cfg, "log_period", 100))
        checkpoint_period = int(getattr(train_cfg, "checkpoint_period", 0))
        train_log_path = run_dir / "train_log.jsonl"

        print(
            "[DDT-DAOD][train] "
            f"epochs={epochs} target_train={len(target_train)} "
            f"student_device={self.device} teacher_device={teacher_device} "
            f"source_ckpt={source_checkpoint}"
        )
        print(
            "[DDT-DAOD][ddt] "
            f"threshold={thresholds[0]:.3f} dynamic={use_dynamic_threshold} "
            f"mask_ratio={float(getattr(mask_cfg, 'masked_ratio', 0.5)):.3f} "
            f"alpha_ema={alpha_ema:.6f} alpha_aema={alpha_aema:.6f}"
        )

        global_step = 0
        history: list[dict[str, float]] = []
        for epoch_idx in range(1, epochs + 1):
            student_model.train()
            teacher_model.eval()
            epoch_loss = 0.0
            epoch_loss_pseudo = 0.0
            epoch_loss_mask = 0.0
            epoch_pseudo_boxes = 0
            score_sums = [0.0] * num_classes
            score_counts = [0] * num_classes

            for _ in range(steps_per_epoch):
                batch = next(target_iter, [])
                if not batch:
                    continue

                with torch.no_grad():
                    teacher_items = _teacher_outputs_for_unlabeled(
                        teacher_adapter,
                        batch,
                        weak_view_rng=weak_rng,
                    )

                pseudo_batch = []
                pseudo_box_count = 0
                for teacher_item in teacher_items:
                    pseudo_rows = filter_pseudo_rows(
                        teacher_item["query_rows"],
                        thresholds=thresholds,
                        dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                    )
                    for row in pseudo_rows:
                        class_id = int(row["category_id"])
                        score_sums[class_id] += float(row["score"])
                        score_counts[class_id] += 1
                    annotations = rows_to_annotations(pseudo_rows)
                    if not annotations:
                        continue
                    pseudo_sample = dict(teacher_item["sample"])
                    pseudo_sample["annotations"] = annotations
                    pseudo_batch.append(pseudo_sample)
                    pseudo_box_count += len(annotations)

                if not pseudo_batch:
                    global_step += 1
                    scheduler.step()
                    continue

                pseudo_inputs = _make_supervised_inputs(
                    student_adapter,
                    pseudo_batch,
                    strong_short_edge=strong_short_edge,
                    max_size=max_size,
                    device=self.device,
                )
                loss_pseudo = sum(student_model(pseudo_inputs).values())
                masked_inputs = apply_block_mask_to_inputs(
                    pseudo_inputs,
                    block_size=int(getattr(mask_cfg, "block_size", 64)),
                    masked_ratio=float(getattr(mask_cfg, "masked_ratio", 0.5)),
                )
                loss_mask = coef_masked_img * sum(student_model(masked_inputs).values())
                loss = loss_pseudo + loss_mask

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(getattr(train_cfg, "clip_max_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        float(getattr(train_cfg, "clip_max_norm", 0.0)),
                    )
                optimizer.step()

                grad_importance: dict[str, torch.Tensor] = {}
                if use_teacher_grad:
                    teacher_ctx = torch.cuda.device(teacher_device) if teacher_device.type == "cuda" else nullcontext()
                    with teacher_ctx:
                        teacher_model.train()
                        for parameter in teacher_model.parameters():
                            parameter.requires_grad_(True)
                        teacher_inputs = _make_supervised_inputs(
                            teacher_adapter,
                            pseudo_batch,
                            strong_short_edge=strong_short_edge,
                            max_size=max_size,
                            device=teacher_device,
                        )
                        teacher_loss = _class_loss_from_dict(teacher_model(teacher_inputs))
                        teacher_model.zero_grad(set_to_none=True)
                        teacher_loss.backward()
                        grad_importance = _collect_grad_importance(teacher_model)
                        teacher_model.zero_grad(set_to_none=True)
                        teacher_model.eval()
                        for parameter in teacher_model.parameters():
                            parameter.requires_grad_(False)

                if update_interval > 0 and global_step % update_interval == 0:
                    if grad_importance:
                        _update_aema(
                            teacher_model,
                            student_model,
                            grad_importance,
                            momentum=alpha_ema,
                            adaptive_momentum=alpha_aema,
                            top_fraction=top_fraction,
                        )
                    else:
                        _update_ema(teacher_model, student_model, alpha_ema)

                global_step += 1
                scheduler.step()
                epoch_loss += float(loss.detach().cpu())
                epoch_loss_pseudo += float(loss_pseudo.detach().cpu())
                epoch_loss_mask += float(loss_mask.detach().cpu())
                epoch_pseudo_boxes += int(pseudo_box_count)

                if log_period > 0 and global_step % log_period == 0:
                    append_jsonl(
                        train_log_path,
                        {
                            "epoch": int(epoch_idx),
                            "step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_total": float(loss.detach().cpu()),
                            "loss_pseudo": float(loss_pseudo.detach().cpu()),
                            "loss_masked": float(loss_mask.detach().cpu()),
                            "pseudo_box_count": int(pseudo_box_count),
                            "thresholds": [float(v) for v in thresholds],
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
                )
                print(f"[DDT-DAOD][thresholds] epoch={epoch_idx} values={thresholds}")

            denom = max(steps_per_epoch, 1)
            epoch_summary = {
                "epoch": float(epoch_idx),
                "loss_total": epoch_loss / denom,
                "loss_pseudo": epoch_loss_pseudo / denom,
                "loss_masked": epoch_loss_mask / denom,
                "pseudo_boxes": float(epoch_pseudo_boxes),
                "thresholds": [float(v) for v in thresholds],
            }
            history.append(epoch_summary)
            append_jsonl(run_dir / "epoch_log.jsonl", epoch_summary)

        student_ckpt = run_dir / "student_last.pth"
        teacher_ckpt = run_dir / "teacher_last.pth"
        DetectionCheckpointer(student_model, save_dir=str(run_dir)).save("student_last")
        DetectionCheckpointer(teacher_model, save_dir=str(run_dir)).save("teacher_last")

        source_val = _limit_samples(materialize_daod_dicts(self.cfg, "source_val"), getattr(eval_cfg, "source_val_limit", 0))
        target_val = _limit_samples(materialize_daod_dicts(self.cfg, "target_val"), getattr(eval_cfg, "target_val_limit", 0))
        teacher_model.eval()
        student_model.eval()
        print(f"[DDT-DAOD][eval] step={global_step} split=source_val teacher")
        teacher_source_metrics = _evaluate_split(self.cfg, teacher_model, "source_val", source_val)
        print(f"[DDT-DAOD][eval] step={global_step} split=target_val teacher")
        teacher_target_metrics = _evaluate_split(self.cfg, teacher_model, "target_val", target_val)
        print(f"[DDT-DAOD][eval] step={global_step} split=target_val student")
        student_target_metrics = _evaluate_split(self.cfg, student_model, "target_val", target_val)

        summary = {
            "epochs": int(epochs),
            "global_step": int(global_step),
            "source_checkpoint": str(source_checkpoint),
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt),
            "history": history,
            "teacher_source_val_metrics": teacher_source_metrics,
            "teacher_target_val_metrics": teacher_target_metrics,
            "student_target_val_metrics": student_target_metrics,
            "thresholds": [float(v) for v in thresholds],
        }
        save_json(run_dir / "summary.json", summary)
        maybe_empty_cuda_cache()
        return summary
