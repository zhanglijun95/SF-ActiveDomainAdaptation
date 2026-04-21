"""Training helpers for the isolated FNP baseline.

The baseline keeps its FNPM-based selection logic local, but delegates the
actual DINO round update to the repo's proven DAOD round trainer. This avoids
duplicating a fragile teacher/student optimization loop while keeping the
baseline orchestration isolated under `baselines/fnp_daod/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.daod import DAODListDataset, collate_daod_batch, cycle_daod_loader
from src.data.daod.detectron2 import materialize_daod_dicts
from src.engine.daod_round_trainer import (
    _build_hard_teacher_rows,
    _cosine_lr_value,
    _evaluate_split,
    _limit_samples,
    _make_supervised_inputs,
    _pseudo_annotations_from_rows,
    _resolve_teacher_device,
    _set_optimizer_lr,
    _teacher_outputs_for_unlabeled,
    _update_ema,
)
from src.models import build_daod_model

from .dino_hooks import extract_pooled_backbone_features
from .utils import maybe_empty_cuda_cache, resolve_aux_device, save_json


class DomainDiscriminator(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(max(int(num_layers) - 1, 1)):
            layers.extend(
                [
                    torch.nn.Linear(in_dim, int(hidden_dim)),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(in_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _limit_samples(dataset: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or int(limit) <= 0:
        return dataset
    return dataset[: int(limit)]


def _balanced_source_target_samples(
    source_samples: list[dict[str, Any]],
    target_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not source_samples or not target_samples:
        return source_samples, target_samples
    keep = min(len(source_samples), len(target_samples))
    return source_samples[:keep], target_samples[:keep]


def _batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    batch_size = max(int(batch_size), 1)
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


class FNPDAODTrainer:
    """Baseline-local hard-pseudo mean-teacher trainer."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.aux_device = resolve_aux_device(cfg, device)

    def fit_round(
        self,
        *,
        run_dir: Path,
        state_in: Any,
        plan: Any,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = getattr(self.cfg.method, "train", object())
        eval_cfg = getattr(self.cfg.method, "eval", object())

        teacher_device = _resolve_teacher_device(train_cfg, self.device)
        teacher_init_from_student = bool(getattr(train_cfg, "teacher_init_from_student", False))
        continue_optimizer_across_rounds = bool(getattr(train_cfg, "continue_optimizer_across_rounds", True))
        use_pseudo_labels = bool(getattr(train_cfg, "use_pseudo_labels", True))
        base_lr = float(getattr(train_cfg, "lr", 1e-4))
        weight_decay = float(getattr(train_cfg, "weight_decay", 1e-4))
        scheduler_name = str(getattr(train_cfg, "lr_scheduler", "cosine")).strip().lower()
        if scheduler_name not in {"cosine", "constant", "none"}:
            raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")

        student_adapter = build_daod_model(self.cfg, load_weights=False, device=self.device)
        teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=teacher_device)
        student_model = student_adapter.model.to(self.device)
        teacher_model = teacher_adapter.model.to(teacher_device)
        DetectionCheckpointer(student_model).load(str(state_in.student_checkpoint))
        teacher_ckpt = str(state_in.student_checkpoint) if teacher_init_from_student else str(state_in.teacher_checkpoint)
        DetectionCheckpointer(teacher_model).load(teacher_ckpt)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

        target_train_dicts = materialize_daod_dicts(self.cfg, "target_train")
        labeled_ids = set(state_in.queried_ids).union(plan.queried_ids)
        labeled_items = [sample for sample in target_train_dicts if sample["sample_id"] in labeled_ids]
        unlabeled_items = [sample for sample in target_train_dicts if sample["sample_id"] not in labeled_ids]

        labeled_loader = DataLoader(
            DAODListDataset(labeled_items),
            batch_size=int(getattr(train_cfg, "labeled_batch_size", 1)),
            shuffle=bool(labeled_items),
            collate_fn=collate_daod_batch,
            num_workers=int(getattr(train_cfg, "num_workers", 0)),
        )
        unlabeled_loader = DataLoader(
            DAODListDataset(unlabeled_items),
            batch_size=int(getattr(train_cfg, "unlabeled_batch_size", 1)),
            shuffle=bool(unlabeled_items),
            collate_fn=collate_daod_batch,
            num_workers=int(getattr(train_cfg, "num_workers", 0)),
        )
        labeled_iter = cycle_daod_loader(labeled_loader)
        unlabeled_iter = cycle_daod_loader(unlabeled_loader)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in student_model.parameters() if parameter.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
        )
        if continue_optimizer_across_rounds and getattr(state_in, "optimizer_checkpoint", None):
            optimizer.load_state_dict(
                torch.load(str(state_in.optimizer_checkpoint), map_location="cpu", weights_only=False)
            )

        steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader), 1) if use_pseudo_labels else max(len(labeled_loader), 1)
        max_epochs = int(getattr(self.cfg.method, "round_epochs", 1))
        total_steps = max(max_epochs * steps_per_epoch, 1)
        global_step = int(getattr(state_in, "global_step", 0))
        if scheduler_name == "cosine":
            _set_optimizer_lr(optimizer, _cosine_lr_value(base_lr, global_step, total_steps))
        else:
            _set_optimizer_lr(optimizer, base_lr)

        strong_short_edge = int(getattr(self.cfg.detector, "min_size_test", 800))
        max_size = int(getattr(self.cfg.detector, "max_size_test", 1333))
        losses_cfg = getattr(train_cfg, "losses", object())
        hard_pseudo_weight = float(getattr(losses_cfg, "hard_pseudo_weight", 1.0))
        ema_cfg = getattr(train_cfg, "ema", object())
        ema_momentum = float(getattr(ema_cfg, "momentum", 0.999))
        ema_update_interval = int(getattr(ema_cfg, "update_interval", 1))
        routing_cfg = getattr(train_cfg, "routing", object())
        hard_cfg = getattr(routing_cfg, "hard", object())
        hard_score_min = float(getattr(hard_cfg, "score_min", 0.5))
        hard_nms_iou = float(getattr(hard_cfg, "dedup_iou_thresh", 0.7))
        log_period = int(getattr(train_cfg, "log_period", 100))
        eval_period = int(getattr(train_cfg, "eval_period", 0))
        checkpoint_period = int(getattr(train_cfg, "checkpoint_period", 0))

        source_val_dicts = _limit_samples(materialize_daod_dicts(self.cfg, "source_val"), getattr(eval_cfg, "source_val_limit", 0))
        target_val_dicts = _limit_samples(materialize_daod_dicts(self.cfg, "target_val"), getattr(eval_cfg, "target_val_limit", 0))

        train_log_path = run_dir / "train_log.jsonl"
        history: list[dict[str, float]] = []
        print(f"[DAOD][round] student_device={self.device} teacher_device={teacher_device}")
        print(
            "[DAOD][teacher_init] "
            f"mode={'student_checkpoint' if teacher_init_from_student else 'teacher_checkpoint'} "
            f"path={teacher_ckpt}"
        )
        print(
            "[DAOD][train] "
            f"continue_optimizer_across_rounds={continue_optimizer_across_rounds} "
            f"lr_scheduler={scheduler_name} "
            "hard_pseudo_only=true"
        )

        for epoch_idx in range(1, max_epochs + 1):
            student_model.train()
            epoch_stats = {"loss_total": 0.0, "loss_sup": 0.0, "loss_hard": 0.0, "steps": 0.0}
            for _ in range(steps_per_epoch):
                labeled_batch = next(labeled_iter, [])
                unlabeled_batch = next(unlabeled_iter, [])
                if not labeled_batch and not unlabeled_batch:
                    continue

                loss = torch.tensor(0.0, device=self.device)
                loss_sup = torch.tensor(0.0, device=self.device)
                loss_hard = torch.tensor(0.0, device=self.device)
                hard_pseudo_count = 0

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

                if unlabeled_batch and use_pseudo_labels:
                    teacher_items = _teacher_outputs_for_unlabeled(
                        teacher_adapter,
                        unlabeled_batch,
                        weak_view_rng=None,
                    )
                    hard_batch = []
                    for teacher_item in teacher_items:
                        hard_rows = _build_hard_teacher_rows(
                            teacher_item,
                            hard_score_min=hard_score_min,
                            hard_nms_iou=hard_nms_iou,
                        )
                        hard_pseudo_count += len(hard_rows)
                        if not hard_rows:
                            continue
                        pseudo_annotations = _pseudo_annotations_from_rows(hard_rows)
                        if not pseudo_annotations:
                            continue
                        pseudo_sample = dict(teacher_item["sample"])
                        pseudo_sample["annotations"] = pseudo_annotations
                        hard_batch.append(pseudo_sample)

                    if hard_batch:
                        hard_inputs = _make_supervised_inputs(
                            student_adapter,
                            hard_batch,
                            strong_short_edge=strong_short_edge,
                            max_size=max_size,
                            device=self.device,
                        )
                        hard_loss_dict = student_model(hard_inputs)
                        loss_hard = hard_pseudo_weight * sum(hard_loss_dict.values())
                        loss = loss + loss_hard

                if loss.requires_grad and float(loss.detach().cpu()) > 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if ema_update_interval > 0 and (global_step + 1) % ema_update_interval == 0:
                        _update_ema(teacher_model, student_model, ema_momentum)

                global_step += 1
                if scheduler_name == "cosine":
                    _set_optimizer_lr(optimizer, _cosine_lr_value(base_lr, global_step, total_steps))

                epoch_stats["loss_total"] += float(loss.detach().cpu())
                epoch_stats["loss_sup"] += float(loss_sup.detach().cpu())
                epoch_stats["loss_hard"] += float(loss_hard.detach().cpu())
                epoch_stats["steps"] += 1.0

                if log_period > 0 and global_step % log_period == 0:
                    from .utils import append_jsonl

                    append_jsonl(
                        train_log_path,
                        {
                            "step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_total": float(loss.detach().cpu()),
                            "loss_sup": float(loss_sup.detach().cpu()),
                            "loss_hard": float(loss_hard.detach().cpu()),
                            "num_labeled_images": int(len(labeled_batch)),
                            "num_unlabeled_images": int(len(unlabeled_batch)),
                            "num_hard_pseudo_boxes": int(hard_pseudo_count),
                        },
                    )

                if eval_period > 0 and global_step % eval_period == 0:
                    student_model.eval()
                    print(f"[DAOD][eval] step={global_step} split=source_val start")
                    source_val_metrics = _evaluate_split(self.cfg, student_model, "source_val", source_val_dicts)
                    print(f"[DAOD][eval] step={global_step} split=target_val start")
                    target_val_metrics = _evaluate_split(self.cfg, student_model, "target_val", target_val_dicts)
                    save_json(run_dir / "source_val_metrics.json", source_val_metrics)
                    save_json(run_dir / "target_val_metrics.json", target_val_metrics)
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
        source_val_metrics = _evaluate_split(self.cfg, student_model, "source_val", source_val_dicts)
        print(f"[DAOD][eval] step={global_step} split=target_val final")
        target_val_metrics = _evaluate_split(self.cfg, student_model, "target_val", target_val_dicts)
        save_json(run_dir / "train_history.json", {"history": history})
        save_json(run_dir / "source_val_metrics.json", source_val_metrics)
        save_json(run_dir / "target_val_metrics.json", target_val_metrics)
        maybe_empty_cuda_cache()
        return {
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt),
            "optimizer_checkpoint": str(optimizer_ckpt),
            "scheduler_checkpoint": None,
            "global_step": int(global_step),
            "train_history": history,
            "source_val_metrics": source_val_metrics,
            "target_val_metrics": target_val_metrics,
        }

    def train_domain_discriminator(
        self,
        *,
        teacher_checkpoint: str,
        run_dir: Path,
        source_samples: list[dict[str, Any]],
        target_samples: list[dict[str, Any]],
    ) -> tuple[DomainDiscriminator, dict[str, Any]]:
        disc_cfg = getattr(self.cfg.method, "discriminator", object())
        source_samples = _limit_samples(source_samples, getattr(disc_cfg, "max_source_samples", 0))
        target_samples = _limit_samples(target_samples, getattr(disc_cfg, "max_target_samples", 0))
        source_samples, target_samples = _balanced_source_target_samples(source_samples, target_samples)
        if not source_samples or not target_samples:
            raise RuntimeError("Domain discriminator training requires both source and target samples.")

        teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=self.aux_device)
        DetectionCheckpointer(teacher_adapter.model).load(str(teacher_checkpoint))
        teacher_adapter.model.eval()

        feature_batch_size = int(getattr(disc_cfg, "feature_batch_size", 1))
        source_feature_rows: list[torch.Tensor] = []
        target_feature_rows: list[torch.Tensor] = []
        for batch in _batched(source_samples, feature_batch_size):
            source_feature_rows.append(extract_pooled_backbone_features(teacher_adapter, batch, with_grad=False).float())
        for batch in _batched(target_samples, feature_batch_size):
            target_feature_rows.append(extract_pooled_backbone_features(teacher_adapter, batch, with_grad=False).float())
        del teacher_adapter
        maybe_empty_cuda_cache()
        source_features = torch.cat(source_feature_rows, dim=0)
        target_features = torch.cat(target_feature_rows, dim=0)
        features = torch.cat([source_features, target_features], dim=0)
        labels = torch.cat(
            [
                torch.zeros(source_features.shape[0], dtype=torch.float32),
                torch.ones(target_features.shape[0], dtype=torch.float32),
            ],
            dim=0,
        )

        dataset = TensorDataset(features, labels)
        loader = DataLoader(
            dataset,
            batch_size=int(getattr(disc_cfg, "batch_size", 32)),
            shuffle=True,
            num_workers=int(getattr(disc_cfg, "num_workers", 0)),
        )

        model = DomainDiscriminator(
            input_dim=int(features.shape[-1]),
            hidden_dim=int(getattr(disc_cfg, "hidden_dim", 256)),
            num_layers=int(getattr(disc_cfg, "num_layers", 3)),
            dropout=float(getattr(disc_cfg, "dropout", 0.1)),
        ).to(torch.device("cpu"))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(getattr(disc_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(disc_cfg, "weight_decay", 1e-4)),
        )

        history: list[dict[str, float]] = []
        for epoch_idx in range(1, int(getattr(disc_cfg, "epochs", 1)) + 1):
            model.train()
            epoch_loss = 0.0
            steps = 0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(torch.device("cpu"))
                batch_labels = batch_labels.to(torch.device("cpu"))
                logits = model(batch_features)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                steps += 1
            history.append(
                {
                    "epoch": float(epoch_idx),
                    "loss": epoch_loss / max(steps, 1),
                    "steps": float(steps),
                }
            )

        model.eval()
        ckpt_path = run_dir / "domain_discriminator.pt"
        torch.save(model.state_dict(), ckpt_path)
        summary = {
            "checkpoint": str(ckpt_path),
            "num_source_samples": len(source_samples),
            "num_target_samples": len(target_samples),
            "history": history,
        }
        save_json(run_dir / "domain_discriminator_summary.json", summary)
        return model, summary
