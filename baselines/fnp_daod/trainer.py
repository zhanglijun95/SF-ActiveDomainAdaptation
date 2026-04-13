"""Detector training loop for the isolated FNP baseline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.data.daod import DAODListDataset, build_daod_dataset, build_strong_view_sample, collate_daod_batch, cycle_daod_loader
from src.models import build_daod_model

from .dino_hooks import extract_pooled_backbone_features, mc_dropout_query_statistics
from .metrics import deduplicate_rows, rows_to_annotations
from .state import FNPDAODState
from .utils import append_jsonl, save_json


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambd, None


def _grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReverse.apply(x, lambd)


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


def _resolve_teacher_device(train_cfg: Any, student_device: torch.device) -> torch.device:
    teacher_device_cfg = getattr(train_cfg, "teacher_device", None)
    if student_device.type != "cuda" or not torch.cuda.is_available():
        return student_device
    if teacher_device_cfg is None:
        return student_device
    raw = str(teacher_device_cfg).strip().lower()
    if raw in {"", "same", "student"}:
        return student_device
    if raw == "auto":
        visible_count = torch.cuda.device_count()
        if visible_count <= 1:
            return student_device
        student_index = student_device.index if student_device.index is not None else torch.cuda.current_device()
        for idx in range(visible_count):
            if idx != student_index:
                return torch.device(f"cuda:{idx}")
        return student_device
    return torch.device(str(teacher_device_cfg))


def _build_loader(items: list[dict[str, Any]], *, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        DAODListDataset(items),
        batch_size=batch_size,
        shuffle=bool(items),
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


def _resize_pil_and_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    short_edge: int,
    max_size: int,
) -> tuple[Image.Image, list[list[float]], int, int]:
    width, height = image.size
    new_h, new_w = _resize_shape(height, width, short_edge, max_size)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    scale_x = float(new_w) / float(width)
    scale_y = float(new_h) / float(height)
    scaled_boxes = [
        [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
        for x0, y0, x1, y1 in boxes
    ]
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
    batch: list[dict[str, Any]],
    *,
    strong_short_edge: int,
    max_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for sample in batch:
        strong_sample = build_strong_view_sample(sample, suffix="fnp_supervised")
        image = strong_sample["image"]
        boxes = [ann["bbox"] for ann in sample["annotations"]]
        image, boxes, new_h, new_w = _resize_pil_and_boxes(image, boxes, strong_short_edge, max_size)
        resized_annotations = []
        for ann, box in zip(sample["annotations"], boxes):
            updated = dict(ann)
            updated["bbox"] = box
            resized_annotations.append(updated)
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


def _limit_samples(dataset: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or int(limit) <= 0:
        return dataset
    return dataset[: int(limit)]


def _bce_domain_loss(
    student_adapter,
    discriminator: DomainDiscriminator,
    *,
    source_batch: list[dict[str, Any]],
    target_batch: list[dict[str, Any]],
    grl_lambda: float,
) -> torch.Tensor:
    if not source_batch or not target_batch:
        return torch.tensor(0.0, device=next(discriminator.parameters()).device)

    source_features = extract_pooled_backbone_features(student_adapter, source_batch, with_grad=True)
    target_features = extract_pooled_backbone_features(student_adapter, target_batch, with_grad=True)
    source_logits = discriminator(_grad_reverse(source_features, grl_lambda))
    target_logits = discriminator(_grad_reverse(target_features, grl_lambda))
    loss_source = torch.nn.functional.binary_cross_entropy_with_logits(
        source_logits,
        torch.zeros_like(source_logits),
    )
    loss_target = torch.nn.functional.binary_cross_entropy_with_logits(
        target_logits,
        torch.ones_like(target_logits),
    )
    return loss_source + loss_target


def _update_ema(teacher_module: torch.nn.Module, student_module: torch.nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_module.parameters(), student_module.parameters()):
            student_value = student_param.data
            if student_value.device != teacher_param.data.device:
                student_value = student_value.to(teacher_param.data.device)
            teacher_param.data.mul_(momentum).add_(student_value, alpha=1.0 - momentum)


def _evaluate_split(cfg: Any, model: torch.nn.Module, split_name: str, dataset_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    from src.engine.daod_train_source import _evaluate_split as _eval_impl

    if not dataset_dicts:
        return {}
    return _eval_impl(cfg, model, split_name, dataset_dicts)


class FNPDAODTrainer:
    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def _feature_dim(self, student_adapter, source_train: list[dict[str, Any]], target_train: list[dict[str, Any]]) -> int:
        probe_batch = source_train[:1] or target_train[:1]
        if not probe_batch:
            return 2048
        pooled = extract_pooled_backbone_features(student_adapter, probe_batch[:1], with_grad=False)
        return int(pooled.shape[-1])

    def fit_stage(
        self,
        *,
        run_dir: Path,
        state_in: FNPDAODState,
        labeled_target_ids: set[str],
        stage_name: str,
        max_epochs: int,
        use_pseudo_labels: bool,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = getattr(self.cfg.method, "train", object())
        disc_cfg = getattr(self.cfg.method, "discriminator", object())
        teacher_device = _resolve_teacher_device(train_cfg, self.device)
        print(f"[FNP-DAOD][train] stage={stage_name} run_dir={run_dir}")

        student_adapter = build_daod_model(self.cfg, load_weights=False, device=self.device)
        teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=teacher_device)
        student_model = student_adapter.model.to(self.device)
        teacher_model = teacher_adapter.model.to(teacher_device)
        DetectionCheckpointer(student_model).load(str(state_in.student_checkpoint))
        DetectionCheckpointer(teacher_model).load(str(state_in.teacher_checkpoint))
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

        source_train_dataset = build_daod_dataset(self.cfg, split="source_train", transform=None)
        target_train_dataset = build_daod_dataset(self.cfg, split="target_train", transform=None)
        source_train = [dict(source_train_dataset[idx]) for idx in range(len(source_train_dataset))]
        target_train = [dict(target_train_dataset[idx]) for idx in range(len(target_train_dataset))]
        source_train = _limit_samples(source_train, getattr(train_cfg, "max_source_samples", 0))
        target_train = _limit_samples(target_train, getattr(train_cfg, "max_target_samples", 0))

        feature_dim = self._feature_dim(student_adapter, source_train, target_train)
        discriminator = DomainDiscriminator(
            input_dim=feature_dim,
            hidden_dim=int(getattr(disc_cfg, "hidden_dim", 256)),
            num_layers=int(getattr(disc_cfg, "num_layers", 3)),
            dropout=float(getattr(disc_cfg, "dropout", 0.1)),
        ).to(self.device)
        teacher_discriminator = DomainDiscriminator(
            input_dim=feature_dim,
            hidden_dim=int(getattr(disc_cfg, "hidden_dim", 256)),
            num_layers=int(getattr(disc_cfg, "num_layers", 3)),
            dropout=float(getattr(disc_cfg, "dropout", 0.1)),
        ).to(teacher_device)

        if state_in.discriminator_checkpoint:
            discriminator.load_state_dict(torch.load(state_in.discriminator_checkpoint, map_location=self.device))
        if state_in.teacher_discriminator_checkpoint:
            teacher_discriminator.load_state_dict(
                torch.load(state_in.teacher_discriminator_checkpoint, map_location=teacher_device)
            )
        else:
            teacher_discriminator.load_state_dict(discriminator.state_dict())

        for parameter in teacher_discriminator.parameters():
            parameter.requires_grad_(False)
        teacher_discriminator.eval()

        labeled_target = [sample for sample in target_train if sample["sample_id"] in labeled_target_ids]
        unlabeled_target = [sample for sample in target_train if sample["sample_id"] not in labeled_target_ids]
        print(
            "[FNP-DAOD][train] "
            f"stage={stage_name} "
            f"source_train={len(source_train)} "
            f"labeled_target={len(labeled_target)} "
            f"unlabeled_target={len(unlabeled_target)} "
            f"use_pseudo={bool(use_pseudo_labels)}"
        )

        num_workers = int(getattr(train_cfg, "num_workers", 0))
        source_loader = _build_loader(source_train, batch_size=int(getattr(train_cfg, "source_batch_size", 1)), num_workers=num_workers)
        labeled_loader = _build_loader(labeled_target, batch_size=int(getattr(train_cfg, "labeled_batch_size", 1)), num_workers=num_workers)
        unlabeled_loader = _build_loader(unlabeled_target, batch_size=int(getattr(train_cfg, "unlabeled_batch_size", 1)), num_workers=num_workers)
        source_iter = cycle_daod_loader(source_loader)
        labeled_iter = cycle_daod_loader(labeled_loader)
        unlabeled_iter = cycle_daod_loader(unlabeled_loader)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in student_model.parameters() if parameter.requires_grad]
            + [parameter for parameter in discriminator.parameters() if parameter.requires_grad],
            lr=float(getattr(train_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(train_cfg, "weight_decay", 1e-4)),
        )
        strong_short_edge = int(getattr(self.cfg.detector, "min_size_test", 800))
        max_size = int(getattr(self.cfg.detector, "max_size_test", 1333))
        ema_momentum = float(getattr(train_cfg, "ema_momentum", 0.999))
        lambda_adv = float(getattr(train_cfg, "lambda_adv", 0.01))
        grl_lambda = float(getattr(train_cfg, "grl_lambda", 1.0))
        pseudo_weight = float(getattr(train_cfg, "pseudo_weight", 1.0))
        pseudo_cfg = getattr(self.cfg.method, "pseudo", object())
        acquisition_cfg = getattr(self.cfg.method, "acquisition", object())
        max_steps_per_epoch = int(getattr(train_cfg, "max_steps_per_epoch", 0))

        steps_per_epoch = max(len(source_loader), len(labeled_loader), len(unlabeled_loader), 1)
        if max_steps_per_epoch > 0:
            steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)

        train_log_path = run_dir / "train_log.jsonl"
        history: list[dict[str, float]] = []
        global_step = int(getattr(state_in, "global_step", 0))

        for epoch_idx in range(1, int(max_epochs) + 1):
            student_model.train()
            discriminator.train()
            epoch_loss_total = 0.0
            steps_ran = 0
            for _ in range(steps_per_epoch):
                source_batch = next(source_iter, [])
                labeled_batch = next(labeled_iter, [])
                unlabeled_batch = next(unlabeled_iter, [])
                if not source_batch and not labeled_batch and not unlabeled_batch:
                    continue

                total_loss = torch.tensor(0.0, device=self.device)
                loss_source = torch.tensor(0.0, device=self.device)
                loss_target = torch.tensor(0.0, device=self.device)
                loss_pseudo = torch.tensor(0.0, device=self.device)
                loss_adv = torch.tensor(0.0, device=self.device)
                pseudo_box_count = 0

                if source_batch:
                    source_inputs = _make_supervised_inputs(
                        source_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                    )
                    loss_source = sum(student_model(source_inputs).values())
                    total_loss = total_loss + loss_source

                if labeled_batch:
                    labeled_inputs = _make_supervised_inputs(
                        labeled_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                    )
                    loss_target = sum(student_model(labeled_inputs).values())
                    total_loss = total_loss + loss_target

                domain_target_batch = unlabeled_batch if unlabeled_batch else labeled_batch
                if source_batch and domain_target_batch:
                    loss_adv = lambda_adv * _bce_domain_loss(
                        student_adapter,
                        discriminator,
                        source_batch=source_batch,
                        target_batch=domain_target_batch,
                        grl_lambda=grl_lambda,
                    )
                    total_loss = total_loss + loss_adv

                if use_pseudo_labels and unlabeled_batch:
                    pseudo_batch: list[dict[str, Any]] = []
                    for sample in unlabeled_batch:
                        stats = mc_dropout_query_statistics(
                            teacher_adapter,
                            sample,
                            num_passes=int(getattr(acquisition_cfg, "mc_dropout_passes_train", getattr(acquisition_cfg, "mc_dropout_passes", 4))),
                            dropout_rate=float(getattr(acquisition_cfg, "dropout_rate", 0.1)),
                            score_floor=float(getattr(pseudo_cfg, "score_floor", 0.1)),
                            max_queries=int(getattr(acquisition_cfg, "max_queries", 300)),
                        )
                        candidate_rows = [
                            row
                            for row in stats["rows"]
                            if float(row["score"]) >= float(getattr(pseudo_cfg, "score_min", 0.3))
                            and float(row["box_var"]) <= float(getattr(pseudo_cfg, "loc_var_max", 0.1))
                        ]
                        candidate_rows = deduplicate_rows(
                            candidate_rows,
                            iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                        )
                        if not candidate_rows:
                            continue
                        pseudo_sample = dict(sample)
                        pseudo_sample["annotations"] = rows_to_annotations(candidate_rows)
                        pseudo_batch.append(pseudo_sample)
                        pseudo_box_count += len(candidate_rows)

                    if pseudo_batch:
                        # We keep the pseudo-label path simple and isolated by
                        # reusing the detector's supervised loss on filtered
                        # pseudo boxes, which is the closest DINO analogue to
                        # the paper's uncertainty-gated pseudo supervision.
                        pseudo_inputs = _make_supervised_inputs(
                            pseudo_batch,
                            strong_short_edge=strong_short_edge,
                            max_size=max_size,
                            device=self.device,
                        )
                        loss_pseudo = pseudo_weight * sum(student_model(pseudo_inputs).values())
                        total_loss = total_loss + loss_pseudo

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                _update_ema(teacher_model, student_model, ema_momentum)
                _update_ema(teacher_discriminator, discriminator, ema_momentum)

                global_step += 1
                steps_ran += 1
                epoch_loss_total += float(total_loss.detach().cpu())
                append_jsonl(
                    train_log_path,
                    {
                        "stage": stage_name,
                        "epoch": int(epoch_idx),
                        "step": int(global_step),
                        "loss_total": float(total_loss.detach().cpu()),
                        "loss_source": float(loss_source.detach().cpu()),
                        "loss_target": float(loss_target.detach().cpu()),
                        "loss_pseudo": float(loss_pseudo.detach().cpu()),
                        "loss_adv": float(loss_adv.detach().cpu()),
                        "pseudo_box_count": int(pseudo_box_count),
                    },
                )

            history.append(
                {
                    "epoch": float(epoch_idx),
                    "loss_total": epoch_loss_total / max(steps_ran, 1),
                    "steps": float(steps_ran),
                }
            )

        student_ckpt = DetectionCheckpointer(student_model, str(run_dir)).save("student_last")
        teacher_ckpt = DetectionCheckpointer(teacher_model, str(run_dir)).save("teacher_last")
        disc_ckpt_path = run_dir / "domain_discriminator_last.pt"
        teacher_disc_ckpt_path = run_dir / "teacher_domain_discriminator_last.pt"
        torch.save(discriminator.state_dict(), disc_ckpt_path)
        torch.save(teacher_discriminator.state_dict(), teacher_disc_ckpt_path)

        method_eval_cfg = getattr(self.cfg.method, "eval", object())
        source_val_dataset = build_daod_dataset(self.cfg, split="source_val", transform=None)
        target_val_dataset = build_daod_dataset(self.cfg, split="target_val", transform=None)
        source_val = [dict(source_val_dataset[idx]) for idx in range(len(source_val_dataset))]
        target_val = [dict(target_val_dataset[idx]) for idx in range(len(target_val_dataset))]
        source_val = _limit_samples(source_val, getattr(method_eval_cfg, "source_val_limit", 0))
        target_val = _limit_samples(target_val, getattr(method_eval_cfg, "target_val_limit", 0))
        teacher_model.eval()
        source_val_metrics = _evaluate_split(self.cfg, teacher_model, "source_val", source_val)
        target_val_metrics = _evaluate_split(self.cfg, teacher_model, "target_val", target_val)

        summary = {
            "stage": stage_name,
            "epochs": int(max_epochs),
            "num_source_train": len(source_train),
            "num_labeled_target": len(labeled_target),
            "num_unlabeled_target": len(unlabeled_target),
            "use_pseudo_labels": bool(use_pseudo_labels),
            "global_step": int(global_step),
            "history": history,
            "source_val_metrics": source_val_metrics,
            "target_val_metrics": target_val_metrics,
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt),
            "discriminator_checkpoint": str(disc_ckpt_path),
            "teacher_discriminator_checkpoint": str(teacher_disc_ckpt_path),
        }
        save_json(run_dir / "summary.json", summary)
        return summary
