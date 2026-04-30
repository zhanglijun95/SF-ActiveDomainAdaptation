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
from src.engine.daod_gradient_surgery import (
    PCGradStats,
    add_grads_in_place,
    assign_grads,
    clone_grad_list,
    target_anchored_cagrad,
    target_anchored_l2rw,
    target_anchored_pcgrad,
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
from src.engine.daod_pseudo_recalibration import compute_pseudo_recalibration
from src.engine.daod_pseudo_score_calibration import (
    PseudoScoreCalibrator,
    apply_pseudo_score_calibrator_to_items,
    apply_pseudo_score_calibrator_to_thresholds,
    fit_pseudo_score_calibrator,
    pseudo_reliability_weight_for_rows,
    pseudo_reliability_weight_for_samples,
)
from src.engine.daod_teacher_guidance import (
    collect_grad_importance,
    importance_map_stats,
    merge_importance_maps,
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


def _without_annotations(sample: dict[str, Any]) -> dict[str, Any]:
    """Return an unlabeled target view so GT cannot leak into pseudo training."""

    cloned = dict(sample)
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
    if budget_k > 0:
        selected_indices = sorted(int(idx) for idx in rng.choice(len(target_train), size=budget_k, replace=False))
    else:
        selected_indices = []
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


def _threshold_dict_to_list(thresholds: dict[int, float], *, num_classes: int, default: float) -> list[float]:
    return [float(thresholds.get(class_id, default)) for class_id in range(num_classes)]


def _compute_recalibration_offsets(
    *,
    target_train: list[dict[str, Any]],
    selected_ids: set[str],
    num_classes: int,
    base_threshold: float,
    recalibration_cfg: Any,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    if not bool(getattr(recalibration_cfg, "enabled", False)) or not selected_ids:
        return [0.0 for _ in range(num_classes)], {
            "enabled": False,
            "offsets": [0.0 for _ in range(num_classes)],
            "class_counts": [0 for _ in range(num_classes)],
        }

    recalibrated_thresholds, stats = compute_pseudo_recalibration(
        target_train,
        selected_ids,
        num_classes=num_classes,
        base_score_min=base_threshold,
        recalibration_cfg=recalibration_cfg,
        teacher_adapter=None,
        stage_idx=0,
        seed=seed,
    )
    threshold_list = _threshold_dict_to_list(
        recalibrated_thresholds,
        num_classes=num_classes,
        default=base_threshold,
    )
    offsets = [float(base_threshold) - float(value) for value in threshold_list]
    return offsets, {
        "enabled": True,
        "base_threshold": float(base_threshold),
        "thresholds": threshold_list,
        "offsets": offsets,
        **stats,
    }


def _effective_thresholds(
    ddt_thresholds: list[float],
    offsets: list[float],
    *,
    pseudo_cfg: Any,
    recalibration_cfg: Any,
    base_threshold: float,
) -> list[float]:
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
    mask_stats: PCGradStats | None,
) -> dict[str, Any]:
    return {
        "grad_surgery_method": method,
        "grad_cos_pseudo_before": pseudo_stats.cosine_before if pseudo_stats is not None else None,
        "grad_cos_pseudo_after": pseudo_stats.cosine_after if pseudo_stats is not None else None,
        "grad_cos_mask_before": mask_stats.cosine_before if mask_stats is not None else None,
        "grad_cos_mask_after": mask_stats.cosine_after if mask_stats is not None else None,
        "grad_adjusted_pseudo": bool(pseudo_stats.projected) if pseudo_stats is not None else None,
        "grad_adjusted_mask": bool(mask_stats.projected) if mask_stats is not None else None,
        "grad_weight_pseudo": pseudo_stats.weight if pseudo_stats is not None else None,
        "grad_weight_mask": mask_stats.weight if mask_stats is not None else None,
    }


def _new_gradient_surgery_epoch_stats() -> dict[str, float]:
    return {
        "steps": 0.0,
        "pseudo_steps": 0.0,
        "mask_steps": 0.0,
        "pseudo_adjusted": 0.0,
        "mask_adjusted": 0.0,
        "pseudo_cos_before_sum": 0.0,
        "pseudo_cos_before_count": 0.0,
        "pseudo_cos_after_sum": 0.0,
        "pseudo_cos_after_count": 0.0,
        "pseudo_weight_sum": 0.0,
        "pseudo_weight_count": 0.0,
        "mask_cos_before_sum": 0.0,
        "mask_cos_before_count": 0.0,
        "mask_cos_after_sum": 0.0,
        "mask_cos_after_count": 0.0,
        "mask_weight_sum": 0.0,
        "mask_weight_count": 0.0,
    }


def _accumulate_gradient_surgery_branch(
    epoch_stats: dict[str, float],
    *,
    prefix: str,
    stats: PCGradStats | None,
) -> None:
    if stats is None:
        return
    epoch_stats[f"{prefix}_steps"] += 1.0
    if stats.projected:
        epoch_stats[f"{prefix}_adjusted"] += 1.0
    if stats.cosine_before is not None:
        epoch_stats[f"{prefix}_cos_before_sum"] += float(stats.cosine_before)
        epoch_stats[f"{prefix}_cos_before_count"] += 1.0
    if stats.cosine_after is not None:
        epoch_stats[f"{prefix}_cos_after_sum"] += float(stats.cosine_after)
        epoch_stats[f"{prefix}_cos_after_count"] += 1.0
    if stats.weight is not None:
        epoch_stats[f"{prefix}_weight_sum"] += float(stats.weight)
        epoch_stats[f"{prefix}_weight_count"] += 1.0


def _gradient_surgery_epoch_summary(
    epoch_stats: dict[str, float],
    *,
    enabled: bool,
    method: str,
) -> dict[str, Any]:
    def _mean(prefix: str, suffix: str) -> float | None:
        count_key = f"{prefix}_cos_{suffix}_count" if suffix in {"before", "after"} else f"{prefix}_{suffix}_count"
        sum_key = f"{prefix}_cos_{suffix}_sum" if suffix in {"before", "after"} else f"{prefix}_{suffix}_sum"
        count = epoch_stats[count_key]
        if count <= 0:
            return None
        return float(epoch_stats[sum_key] / count)

    def _rate(prefix: str) -> float | None:
        steps = epoch_stats[f"{prefix}_steps"]
        if steps <= 0:
            return None
        return float(epoch_stats[f"{prefix}_adjusted"] / steps)

    return {
        "enabled": bool(enabled),
        "method": method if enabled else None,
        "steps": int(epoch_stats["steps"]),
        "pseudo_adjustment_rate": _rate("pseudo"),
        "mask_adjustment_rate": _rate("mask"),
        "mean_cos_pseudo_before": _mean("pseudo", "before"),
        "mean_cos_pseudo_after": _mean("pseudo", "after"),
        "mean_weight_pseudo": _mean("pseudo", "weight"),
        "mean_cos_mask_before": _mean("mask", "before"),
        "mean_cos_mask_after": _mean("mask", "after"),
        "mean_weight_mask": _mean("mask", "weight"),
    }


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
        active_cfg = getattr(method_cfg, "active", object())
        recalibration_cfg = getattr(method_cfg, "pseudo_recalibration", object())
        score_calibration_cfg = getattr(method_cfg, "pseudo_score_calibration", object())
        pseudo_reliability_cfg = getattr(method_cfg, "pseudo_reliability_weighting", object())
        pseudo_reliability_enabled = bool(getattr(pseudo_reliability_cfg, "enabled", False))
        label_guided_cfg = getattr(method_cfg, "label_guided_aema", object())
        label_guided_enabled = bool(getattr(label_guided_cfg, "enabled", False))
        gradient_surgery_cfg = getattr(method_cfg, "gradient_surgery", object())
        gradient_surgery_enabled = bool(getattr(gradient_surgery_cfg, "enabled", False))
        gradient_surgery_method = str(
            getattr(gradient_surgery_cfg, "method", "target_anchored_pcgrad")
        ).strip().lower()
        gradient_surgery_apply_pseudo = bool(getattr(gradient_surgery_cfg, "apply_to_pseudo", True))
        gradient_surgery_apply_masked = bool(getattr(gradient_surgery_cfg, "apply_to_masked", True))
        gradient_surgery_eps = max(0.0, float(getattr(gradient_surgery_cfg, "eps", 1e-12)))
        cagrad_c = float(getattr(gradient_surgery_cfg, "c", 0.4))
        cagrad_rescale = int(getattr(gradient_surgery_cfg, "rescale", 1))
        cagrad_sum_scale = bool(getattr(gradient_surgery_cfg, "sum_scale", True))
        l2rw_min_weight = float(getattr(gradient_surgery_cfg, "min_weight", 0.25))
        l2rw_max_weight = float(getattr(gradient_surgery_cfg, "max_weight", 1.0))
        supported_surgery_methods = {
            "target_anchored_pcgrad",
            "target_anchored_cagrad",
            "target_anchored_l2rw",
        }
        if gradient_surgery_enabled and gradient_surgery_method not in supported_surgery_methods:
            raise ValueError(
                "method.gradient_surgery.method must be one of "
                f"{sorted(supported_surgery_methods)}, got {gradient_surgery_method!r}"
            )
        if (
            gradient_surgery_enabled
            and gradient_surgery_method == "target_anchored_cagrad"
            and gradient_surgery_apply_masked
        ):
            raise ValueError("target_anchored_cagrad currently supports apply_to_masked=false.")
        seed = int(getattr(self.cfg, "seed", 42))

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
        base_threshold = float(getattr(pseudo_cfg, "threshold", 0.4))
        thresholds = [base_threshold] * num_classes
        recalibration_offsets, recalibration_stats = _compute_recalibration_offsets(
            target_train=target_train,
            selected_ids=selected_ids,
            num_classes=num_classes,
            base_threshold=base_threshold,
            recalibration_cfg=recalibration_cfg,
            seed=seed,
        )
        if bool(recalibration_stats.get("enabled", False)):
            save_json(run_dir / "pseudo_recalibration.json", recalibration_stats)
        score_calibration_enabled = bool(getattr(score_calibration_cfg, "enabled", False))
        score_calibration_replace_score = bool(getattr(score_calibration_cfg, "replace_score", True))
        score_calibration_map_thresholds = bool(getattr(score_calibration_cfg, "map_thresholds", False))
        score_calibration_dynamic_update_space = str(
            getattr(score_calibration_cfg, "dynamic_update_space", "calibrated")
        ).strip().lower()
        if score_calibration_dynamic_update_space not in {"raw", "calibrated"}:
            raise ValueError(
                "method.pseudo_score_calibration.dynamic_update_space must be 'raw' or 'calibrated', "
                f"got {score_calibration_dynamic_update_space!r}"
            )
        if score_calibration_map_thresholds and not score_calibration_replace_score:
            raise ValueError(
                "method.pseudo_score_calibration.map_thresholds requires replace_score=true; "
                "use map_thresholds=false when calibrated scores are only used for reliability weighting."
            )
        score_calibrator: PseudoScoreCalibrator | None = None
        score_calibration_stats: dict[str, Any] = {}
        if score_calibration_enabled:
            score_calibrator, score_calibration_stats = fit_pseudo_score_calibrator(
                target_train,
                selected_ids,
                num_classes=num_classes,
                calibration_cfg=score_calibration_cfg,
                teacher_adapter=teacher_adapter,
                stage_idx=0,
                seed=seed,
            )
            save_json(run_dir / "pseudo_score_calibration.json", score_calibration_stats)
        weak_rng = np.random.default_rng(seed)
        strong_short_edge = int(getattr(self.cfg.detector, "min_size_test", 800))
        max_size = int(getattr(self.cfg.detector, "max_size_test", 1333))
        coef_masked_img = float(getattr(mask_cfg, "coef_masked_img", 1.0))
        supervised_weight = float(getattr(active_cfg, "supervised_weight", 1.0))
        alpha_ema = float(getattr(aema_cfg, "alpha_ema", 0.9996))
        alpha_aema = float(getattr(aema_cfg, "alpha_aema", 0.997))
        top_fraction = float(getattr(aema_cfg, "top_fraction", 0.1))
        update_interval = int(getattr(aema_cfg, "update_interval", 2))
        use_teacher_grad = bool(getattr(aema_cfg, "use_teacher_grad_importance", True))
        label_guided_merge = str(getattr(label_guided_cfg, "merge", "max")).strip().lower()
        label_guided_weight = float(getattr(label_guided_cfg, "guidance_weight", 1.0))
        label_guided_normalize = bool(getattr(label_guided_cfg, "normalize", True))
        label_guided_loss_mode = str(getattr(label_guided_cfg, "loss", "full")).strip().lower()
        label_guided_loss_weight = float(getattr(label_guided_cfg, "supervised_loss_weight", 1.0))
        label_guided_top_fraction = float(getattr(label_guided_cfg, "top_fraction", top_fraction))
        if label_guided_merge not in {"max", "add", "gt_only", "base_only"}:
            raise ValueError(
                "method.label_guided_aema.merge must be one of max/add/gt_only/base_only, "
                f"got {label_guided_merge!r}"
            )
        if label_guided_loss_mode not in {"full", "class"}:
            raise ValueError(
                "method.label_guided_aema.loss must be 'full' or 'class', "
                f"got {label_guided_loss_mode!r}"
            )
        use_dynamic_threshold = bool(getattr(pseudo_cfg, "dynamic", True))
        log_period = int(getattr(train_cfg, "log_period", 100))
        checkpoint_period = int(getattr(train_cfg, "checkpoint_period", 0))
        train_log_path = run_dir / "train_log.jsonl"

        print(
            "[DDT-DAOD][train] "
            f"epochs={epochs} target_train={len(target_train)} "
            f"labeled_target={len(target_labeled)} unlabeled_target={len(target_unlabeled)} "
            f"student_device={self.device} teacher_device={teacher_device} "
            f"source_ckpt={source_checkpoint}"
        )
        print(
            "[DDT-DAOD][ddt] "
            f"threshold={thresholds[0]:.3f} dynamic={use_dynamic_threshold} "
            f"mask_ratio={float(getattr(mask_cfg, 'masked_ratio', 0.5)):.3f} "
            f"alpha_ema={alpha_ema:.6f} alpha_aema={alpha_aema:.6f}"
        )
        if bool(getattr(recalibration_cfg, "enabled", False)):
            print(
                "[DDT-DAOD][recalibration] "
                f"method={recalibration_stats.get('method')} "
                f"offsets={recalibration_stats.get('offsets')} "
                f"class_counts={recalibration_stats.get('class_counts')}"
            )
        if score_calibration_enabled:
            print(
                "[DDT-DAOD][score_calibration] "
                f"method={score_calibration_stats.get('method')} "
                f"fallback={score_calibration_stats.get('fallback_to_identity')} "
                f"examples={score_calibration_stats.get('num_examples')} "
                f"replace_score={score_calibration_replace_score} "
                f"map_thresholds={score_calibration_map_thresholds} "
                f"dynamic_update_space={score_calibration_dynamic_update_space}"
            )
        if pseudo_reliability_enabled:
            print(
                "[DDT-DAOD][pseudo_reliability] "
                f"score_key={getattr(pseudo_reliability_cfg, 'score_key', 'calibrated_score')} "
                f"min_weight={float(getattr(pseudo_reliability_cfg, 'min_weight', 0.25)):.3f} "
                f"power={float(getattr(pseudo_reliability_cfg, 'power', 1.0)):.3f}"
            )
        if label_guided_enabled:
            print(
                "[DDT-DAOD][label_guided_aema] "
                f"merge={label_guided_merge} guidance_weight={label_guided_weight:.3f} "
                f"normalize={label_guided_normalize} loss={label_guided_loss_mode} "
                f"top_fraction={label_guided_top_fraction:.3f}"
            )
        if gradient_surgery_enabled:
            print(
                "[DDT-DAOD][gradient_surgery] "
                f"method={gradient_surgery_method} apply_to_pseudo={gradient_surgery_apply_pseudo} "
                f"apply_to_masked={gradient_surgery_apply_masked}"
            )

        global_step = 0
        history: list[dict[str, float]] = []
        for epoch_idx in range(1, epochs + 1):
            student_model.train()
            teacher_model.eval()
            epoch_loss = 0.0
            epoch_loss_pseudo = 0.0
            epoch_loss_mask = 0.0
            epoch_loss_supervised = 0.0
            epoch_pseudo_boxes = 0
            epoch_pseudo_reliability_weight = 0.0
            epoch_pseudo_reliability_steps = 0
            epoch_label_guided_teacher_loss = 0.0
            epoch_label_guided_teacher_steps = 0
            epoch_pseudo_importance_steps = 0
            epoch_gt_importance_steps = 0
            epoch_aema_update_steps = 0
            epoch_ema_update_steps = 0
            epoch_gradient_surgery_stats = _new_gradient_surgery_epoch_stats()
            score_sums = [0.0] * num_classes
            score_counts = [0] * num_classes

            for _ in range(steps_per_epoch):
                batch = next(target_iter, [])
                labeled_batch = next(labeled_iter, [])
                effective_thresholds_raw = _effective_thresholds(
                    thresholds,
                    recalibration_offsets,
                    pseudo_cfg=pseudo_cfg,
                    recalibration_cfg=recalibration_cfg,
                    base_threshold=base_threshold,
                )
                effective_thresholds = (
                    apply_pseudo_score_calibrator_to_thresholds(effective_thresholds_raw, score_calibrator)
                    if score_calibration_enabled and score_calibration_map_thresholds
                    else [float(value) for value in effective_thresholds_raw]
                )

                pseudo_batch: list[dict[str, Any]] = []
                pseudo_box_count = 0
                if batch:
                    with _device_context(teacher_device), torch.no_grad():
                        teacher_items = _teacher_outputs_for_unlabeled(
                            teacher_adapter,
                            batch,
                            weak_view_rng=weak_rng,
                        )
                    if score_calibration_enabled:
                        teacher_items = apply_pseudo_score_calibrator_to_items(
                            teacher_items,
                            score_calibrator,
                            replace_score=score_calibration_replace_score,
                        )

                    for teacher_item in teacher_items:
                        pseudo_rows = filter_pseudo_rows(
                            teacher_item["query_rows"],
                            thresholds=effective_thresholds,
                            dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
                        )
                        for row in pseudo_rows:
                            class_id = int(row["category_id"])
                            update_score = (
                                float(row.get("raw_score", row["score"]))
                                if score_calibration_dynamic_update_space == "raw"
                                else float(row["score"])
                            )
                            score_sums[class_id] += update_score
                            score_counts[class_id] += 1
                        annotations = rows_to_annotations(pseudo_rows)
                        if not annotations:
                            continue
                        pseudo_reliability_weight = 1.0
                        if pseudo_reliability_enabled:
                            pseudo_reliability_weight, _ = pseudo_reliability_weight_for_rows(
                                pseudo_rows,
                                pseudo_reliability_cfg,
                                thresholds=effective_thresholds,
                            )
                        pseudo_sample = dict(teacher_item["sample"])
                        pseudo_sample["annotations"] = annotations
                        pseudo_sample["_pseudo_reliability_weight"] = pseudo_reliability_weight
                        pseudo_sample["_pseudo_reliability_num_rows"] = len(pseudo_rows)
                        pseudo_batch.append(pseudo_sample)
                        pseudo_box_count += len(annotations)

                loss_terms: list[torch.Tensor] = []
                loss_pseudo_value = 0.0
                loss_mask_value = 0.0
                loss_supervised_value = 0.0
                loss_pseudo: torch.Tensor | None = None
                loss_mask: torch.Tensor | None = None
                loss_supervised: torch.Tensor | None = None
                pseudo_reliability_weight = 1.0
                pseudo_stats: PCGradStats | None = None
                mask_stats: PCGradStats | None = None
                if pseudo_batch:
                    pseudo_reliability_weight = (
                        pseudo_reliability_weight_for_samples(pseudo_batch)
                        if pseudo_reliability_enabled
                        else 1.0
                    )
                    pseudo_inputs = _make_supervised_inputs(
                        student_adapter,
                        pseudo_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                    )
                    loss_pseudo = pseudo_reliability_weight * sum(student_model(pseudo_inputs).values())
                    masked_inputs = apply_block_mask_to_inputs(
                        pseudo_inputs,
                        block_size=int(getattr(mask_cfg, "block_size", 64)),
                        masked_ratio=float(getattr(mask_cfg, "masked_ratio", 0.5)),
                    )
                    loss_mask = pseudo_reliability_weight * coef_masked_img * sum(student_model(masked_inputs).values())
                    loss_terms.extend([loss_pseudo, loss_mask])
                    loss_pseudo_value = float(loss_pseudo.detach().cpu())
                    loss_mask_value = float(loss_mask.detach().cpu())
                    epoch_pseudo_reliability_weight += float(pseudo_reliability_weight)
                    epoch_pseudo_reliability_steps += 1

                if labeled_batch:
                    supervised_inputs = _make_supervised_inputs(
                        student_adapter,
                        labeled_batch,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        device=self.device,
                    )
                    loss_supervised = supervised_weight * sum(student_model(supervised_inputs).values())
                    loss_terms.append(loss_supervised)
                    loss_supervised_value = float(loss_supervised.detach().cpu())

                if not loss_terms:
                    global_step += 1
                    scheduler.step()
                    continue

                loss = sum(loss_terms)

                optimizer.zero_grad(set_to_none=True)
                gradient_surgery_can_run = (
                    gradient_surgery_enabled
                    and loss_supervised is not None
                    and loss_supervised.requires_grad
                    and (loss_pseudo is not None or loss_mask is not None)
                )
                if gradient_surgery_can_run:
                    supervised_grads = _loss_grads(
                        loss_supervised,
                        trainable_params,
                        retain_graph=False,
                    )
                    combined_grads = clone_grad_list(supervised_grads)
                    if loss_pseudo is not None:
                        pseudo_grads = _loss_grads(
                            loss_pseudo,
                            trainable_params,
                            retain_graph=False,
                        )
                        if gradient_surgery_apply_pseudo:
                            if gradient_surgery_method == "target_anchored_pcgrad":
                                pseudo_grads, pseudo_stats = target_anchored_pcgrad(
                                    anchor_grads=supervised_grads,
                                    aux_grads=pseudo_grads,
                                    eps=gradient_surgery_eps,
                                )
                                add_grads_in_place(combined_grads, pseudo_grads)
                            elif gradient_surgery_method == "target_anchored_l2rw":
                                pseudo_grads, pseudo_stats = target_anchored_l2rw(
                                    anchor_grads=supervised_grads,
                                    aux_grads=pseudo_grads,
                                    min_weight=l2rw_min_weight,
                                    max_weight=l2rw_max_weight,
                                    eps=gradient_surgery_eps,
                                )
                                add_grads_in_place(combined_grads, pseudo_grads)
                            elif gradient_surgery_method == "target_anchored_cagrad":
                                combined_grads, pseudo_stats = target_anchored_cagrad(
                                    anchor_grads=supervised_grads,
                                    aux_grads=pseudo_grads,
                                    c=cagrad_c,
                                    rescale=cagrad_rescale,
                                    sum_scale=cagrad_sum_scale,
                                    eps=gradient_surgery_eps,
                                )
                        else:
                            add_grads_in_place(combined_grads, pseudo_grads)
                    if loss_mask is not None:
                        mask_grads = _loss_grads(
                            loss_mask,
                            trainable_params,
                            retain_graph=False,
                        )
                        if gradient_surgery_apply_masked:
                            if gradient_surgery_method == "target_anchored_pcgrad":
                                mask_grads, mask_stats = target_anchored_pcgrad(
                                    anchor_grads=supervised_grads,
                                    aux_grads=mask_grads,
                                    eps=gradient_surgery_eps,
                                )
                            elif gradient_surgery_method == "target_anchored_l2rw":
                                mask_grads, mask_stats = target_anchored_l2rw(
                                    anchor_grads=supervised_grads,
                                    aux_grads=mask_grads,
                                    min_weight=l2rw_min_weight,
                                    max_weight=l2rw_max_weight,
                                    eps=gradient_surgery_eps,
                                )
                        add_grads_in_place(combined_grads, mask_grads)
                    assign_grads(trainable_params, combined_grads)
                    epoch_gradient_surgery_stats["steps"] += 1.0
                    _accumulate_gradient_surgery_branch(
                        epoch_gradient_surgery_stats,
                        prefix="pseudo",
                        stats=pseudo_stats,
                    )
                    _accumulate_gradient_surgery_branch(
                        epoch_gradient_surgery_stats,
                        prefix="mask",
                        stats=mask_stats,
                    )
                else:
                    loss.backward()
                if float(getattr(train_cfg, "clip_max_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        float(getattr(train_cfg, "clip_max_norm", 0.0)),
                    )
                optimizer.step()

                pseudo_importance: dict[str, torch.Tensor] = {}
                gt_importance: dict[str, torch.Tensor] = {}
                label_guided_teacher_loss_value = 0.0
                grad_importance: dict[str, torch.Tensor] = {}
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
                        loss_weight=pseudo_reliability_weight,
                    )
                    if pseudo_importance:
                        epoch_pseudo_importance_steps += 1
                    grad_importance = pseudo_importance
                if teacher_update_due and label_guided_enabled and labeled_batch:
                    gt_importance, label_guided_teacher_loss_value = _teacher_grad_importance_from_supervised_batch(
                        teacher_model=teacher_model,
                        teacher_adapter=teacher_adapter,
                        batch=labeled_batch,
                        teacher_device=teacher_device,
                        strong_short_edge=strong_short_edge,
                        max_size=max_size,
                        loss_mode=label_guided_loss_mode,
                        loss_weight=label_guided_loss_weight,
                    )
                    if gt_importance:
                        epoch_gt_importance_steps += 1
                        epoch_label_guided_teacher_loss += float(label_guided_teacher_loss_value)
                        epoch_label_guided_teacher_steps += 1
                    grad_importance = merge_importance_maps(
                        pseudo_importance,
                        gt_importance,
                        merge=label_guided_merge,
                        guidance_weight=label_guided_weight,
                        normalize=label_guided_normalize,
                    )

                if teacher_update_due:
                    if grad_importance:
                        _update_aema(
                            teacher_model,
                            student_model,
                            grad_importance,
                            momentum=alpha_ema,
                            adaptive_momentum=alpha_aema,
                            top_fraction=label_guided_top_fraction if label_guided_enabled else top_fraction,
                        )
                        epoch_aema_update_steps += 1
                    else:
                        _update_ema(teacher_model, student_model, alpha_ema)
                        epoch_ema_update_steps += 1

                global_step += 1
                scheduler.step()
                epoch_loss += float(loss.detach().cpu())
                epoch_loss_pseudo += loss_pseudo_value
                epoch_loss_mask += loss_mask_value
                epoch_loss_supervised += loss_supervised_value
                epoch_pseudo_boxes += int(pseudo_box_count)

                if log_period > 0 and global_step % log_period == 0:
                    append_jsonl(
                        train_log_path,
                        {
                            "epoch": int(epoch_idx),
                            "step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss_total": float(loss.detach().cpu()),
                            "loss_pseudo": loss_pseudo_value,
                            "loss_masked": loss_mask_value,
                            "loss_supervised": loss_supervised_value,
                            "pseudo_box_count": int(pseudo_box_count),
                            "ddt_thresholds": [float(v) for v in thresholds],
                            "effective_thresholds_raw": [float(v) for v in effective_thresholds_raw],
                            "effective_thresholds": [float(v) for v in effective_thresholds],
                            "recalibration_offsets": [float(v) for v in recalibration_offsets],
                            "pseudo_score_calibration_enabled": bool(score_calibration_enabled),
                            "pseudo_score_calibration_method": score_calibration_stats.get("method")
                            if score_calibration_enabled
                            else None,
                            "pseudo_score_calibration_fallback": bool(score_calibration_stats.get("fallback_to_identity", False))
                            if score_calibration_enabled
                            else None,
                            "pseudo_score_calibration_map_thresholds": bool(score_calibration_map_thresholds)
                            if score_calibration_enabled
                            else None,
                            "pseudo_score_calibration_replace_score": bool(score_calibration_replace_score)
                            if score_calibration_enabled
                            else None,
                            "pseudo_score_calibration_dynamic_update_space": score_calibration_dynamic_update_space
                            if score_calibration_enabled
                            else None,
                            "pseudo_reliability_weighting_enabled": bool(pseudo_reliability_enabled),
                            "pseudo_reliability_weight": float(pseudo_reliability_weight)
                            if pseudo_reliability_enabled and pseudo_batch
                            else None,
                            "label_guided_aema_enabled": bool(label_guided_enabled),
                            "label_guided_aema_merge": label_guided_merge if label_guided_enabled else None,
                            "label_guided_teacher_loss": float(label_guided_teacher_loss_value)
                            if label_guided_enabled and gt_importance
                            else None,
                            "teacher_importance_has_pseudo": bool(pseudo_importance),
                            "teacher_importance_has_gt": bool(gt_importance),
                            "teacher_importance_stats": importance_map_stats(grad_importance)
                            if grad_importance
                            else None,
                            **(
                                _gradient_surgery_log_fields(
                                    method=gradient_surgery_method,
                                    pseudo_stats=pseudo_stats,
                                    mask_stats=mask_stats,
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
                )
                print(f"[DDT-DAOD][thresholds] epoch={epoch_idx} values={thresholds}")

            denom = max(steps_per_epoch, 1)
            epoch_summary = {
                "epoch": float(epoch_idx),
                "loss_total": epoch_loss / denom,
                "loss_pseudo": epoch_loss_pseudo / denom,
                "loss_masked": epoch_loss_mask / denom,
                "loss_supervised": epoch_loss_supervised / denom,
                "pseudo_boxes": float(epoch_pseudo_boxes),
                "ddt_thresholds": [float(v) for v in thresholds],
                "effective_thresholds_raw": _effective_thresholds(
                    thresholds,
                    recalibration_offsets,
                    pseudo_cfg=pseudo_cfg,
                    recalibration_cfg=recalibration_cfg,
                    base_threshold=base_threshold,
                ),
                "effective_thresholds": (
                    apply_pseudo_score_calibrator_to_thresholds(
                        _effective_thresholds(
                            thresholds,
                            recalibration_offsets,
                            pseudo_cfg=pseudo_cfg,
                            recalibration_cfg=recalibration_cfg,
                            base_threshold=base_threshold,
                        ),
                        score_calibrator,
                    )
                    if score_calibration_enabled and score_calibration_map_thresholds
                    else _effective_thresholds(
                        thresholds,
                        recalibration_offsets,
                        pseudo_cfg=pseudo_cfg,
                        recalibration_cfg=recalibration_cfg,
                        base_threshold=base_threshold,
                    )
                ),
                "recalibration_offsets": [float(v) for v in recalibration_offsets],
                "pseudo_score_calibration": score_calibration_stats if score_calibration_enabled else {},
                "pseudo_reliability_weighting": {
                    "enabled": bool(pseudo_reliability_enabled),
                    "average_weight": float(epoch_pseudo_reliability_weight / max(epoch_pseudo_reliability_steps, 1))
                    if pseudo_reliability_enabled
                    else None,
                    "steps": int(epoch_pseudo_reliability_steps),
                },
                "label_guided_aema": {
                    "enabled": bool(label_guided_enabled),
                    "merge": label_guided_merge if label_guided_enabled else None,
                    "guidance_weight": float(label_guided_weight) if label_guided_enabled else None,
                    "loss": label_guided_loss_mode if label_guided_enabled else None,
                    "average_teacher_loss": float(
                        epoch_label_guided_teacher_loss / max(epoch_label_guided_teacher_steps, 1)
                    )
                    if label_guided_enabled
                    else None,
                    "teacher_loss_steps": int(epoch_label_guided_teacher_steps),
                    "pseudo_importance_steps": int(epoch_pseudo_importance_steps),
                    "gt_importance_steps": int(epoch_gt_importance_steps),
                    "aema_update_steps": int(epoch_aema_update_steps),
                    "ema_update_steps": int(epoch_ema_update_steps),
                },
                "gradient_surgery": _gradient_surgery_epoch_summary(
                    epoch_gradient_surgery_stats,
                    enabled=gradient_surgery_enabled,
                    method=gradient_surgery_method,
                ),
            }
            history.append(epoch_summary)
            append_jsonl(run_dir / "epoch_log.jsonl", epoch_summary)

        student_ckpt = run_dir / "student_last.pth"
        teacher_ckpt = run_dir / "teacher_last.pth"
        DetectionCheckpointer(student_model, save_dir=str(run_dir)).save("student_last")
        DetectionCheckpointer(teacher_model, save_dir=str(run_dir)).save("teacher_last")

        target_val = _limit_samples(materialize_daod_dicts(self.cfg, "target_val"), getattr(eval_cfg, "target_val_limit", 0))
        evaluate_teacher = bool(getattr(eval_cfg, "evaluate_teacher", True))
        teacher_source_metrics: dict[str, Any] = {}
        teacher_target_metrics: dict[str, Any] = {}
        teacher_eval_error: str | None = None
        teacher_model.eval()
        student_model.eval()
        print(f"[DDT-DAOD][eval] step={global_step} split=target_val student")
        student_target_metrics = _evaluate_split(self.cfg, student_model, "target_val", target_val)
        if evaluate_teacher:
            try:
                source_val = _limit_samples(
                    materialize_daod_dicts(self.cfg, "source_val"),
                    getattr(eval_cfg, "source_val_limit", 0),
                )
                with _device_context(teacher_device):
                    print(f"[DDT-DAOD][eval] step={global_step} split=source_val teacher")
                    teacher_source_metrics = _evaluate_split(self.cfg, teacher_model, "source_val", source_val)
                    print(f"[DDT-DAOD][eval] step={global_step} split=target_val teacher")
                    teacher_target_metrics = _evaluate_split(self.cfg, teacher_model, "target_val", target_val)
            except RuntimeError as exc:
                teacher_eval_error = str(exc)
                print(f"[DDT-DAOD][eval][warning] teacher eval failed; using student final metrics: {exc}")
        final_model = "teacher" if teacher_target_metrics else "student"
        final_checkpoint = teacher_ckpt if teacher_target_metrics else student_ckpt
        final_target_metrics = teacher_target_metrics if teacher_target_metrics else student_target_metrics

        save_json(run_dir / "target_val_metrics.json", final_target_metrics)
        save_json(run_dir / "student_target_val_metrics.json", student_target_metrics)
        save_json(run_dir / "teacher_target_val_metrics.json", teacher_target_metrics)
        save_json(run_dir / "source_val_metrics.json", teacher_source_metrics)

        summary = {
            "epochs": int(epochs),
            "global_step": int(global_step),
            "source_checkpoint": str(source_checkpoint),
            "final_model": final_model,
            "final_checkpoint": str(final_checkpoint),
            "student_checkpoint": str(student_ckpt),
            "teacher_checkpoint": str(teacher_ckpt),
            "history": history,
            "teacher_source_val_metrics": teacher_source_metrics,
            "teacher_target_val_metrics": teacher_target_metrics,
            "teacher_eval_error": teacher_eval_error,
            "student_target_val_metrics": student_target_metrics,
            "final_target_val_metrics": final_target_metrics,
            "active_plan": active_plan,
            "pseudo_recalibration": recalibration_stats,
            "pseudo_score_calibration": score_calibration_stats if score_calibration_enabled else {},
            "pseudo_reliability_weighting": {
                "enabled": bool(pseudo_reliability_enabled),
                "score_key": getattr(pseudo_reliability_cfg, "score_key", None) if pseudo_reliability_enabled else None,
                "min_weight": float(getattr(pseudo_reliability_cfg, "min_weight", 0.25))
                if pseudo_reliability_enabled
                else None,
                "max_weight": float(getattr(pseudo_reliability_cfg, "max_weight", 1.0))
                if pseudo_reliability_enabled
                else None,
                "power": float(getattr(pseudo_reliability_cfg, "power", 1.0)) if pseudo_reliability_enabled else None,
            },
            "label_guided_aema": {
                "enabled": bool(label_guided_enabled),
                "merge": label_guided_merge if label_guided_enabled else None,
                "guidance_weight": float(label_guided_weight) if label_guided_enabled else None,
                "normalize": bool(label_guided_normalize) if label_guided_enabled else None,
                "loss": label_guided_loss_mode if label_guided_enabled else None,
                "supervised_loss_weight": float(label_guided_loss_weight) if label_guided_enabled else None,
                "top_fraction": float(label_guided_top_fraction) if label_guided_enabled else None,
            },
            "gradient_surgery": {
                "enabled": bool(gradient_surgery_enabled),
                "method": gradient_surgery_method if gradient_surgery_enabled else None,
                "apply_to_pseudo": bool(gradient_surgery_apply_pseudo) if gradient_surgery_enabled else None,
                "apply_to_masked": bool(gradient_surgery_apply_masked) if gradient_surgery_enabled else None,
                "c": float(cagrad_c)
                if gradient_surgery_enabled and gradient_surgery_method == "target_anchored_cagrad"
                else None,
                "rescale": int(cagrad_rescale)
                if gradient_surgery_enabled and gradient_surgery_method == "target_anchored_cagrad"
                else None,
                "sum_scale": bool(cagrad_sum_scale)
                if gradient_surgery_enabled and gradient_surgery_method == "target_anchored_cagrad"
                else None,
                "min_weight": float(l2rw_min_weight)
                if gradient_surgery_enabled and gradient_surgery_method == "target_anchored_l2rw"
                else None,
                "max_weight": float(l2rw_max_weight)
                if gradient_surgery_enabled and gradient_surgery_method == "target_anchored_l2rw"
                else None,
            },
            "thresholds": [float(v) for v in thresholds],
            "ddt_thresholds": [float(v) for v in thresholds],
            "effective_thresholds_raw": _effective_thresholds(
                thresholds,
                recalibration_offsets,
                pseudo_cfg=pseudo_cfg,
                recalibration_cfg=recalibration_cfg,
                base_threshold=base_threshold,
            ),
            "effective_thresholds": (
                apply_pseudo_score_calibrator_to_thresholds(
                    _effective_thresholds(
                        thresholds,
                        recalibration_offsets,
                        pseudo_cfg=pseudo_cfg,
                        recalibration_cfg=recalibration_cfg,
                        base_threshold=base_threshold,
                    ),
                    score_calibrator,
                )
                if score_calibration_enabled and score_calibration_map_thresholds
                else _effective_thresholds(
                    thresholds,
                    recalibration_offsets,
                    pseudo_cfg=pseudo_cfg,
                    recalibration_cfg=recalibration_cfg,
                    base_threshold=base_threshold,
                )
            ),
        }
        save_json(run_dir / "summary.json", summary)
        if teacher_eval_error is None or "CUDA" not in teacher_eval_error:
            maybe_empty_cuda_cache()
        return summary
