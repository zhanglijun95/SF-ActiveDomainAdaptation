"""Thin DAOD training entry for source-only and oracle target finetuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import SimpleTrainer, create_ddp_model, hooks, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils import comm
import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.data.daod.detectron2 import (
    build_daod_detection_test_loader,
    build_daod_detection_train_loader,
    export_daod_coco_json,
    materialize_daod_dicts,
    register_daod_eval_dataset,
)
from src.engine.utils import (
    resolve_daod_oracle_run_dir,
    resolve_optional_daod_checkpoint_path,
    resolve_daod_source_run_dir,
    save_resolved_config,
)
from src.models.detrex_adapter import build_daod_model


def _output_dir(cfg: Any, *, oracle: bool = False) -> Path:
    if hasattr(cfg.train, "output_dir"):
        return Path(cfg.train.output_dir)
    if oracle:
        return resolve_daod_oracle_run_dir(cfg)
    return resolve_daod_source_run_dir(cfg)


def _optimizer(cfg: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(getattr(cfg.train, "lr", 1e-4)),
        weight_decay=float(getattr(cfg.train, "weight_decay", 1e-4)),
    )


def _scheduler(cfg: Any, optimizer: torch.optim.Optimizer, max_iter: int):
    scheduler_name = str(getattr(cfg.train, "lr_scheduler", "cosine")).strip().lower()
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)


def _max_iter(cfg: Any, train_size: int) -> int:
    if hasattr(cfg.train, "max_iter"):
        return int(cfg.train.max_iter)
    epochs = int(getattr(cfg.train, "epochs", 1))
    batch_size = int(cfg.train.batch_size)
    steps_per_epoch = max(1, (train_size + batch_size - 1) // batch_size)
    return steps_per_epoch * epochs


def _limit_samples(dataset_dicts: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return dataset_dicts
    return dataset_dicts[:limit]


def _evaluate_split(
    cfg: Any,
    model: torch.nn.Module,
    split_name: str,
    dataset_dicts: list[dict[str, Any]],
) -> dict[str, Any]:
    # Each eval writes a fresh temporary COCO json, so the dataset name also has
    # to be fresh; detectron2 metadata does not allow mutating `json_file`.
    eval_name = f"daod_{split_name}_{time.time_ns()}"
    with tempfile.TemporaryDirectory(prefix=f"{split_name}_eval_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        json_path = export_daod_coco_json(cfg, dataset_dicts, tmp_dir / f"{split_name}.json")
        register_daod_eval_dataset(eval_name, cfg, dataset_dicts, json_path)
        loader = build_daod_detection_test_loader(cfg, dataset_dicts)
        evaluator = COCOEvaluator(eval_name, output_dir=str(tmp_dir / "eval"))
        return inference_on_dataset(model, loader, evaluator)


def _split_eval_fn(cfg: Any, model: torch.nn.Module, split_name: str, dataset_dicts: list[dict[str, Any]]):
    def _run_eval():
        return _evaluate_split(cfg, model, split_name, dataset_dicts)

    return _run_eval


def _log_eval_metrics(output_dir: Path, step: int, split_name: str, metrics: dict[str, Any]) -> None:
    if not comm.is_main_process():
        return
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


def _resolve_eval_checkpoint(cfg: Any, output_dir: Path, checkpoint_name: str | None) -> Path:
    if checkpoint_name is not None:
        checkpoint_path = Path(checkpoint_name)
        if not checkpoint_path.is_absolute():
            checkpoint_path = output_dir / checkpoint_path
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    override_path = resolve_optional_daod_checkpoint_path(
        getattr(getattr(cfg, "detector", object()), "source_ckpt_path", None),
        which=str(getattr(getattr(cfg, "detector", object()), "source_ckpt", "best")),
    )
    if override_path is not None:
        return override_path

    for candidate in ("model_best.pth", "model_final.pth"):
        checkpoint_path = output_dir / candidate
        if checkpoint_path.exists():
            return checkpoint_path
    raise FileNotFoundError(f"No evaluation checkpoint found under {output_dir}")


def _resolve_source_init_checkpoint(cfg: Any, checkpoint_name: str | None) -> Path:
    source_output_dir = _output_dir(cfg, oracle=False)
    if checkpoint_name is not None:
        checkpoint_path = Path(checkpoint_name)
        if not checkpoint_path.is_absolute():
            checkpoint_path = source_output_dir / checkpoint_path
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"Oracle init checkpoint not found: {checkpoint_path}")
    override_path = resolve_optional_daod_checkpoint_path(
        getattr(getattr(cfg, "detector", object()), "source_ckpt_path", None),
        which=str(getattr(getattr(cfg, "detector", object()), "source_ckpt", "best")),
    )
    if override_path is not None:
        return override_path
    return _resolve_eval_checkpoint(cfg, source_output_dir, checkpoint_name=None)


def _build_eval_splits(cfg: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_val_dicts = materialize_daod_dicts(cfg, "source_val")
    target_val_dicts = materialize_daod_dicts(cfg, "target_val")
    source_val_dicts = _limit_samples(
        source_val_dicts,
        getattr(getattr(cfg, "eval", object()), "source_val_limit", None),
    )
    target_val_dicts = _limit_samples(
        target_val_dicts,
        getattr(getattr(cfg, "eval", object()), "target_val_limit", None),
    )
    return source_val_dicts, target_val_dicts


def _train_daod(
    cfg: Any,
    *,
    train_split: str,
    output_dir: Path,
    init_checkpoint: str | None,
    best_metric_split: str,
    resume: bool = False,
) -> dict[str, Any]:
    adapter = build_daod_model(cfg, load_weights=init_checkpoint is None)
    model = adapter.model
    if init_checkpoint is not None:
        DetectionCheckpointer(model).load(str(init_checkpoint))

    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_ddp_model(model, broadcast_buffers=False)

    train_dicts = materialize_daod_dicts(cfg, train_split)
    source_val_dicts, target_val_dicts = _build_eval_splits(cfg)
    best_eval_dicts = source_val_dicts if best_metric_split == "source_val" else target_val_dicts

    train_loader = build_daod_detection_train_loader(cfg, train_dicts)
    optimizer = _optimizer(cfg, model)
    trainer = SimpleTrainer(model, train_loader, optimizer)
    checkpointer = DetectionCheckpointer(model, str(output_dir), trainer=trainer)

    max_iter = _max_iter(cfg, train_size=len(train_dicts))
    scheduler = _scheduler(cfg, optimizer, max_iter)
    writers = []
    if comm.is_main_process():
        writers = [
            CommonMetricPrinter(max_iter),
            JSONWriter(str(output_dir / "metrics.json")),
            TensorboardXWriter(str(output_dir)),
        ]

    trainer_hooks: list[Any] = [hooks.IterationTimer()]
    trainer_hooks.append(hooks.LRScheduler(optimizer=optimizer, scheduler=scheduler))
    checkpoint_period = int(getattr(cfg.train, "checkpoint_period", 0))
    if checkpoint_period > 0 and comm.is_main_process():
        trainer_hooks.append(
            hooks.PeriodicCheckpointer(
                checkpointer,
                period=checkpoint_period,
                max_to_keep=int(getattr(cfg.train, "max_to_keep", 10)),
            )
        )

    eval_period = int(
        getattr(
            cfg.train,
            "eval_period",
            max(1, (len(train_dicts) + int(cfg.train.batch_size) - 1) // int(cfg.train.batch_size)),
        )
    )
    trainer_hooks.append(hooks.EvalHook(eval_period, _split_eval_fn(cfg, model, best_metric_split, best_eval_dicts)))
    if comm.is_main_process():
        trainer_hooks.append(
            hooks.BestCheckpointer(
                eval_period,
                checkpointer,
                val_metric="bbox/AP",
                mode="max",
                file_prefix="model_best",
            )
        )
    if writers:
        trainer_hooks.append(hooks.PeriodicWriter(writers, period=int(getattr(cfg.train, "log_period", 20))))
    trainer.register_hooks(trainer_hooks)

    start_iter = 0
    if resume and checkpointer.has_checkpoint():
        resume_checkpoint = checkpointer.get_checkpoint_file()
        checkpointer.resume_or_load(resume_checkpoint, resume=True)
        start_iter = trainer.iter + 1
        if comm.is_main_process():
            print(f"[Resume] Resuming DAOD training from {resume_checkpoint} at iteration {start_iter}.")

    model.train()
    trainer.train(start_iter, max_iter)
    checkpointer.save("model_final")

    model.eval()
    source_val_metrics = _evaluate_split(cfg, model, "source_val", source_val_dicts)
    target_val_metrics = _evaluate_split(cfg, model, "target_val", target_val_dicts)
    if comm.is_main_process():
        _log_eval_metrics(output_dir, max_iter, "source_val", source_val_metrics)
        _log_eval_metrics(output_dir, max_iter, "target_val", target_val_metrics)
        (output_dir / "source_val_metrics.json").write_text(json.dumps(source_val_metrics), encoding="utf-8")
        (output_dir / "target_val_metrics.json").write_text(json.dumps(target_val_metrics), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "train_split": train_split,
        "best_metric_split": best_metric_split,
        "start_iter": start_iter,
        "max_iter": max_iter,
        "source_val_metrics": source_val_metrics,
        "target_val_metrics": target_val_metrics,
        "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
        "resume": resume,
    }


def train_daod_source_only(cfg: Any, *, resume: bool = False) -> dict[str, Any]:
    return _train_daod(
        cfg,
        train_split="source_train",
        output_dir=_output_dir(cfg, oracle=False),
        init_checkpoint=None,
        best_metric_split="source_val",
        resume=resume,
    )


def train_daod_oracle(cfg: Any, checkpoint_name: str | None = None, *, resume: bool = False) -> dict[str, Any]:
    init_checkpoint = _resolve_source_init_checkpoint(cfg, checkpoint_name)
    return _train_daod(
        cfg,
        train_split="target_train",
        output_dir=_output_dir(cfg, oracle=True),
        init_checkpoint=str(init_checkpoint),
        best_metric_split="target_val",
        resume=resume,
    )


def eval_daod_source_model(cfg: Any, checkpoint_name: str | None = None, *, oracle: bool = False) -> dict[str, Any]:
    output_dir = _output_dir(cfg, oracle=oracle)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = build_daod_model(cfg, load_weights=False)
    model = adapter.model
    checkpoint_path = _resolve_eval_checkpoint(cfg, output_dir, checkpoint_name)
    DetectionCheckpointer(model).load(str(checkpoint_path))
    if comm.get_world_size() > 1:
        model = create_ddp_model(model, broadcast_buffers=False)
    model.eval()

    source_val_dicts, target_val_dicts = _build_eval_splits(cfg)

    source_val_metrics = _evaluate_split(cfg, model, "source_val", source_val_dicts)
    target_val_metrics = _evaluate_split(cfg, model, "target_val", target_val_dicts)
    if comm.is_main_process():
        _log_eval_metrics(output_dir, 0, "source_val", source_val_metrics)
        _log_eval_metrics(output_dir, 0, "target_val", target_val_metrics)
        (output_dir / "source_val_metrics.json").write_text(json.dumps(source_val_metrics), encoding="utf-8")
        (output_dir / "target_val_metrics.json").write_text(json.dumps(target_val_metrics), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "source_val_metrics": source_val_metrics,
        "target_val_metrics": target_val_metrics,
    }


def _main_worker(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.eval_only:
        summary = eval_daod_source_model(cfg, checkpoint_name=args.checkpoint, oracle=args.oracle)
    elif args.oracle:
        summary = train_daod_oracle(cfg, checkpoint_name=args.init_checkpoint, resume=args.resume)
    else:
        summary = train_daod_source_only(cfg, resume=args.resume)
    if not comm.is_main_process():
        return

    output_dir = Path(summary["output_dir"])
    save_resolved_config(output_dir / "resolved_config.yaml", cfg)
    print(f"Saved DAOD source outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count() if torch.cuda.is_available() else 1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    args = parser.parse_args()
    launch(
        _main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
