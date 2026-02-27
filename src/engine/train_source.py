"""Source supervised pretraining entry."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config
from src.data import build_pretrain_loaders
from src.engine.trainer import SourceTrainer
from src.engine.utils import build_optimizer, build_scheduler, resolve_source_run_dir, save_json, save_resolved_config
from src.models import build_model
from src.models.lora import apply_finetune_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_pretrain_loaders(cfg)

    model = build_model(cfg).to(device)
    apply_finetune_mode(model, mode="source_train")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    trainer = SourceTrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    out_dir = resolve_source_run_dir(cfg)
    ckpt_dir = out_dir / "ckpt"
    ckpt_last = ckpt_dir / "ckpt_last.pt"
    ckpt_best = ckpt_dir / "ckpt_best.pt"

    summary = trainer.fit(
        train_loader=loaders["source_train"],
        eval_loaders={"source_val": loaders["source_val"]},
        max_epochs=int(cfg.train.source_epochs),
        ckpt_last_path=str(ckpt_last),
        ckpt_best_path=str(ckpt_best),
        monitor_loader="source_val",
        monitor_metric="acc_top1",
        ckpt_extra={"stage": "source_pretrain"},
        train_log_path=str(out_dir / "train_log.jsonl"),
        log_every_iters=int(getattr(cfg.train, "log_every_iters", 0)),
    )

    save_json(
        out_dir / "metrics.json",
        {
            "stage": "source_pretrain",
            "eval_history": summary.eval_history,
            "best_epoch": summary.best_epoch,
            "best_score": summary.best_score,
            "monitor_metric": "source_val.acc_top1",
        },
    )
    save_resolved_config(out_dir / "resolved_config.yaml", cfg)
    print(f"Saved source checkpoint: {ckpt_last}")


if __name__ == "__main__":
    main()
