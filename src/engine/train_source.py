"""Source supervised pretraining entry."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.data import build_pretrain_loaders, load_ids
from src.engine.ckpt import save_checkpoint
from src.engine.trainer import SourceTrainer
from src.engine.utils import apply_train_mode, build_optimizer, build_scheduler, save_json, save_resolved_config
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-train-ids", required=True)
    parser.add_argument("--source-val-ids", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_train_ids = load_ids(args.source_train_ids)
    source_val_ids = load_ids(args.source_val_ids)
    loaders = build_pretrain_loaders(cfg, source_train_ids, source_val_ids)

    model = build_model(cfg, num_classes=int(cfg.data.num_classes)).to(device)
    apply_train_mode(cfg, model, mode="source_train")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    trainer = SourceTrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=int(cfg.data.num_classes),
    )

    summary = trainer.fit(
        train_loader=loaders["source_train"],
        eval_loaders={"source_val": loaders["source_val"]},
        max_epochs=int(cfg.train.source_epochs),
    )

    eval_metrics = trainer.evaluate({"source_val": loaders["source_val"]})

    out_dir = Path(cfg.run.dir) / "source_pretrain"
    save_checkpoint(
        str(out_dir / "ckpt" / "ckpt_last.pt"),
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=0,
        epoch=int(cfg.train.source_epochs),
        extra={"stage": "source_pretrain"},
    )
    save_json(
        out_dir / "metrics.json",
        {
            "stage": "source_pretrain",
            "train_history": summary.train_history,
            "eval_history": summary.eval_history,
            "eval": eval_metrics,
        },
    )
    save_resolved_config(out_dir / "resolved_config.yaml", cfg)


if __name__ == "__main__":
    main()
