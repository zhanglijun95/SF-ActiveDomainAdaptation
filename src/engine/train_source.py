"""Source supervised pretraining entry."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import torch

from src.config import load_config
from src.data import build_pretrain_loaders
from src.engine.ckpt import save_checkpoint
from src.engine.trainer import SourceTrainer
from src.engine.utils import apply_train_mode, build_optimizer, build_scheduler, save_json, save_resolved_config
from src.models import build_model


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())


def _resolve_source_run_dir(cfg) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    dataset = _slug(str(cfg.data.dataset_name))
    source = _slug(str(cfg.data.source_domain))
    return root / "source" / dataset / source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_pretrain_loaders(cfg)

    model = build_model(cfg).to(device)
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

    out_dir = _resolve_source_run_dir(cfg)
    ckpt_dir = out_dir / "ckpt"
    ckpt_last = ckpt_dir / "ckpt_last.pt"
    ckpt_best = ckpt_dir / "ckpt_best.pt"
    save_checkpoint(
        str(ckpt_last),
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=0,
        epoch=int(cfg.train.source_epochs),
        extra={"stage": "source_pretrain"},
    )
    # Placeholder best checkpoint policy: same as final checkpoint for now.
    save_checkpoint(
        str(ckpt_best),
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=0,
        epoch=int(cfg.train.source_epochs),
        extra={"stage": "source_pretrain", "best_metric": "source_val_acc_top1"},
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
    print(f"Saved source checkpoint: {ckpt_last}")


if __name__ == "__main__":
    main()
