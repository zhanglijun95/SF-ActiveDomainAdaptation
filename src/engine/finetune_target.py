"""Single-stage target finetune entry."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import torch

from src.config import load_config
from src.engine.ckpt import load_checkpoint, save_checkpoint
from src.engine.trainer import TargetFinetuneTrainer
from src.engine.utils import apply_train_mode, build_optimizer, build_scheduler, save_json, save_resolved_config
from src.methods import RoundState
from src.models import build_model
from src.data import build_adapt_loaders


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())


def _resolve_source_run_dir(cfg) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    dataset = _slug(str(cfg.data.dataset_name))
    source = _slug(str(cfg.data.source_domain))
    return root / "source" / dataset / source


def _resolve_finetune_run_dir(cfg) -> Path:
    root = Path(getattr(cfg.run, "root_dir", "runs"))
    dataset = _slug(str(cfg.data.dataset_name))
    source = _slug(str(cfg.data.source_domain))
    target = _slug(str(cfg.data.target_domain))
    return root / "finetune" / dataset / f"{source}_to_{target}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--round-idx", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = RoundState(round_idx=args.round_idx, queried_ids=set(), pseudo_store={})
    loaders = build_adapt_loaders(cfg, state)

    model = build_model(cfg).to(device)
    ckpt_in = args.init_ckpt
    if ckpt_in is None:
        ckpt_in = str(_resolve_source_run_dir(cfg) / "ckpt" / "ckpt_last.pt")
    load_checkpoint(ckpt_in, model, load_optimizer=False)
    apply_train_mode(cfg, model, mode=str(cfg.train.finetune_mode))

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    trainer = TargetFinetuneTrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=int(cfg.data.num_classes),
    )

    pseudo_loader = loaders.get("target_adapt_pseudo")
    hist = []
    for _ in range(int(cfg.train.epochs)):
        hist.append(trainer.train_one_epoch(loaders["target_adapt_labeled"], pseudo_loader))

    eval_loaders = {k: v for k, v in loaders.items() if k in {"target_test", "source_val"}}
    eval_metrics = trainer.evaluate(eval_loaders)

    out_dir = _resolve_finetune_run_dir(cfg)
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
        epoch=int(cfg.train.epochs),
        extra={"stage": "target_finetune", "round_idx": args.round_idx, "init_ckpt": ckpt_in},
    )
    # Placeholder best checkpoint policy: same as final checkpoint for now.
    save_checkpoint(
        str(ckpt_best),
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=0,
        epoch=int(cfg.train.epochs),
        extra={"stage": "target_finetune", "round_idx": args.round_idx, "best_metric": "target_test_acc_top1"},
    )
    save_json(
        out_dir / "metrics.json",
        {
            "stage": "target_finetune",
            "round_idx": args.round_idx,
            "init_ckpt": ckpt_in,
            "train_history": hist,
            "eval": eval_metrics,
        },
    )
    save_resolved_config(out_dir / "resolved_config.yaml", cfg)
    print(f"Loaded source checkpoint: {ckpt_in}")
    print(f"Saved finetuned checkpoint: {ckpt_last}")


if __name__ == "__main__":
    main()
