"""Eval-only entry."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.data import build_adapt_loaders
from src.engine.ckpt import load_checkpoint
from src.engine.trainer import SupervisedTrainer
from src.engine.utils import save_json
from src.methods import RoundState
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, num_classes=int(cfg.data.num_classes)).to(device)
    load_checkpoint(args.ckpt, model, load_optimizer=False)

    state = RoundState(round_idx=0, queried_ids=set(), pseudo_store={})
    loaders = build_adapt_loaders(cfg, state)
    eval_loaders = {k: v for k, v in loaders.items() if k in {"target_test", "source_val"}}

    trainer = SupervisedTrainer(
        cfg=cfg,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        num_classes=int(cfg.data.num_classes),
    )
    metrics = trainer.evaluate(eval_loaders)

    out_path = Path(cfg.run.dir) / "metrics_eval.json"
    save_json(out_path, metrics)


if __name__ == "__main__":
    main()
