"""Evaluate a source-trained model from source run directory."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config
from src.data.utils import build_eval_loaders_for_source
from src.engine.ckpt import load_checkpoint
from src.engine.trainer import SupervisedTrainer
from src.engine.utils import resolve_source_ckpt_path, resolve_source_run_dir, save_json
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    which = str(getattr(getattr(cfg, "eval", object()), "source_ckpt", "best"))
    ckpt_path = resolve_source_ckpt_path(cfg, which=which)
    load_checkpoint(str(ckpt_path), model, load_optimizer=False)
    print(f"Loaded checkpoint: {ckpt_path}")

    eval_loaders = build_eval_loaders_for_source(cfg)
    if not eval_loaders:
        raise RuntimeError("No eval loaders were built for source evaluation.")

    trainer = SupervisedTrainer(
        cfg=cfg,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
    )
    metrics = trainer.evaluate(eval_loaders)

    out_dir = resolve_source_run_dir(cfg)
    out_path = out_dir / "metrics_eval_source.json"
    save_json(
        out_path,
        {
            "ckpt_path": str(ckpt_path),
            "metrics": metrics,
        },
    )
    print(f"Saved eval metrics: {out_path}")


if __name__ == "__main__":
    main()
