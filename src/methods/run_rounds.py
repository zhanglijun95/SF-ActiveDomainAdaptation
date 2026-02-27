"""Round-based SF-ADA entry."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config
from src.engine.utils import resolve_source_ckpt_path
from src.methods.method import OurMethod, RoundState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method = OurMethod(cfg=cfg, num_classes=int(cfg.data.num_classes), device=device)
    state = RoundState(
        round_idx=0,
        queried_ids=set(),
        pseudo_store={},
        budget_total=0,
        budget_used=0,
    )
    which = str(getattr(getattr(cfg, "model", object()), "source_ckpt", "best"))
    ckpt_init = str(resolve_source_ckpt_path(cfg, which=which))
    method.run_all_rounds(ckpt_init=ckpt_init, state_init=state)


if __name__ == "__main__":
    main()
