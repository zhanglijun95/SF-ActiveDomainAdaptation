"""Round-based SF-ADA entry."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config
from src.methods.method import OurMethod, RoundState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method = OurMethod(cfg=cfg, num_classes=int(cfg.data.num_classes), device=device)
    state = RoundState(
        round_idx=0,
        queried_ids=set(),
        pseudo_store={},
        budget_total=int(cfg.method.budget_total),
        budget_used=0,
    )
    method.run_all_rounds(ckpt_init=args.init_ckpt, state_init=state)


if __name__ == "__main__":
    main()
