"""CLI entrypoint for the isolated FNP DAOD baseline."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config

from .method import FNPDAODMethod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FNP-DAOD] config={args.config} device={device}")
    method = FNPDAODMethod(cfg=cfg, device=device)
    final_state = method.run_all_rounds(config_path=args.config)
    print(
        f"[FNP-DAOD] rounds={final_state.round_idx} "
        f"budget={final_state.budget_used}/{final_state.budget_total} "
        f"teacher={final_state.teacher_checkpoint} "
        f"student={final_state.student_checkpoint}"
    )


if __name__ == "__main__":
    main()
