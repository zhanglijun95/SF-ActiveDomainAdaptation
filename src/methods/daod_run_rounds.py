"""Round-based DAOD entry."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config
from src.methods.daod_method import DAODRoundMethod, build_default_daod_round_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DAOD] config={args.config} device={device}")

    state = build_default_daod_round_state(cfg)
    print(f"[DAOD] init_student={state.student_checkpoint}")
    method = DAODRoundMethod(cfg=cfg, device=device)
    final_state = method.run_all_rounds(source_ckpt=state.student_checkpoint, state_init=state)
    print(
        f"[DAOD] final_budget_used={final_state.budget_used}/{final_state.budget_total} "
        f"teacher={final_state.teacher_checkpoint} student={final_state.student_checkpoint}"
    )


if __name__ == "__main__":
    main()
