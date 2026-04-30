"""Round-based DAOD entry."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch

from src.config import load_config
from src.engine.utils import seed_everything
from src.methods.daod_method import DAODRoundMethod, build_default_daod_round_state
from src.methods.daod_stepwise_injection_method import (
    DAODStepwiseInjectionMethod,
    build_default_daod_stepwise_injection_state,
)


def build_daod_method_from_cfg(cfg, device: torch.device):
    scheme = str(getattr(getattr(cfg, "method", object()), "training_scheme", "episodic")).strip().lower()
    if scheme == "stepwise_injection":
        return DAODStepwiseInjectionMethod(cfg=cfg, device=device)
    return DAODRoundMethod(cfg=cfg, device=device)


def build_daod_state_from_cfg(cfg):
    scheme = str(getattr(getattr(cfg, "method", object()), "training_scheme", "episodic")).strip().lower()
    if scheme == "stepwise_injection":
        return build_default_daod_stepwise_injection_state(cfg)
    return build_default_daod_round_state(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = getattr(getattr(cfg, "method", object()), "train", object())
    stability_cfg = getattr(train_cfg, "stability", object())
    if bool(getattr(stability_cfg, "seed_everything", True)):
        seed_everything(
            int(getattr(cfg, "seed", 42)),
            deterministic=bool(getattr(stability_cfg, "deterministic", False)),
            warn_only=bool(getattr(stability_cfg, "deterministic_warn_only", True)),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DAOD] config={args.config} device={device}")

    state = build_daod_state_from_cfg(cfg)
    print(f"[DAOD] init_student={state.student_checkpoint}")
    method = build_daod_method_from_cfg(cfg, device=device)
    method.run_dir.mkdir(parents=True, exist_ok=True)
    config_src = Path(args.config).resolve()
    if config_src.is_file():
        shutil.copy2(config_src, method.run_dir / "config.yaml")
    final_state = method.run_all_rounds(source_ckpt=state.student_checkpoint, state_init=state)
    print(
        f"[DAOD] final_budget_used={final_state.budget_used}/{final_state.budget_total} "
        f"teacher={final_state.teacher_checkpoint} student={final_state.student_checkpoint}"
    )


if __name__ == "__main__":
    main()
