"""CLI entrypoint for the isolated LPU DAOD baseline."""

from __future__ import annotations

import argparse

import torch

from src.config import load_config

from .method import LPUDAODMethod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LPU-DAOD] config={args.config} device={device}")
    method = LPUDAODMethod(cfg=cfg, device=device)
    summary = method.run(config_path=args.config)
    target_ap = summary.get("final_target_val_metrics", {}).get("bbox", {}).get("AP", None)
    final_checkpoint = summary.get("final_checkpoint", summary.get("student_checkpoint"))
    print(
        f"[LPU-DAOD] epochs={summary['epochs']} step={summary['global_step']} "
        f"final_model={summary.get('final_model')} checkpoint={final_checkpoint} target_AP={target_ap}"
    )


if __name__ == "__main__":
    main()

