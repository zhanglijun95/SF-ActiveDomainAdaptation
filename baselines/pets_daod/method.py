"""Entrypoint orchestration for the isolated PETS DAOD baseline."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import torch

from baselines.sfod_common.trainer import SFODBaselineTrainer
from baselines.sfod_common.utils import save_resolved_config, set_seed

from .config import resolve_pets_daod_run_dir, resolve_pets_daod_source_ckpt_path


class PETSDAODMethod:
    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.run_dir = resolve_pets_daod_run_dir(cfg)

    def run(self, *, config_path: str | Path | None = None) -> dict[str, Any]:
        seed = int(getattr(self.cfg, "seed", 42))
        set_seed(seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(self.run_dir / "resolved_config.yaml", self.cfg)
        if config_path is not None:
            config_src = Path(config_path).resolve()
            if config_src.is_file():
                shutil.copy2(config_src, self.run_dir / "config.yaml")

        source_ckpt = resolve_pets_daod_source_ckpt_path(
            self.cfg,
            which=str(getattr(getattr(self.cfg, "detector", object()), "source_ckpt", "final")),
        )
        if not source_ckpt.exists():
            raise FileNotFoundError(
                f"Missing DAOD source checkpoint for PETS baseline: {source_ckpt}\n"
                "Run source DINO training first, or set detector.source_ckpt to an existing checkpoint."
            )

        trainer = SFODBaselineTrainer(cfg=self.cfg, device=self.device, algorithm="pets", log_prefix="PETS-DAOD")
        return trainer.fit(run_dir=self.run_dir, source_checkpoint=str(source_ckpt))

