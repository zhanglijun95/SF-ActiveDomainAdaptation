"""SageMaker entry point — bridges SM environment to existing CLI."""

import os
import shutil
import sys
from pathlib import Path

# SageMaker unpacks S3 input channels to /opt/ml/input/data/<channel>/
SM_DATA_DIR = os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data")
SM_SOURCE_CKPT_DIR = os.environ.get("SM_CHANNEL_SOURCE_CKPT", "/opt/ml/input/data/source_ckpt")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

CONFIG_REL = os.environ.get("SM_HP_CONFIG", "configs/daod/round_cityscapes_to_foggy_cityscapes_dino.yaml")

# S3 sync settings (set via hyperparameters)
S3_SYNC_URI = os.environ.get("SM_HP_S3_SYNC_URI", "")
S3_SYNC_INTERVAL = int(os.environ.get("SM_HP_S3_SYNC_INTERVAL", "30"))


def _patch_config(cfg_path: str) -> str:
    """Rewrite data.root and run.root_dir in the YAML config, return new path."""
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})["root"] = SM_DATA_DIR

    # run.root_dir: the DAOD method writes outputs here AND reads source ckpt from here.
    # We point it at /opt/ml/model/runs so outputs get uploaded to S3.
    run_root = os.path.join(SM_MODEL_DIR, "runs")
    cfg.setdefault("run", {})["root_dir"] = run_root

    patched = cfg_path.replace(".yaml", "_sm.yaml")
    with open(patched, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[SM] patched config -> {patched}  data.root={SM_DATA_DIR}  run.root_dir={run_root}")
    return patched


def _link_source_checkpoint(cfg_path: str):
    """Symlink the source checkpoint from the S3 channel into the expected runs/ path."""
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    source = cfg["data"]["source_domain"].lower().replace(" ", "_")
    target = cfg["data"]["target_domain"].lower().replace(" ", "_")
    model_name = cfg["detector"]["model_name"].lower().replace(" ", "_")
    run_root = cfg["run"]["root_dir"]

    expected_dir = Path(run_root) / "daod_source" / f"{source}__to__{target}" / model_name
    expected_dir.mkdir(parents=True, exist_ok=True)

    # Copy/link all files from the source_ckpt channel into the expected directory
    src = Path(SM_SOURCE_CKPT_DIR)
    if src.exists():
        for f in src.iterdir():
            dest = expected_dir / f.name
            if not dest.exists():
                os.symlink(str(f), str(dest))
                print(f"[SM] linked {f} -> {dest}")
    else:
        print(f"[SM] WARNING: source_ckpt channel not found at {SM_SOURCE_CKPT_DIR}")


def main():
    patched_cfg = _patch_config(CONFIG_REL)
    _link_source_checkpoint(patched_cfg)
    run_root = os.path.join(SM_MODEL_DIR, "runs")

    # Start background S3 sync for intermediate results
    if S3_SYNC_URI:
        from s3_sync import start_s3_sync
        start_s3_sync(run_root, S3_SYNC_URI, interval_minutes=S3_SYNC_INTERVAL)

    sys.argv = ["sagemaker_entry", "--config", patched_cfg]

    if CONFIG_REL.startswith("configs/baselines/ddt_daod/"):
        from baselines.ddt_daod.run import main as run_main
    elif CONFIG_REL.startswith("configs/baselines/lpld_daod/"):
        from baselines.lpld_daod.run import main as run_main
    elif CONFIG_REL.startswith("configs/baselines/pets_daod/"):
        from baselines.pets_daod.run import main as run_main
    elif CONFIG_REL.startswith("configs/baselines/lpu_daod/"):
        from baselines.lpu_daod.run import main as run_main
    else:
        from src.methods.daod_run_rounds import main as run_main

    try:
        run_main()
    finally:
        if S3_SYNC_URI:
            try:
                from s3_sync import sync_to_s3_once
                sync_to_s3_once(run_root, S3_SYNC_URI)
            except Exception as exc:
                print(f"[SM] final S3 sync failed: {exc}")


if __name__ == "__main__":
    main()
