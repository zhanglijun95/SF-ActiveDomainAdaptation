"""Thin detrex adapter for DAOD detector construction and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import numpy as np
from PIL import Image
import torch


_MODEL_ZOO_URL = "https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DETREX_ROOT = _REPO_ROOT / "external" / "detrex"
_PROJECTS_ROOT = _DETREX_ROOT / "projects"
_CKPT_ROOT = _DETREX_ROOT / "ckpts"


@dataclass
class DetrexAdapterConfig:
    model_name: str
    config_path: str
    init_checkpoint: str
    min_size_test: int
    max_size_test: int


@dataclass
class DetrexAdapter:
    config: DetrexAdapterConfig
    lazy_cfg: Any
    model: torch.nn.Module
    aug: T.ResizeShortestEdge


def _get_detector_cfg(cfg: Any) -> Any:
    if hasattr(cfg, "detector"):
        return cfg.detector
    if hasattr(cfg, "model") and hasattr(cfg.model, "detector"):
        return cfg.model.detector
    if hasattr(cfg, "daod") and hasattr(cfg.daod, "detector"):
        return cfg.daod.detector
    raise AttributeError("Expected detector config under cfg.detector, cfg.model.detector, or cfg.daod.detector")


def _find_config_path(model_name: str) -> Path:
    direct = _PROJECTS_ROOT.glob(f"*/configs/{model_name}.py")
    nested = _PROJECTS_ROOT.glob(f"*/configs/*/{model_name}.py")
    matches = sorted([*direct, *nested])
    if not matches:
        raise FileNotFoundError(f"Could not find detrex config for model_name={model_name}")
    return matches[0]


def _get_checkpoint_path(detector_cfg: Any, model_name: str) -> Path:
    checkpoint_name = str(getattr(detector_cfg, "checkpoint_name", f"{model_name}.pth"))
    checkpoint_path = _CKPT_ROOT / checkpoint_name
    if checkpoint_path.exists():
        return checkpoint_path
    raise FileNotFoundError(
        f"Missing detrex checkpoint: {checkpoint_path}\n"
        f"Download the matching weights from {_MODEL_ZOO_URL} and place the file there."
    )


def load_daod_model_config(cfg: Any) -> DetrexAdapterConfig:
    detector_cfg = _get_detector_cfg(cfg)
    model_name = str(detector_cfg.model_name)
    config_path = _find_config_path(model_name)
    checkpoint_path = _get_checkpoint_path(detector_cfg, model_name)
    return DetrexAdapterConfig(
        model_name=model_name,
        config_path=str(config_path),
        init_checkpoint=str(checkpoint_path),
        min_size_test=int(getattr(detector_cfg, "min_size_test", 800)),
        max_size_test=int(getattr(detector_cfg, "max_size_test", 1333)),
    )


def build_daod_model(cfg: Any, *, load_weights: bool = True) -> DetrexAdapter:
    sys.path.insert(0, str(_DETREX_ROOT))

    adapter_cfg = load_daod_model_config(cfg)
    lazy_cfg = LazyConfig.load(adapter_cfg.config_path)
    lazy_cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    lazy_cfg.model.device = lazy_cfg.train.device

    model = instantiate(lazy_cfg.model)
    model.to(lazy_cfg.train.device)
    model.eval()

    if load_weights:
        DetectionCheckpointer(model).load(adapter_cfg.init_checkpoint)

    aug = T.ResizeShortestEdge(
        [adapter_cfg.min_size_test, adapter_cfg.min_size_test],
        adapter_cfg.max_size_test,
    )
    return DetrexAdapter(config=adapter_cfg, lazy_cfg=lazy_cfg, model=model, aug=aug)


def _load_image(file_name: str) -> np.ndarray:
    return np.asarray(Image.open(file_name).convert("RGB"))


def _prepare_input(adapter: DetrexAdapter, sample: dict[str, Any]) -> dict[str, Any]:
    image = _load_image(sample["file_name"])
    height, width = image.shape[:2]
    image = adapter.aug.get_transform(image).apply_image(image)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return {
        "image": torch.as_tensor(image, dtype=torch.float32),
        "height": int(sample["height"]),
        "width": int(sample["width"]),
        "file_name": sample["file_name"],
        "sample_id": sample["sample_id"],
        "image_id": sample.get("image_id", sample["sample_id"]),
    }


def run_daod_inference(
    adapter: DetrexAdapter,
    batch: dict[str, Any] | Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples = [batch] if isinstance(batch, dict) else list(batch)
    inputs = [_prepare_input(adapter, sample) for sample in samples]
    with torch.no_grad():
        outputs = adapter.model(inputs)
    return [
        {
            "sample_id": sample["sample_id"],
            "file_name": sample["file_name"],
            "prediction": output,
        }
        for sample, output in zip(samples, outputs)
    ]
