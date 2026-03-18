"""Thin detrex adapter for DAOD detector construction, loading, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import detectron2.data.transforms as T
import numpy as np
from PIL import Image
import torch
from detectron2.config import LazyConfig, instantiate


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
    num_classes: int
    min_size_test: int
    max_size_test: int


@dataclass
class DetrexLoadReport:
    mode: str
    loaded_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass
class DetrexAdapter:
    config: DetrexAdapterConfig
    lazy_cfg: Any
    model: torch.nn.Module
    aug: T.ResizeShortestEdge
    load_report: DetrexLoadReport


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
    if hasattr(detector_cfg, "init_checkpoint"):
        checkpoint_name = str(detector_cfg.init_checkpoint)
    else:
        checkpoint_name = str(getattr(detector_cfg, "checkpoint_name", f"{model_name}.pth"))
    checkpoint_path = _CKPT_ROOT / checkpoint_name
    if checkpoint_path.exists():
        return checkpoint_path
    raise FileNotFoundError(
        f"Missing detrex checkpoint: {checkpoint_path}\n"
        f"Download the matching weights from {_MODEL_ZOO_URL} and place the file there."
    )


def _get_num_classes(cfg: Any) -> int:
    return int(cfg.data.num_classes)


def _apply_num_classes(lazy_cfg: Any, num_classes: int) -> None:
    if hasattr(lazy_cfg.model, "num_classes"):
        lazy_cfg.model.num_classes = num_classes
    if hasattr(lazy_cfg.model, "criterion") and hasattr(lazy_cfg.model.criterion, "num_classes"):
        lazy_cfg.model.criterion.num_classes = num_classes
    if hasattr(lazy_cfg.model, "transformer") and hasattr(lazy_cfg.model.transformer, "num_classes"):
        lazy_cfg.model.transformer.num_classes = num_classes


def _load_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        new_key = key[7:] if key.startswith("module.") else key
        state_dict[new_key] = value
    return state_dict


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str) -> DetrexLoadReport:
    checkpoint_state = _load_checkpoint_state_dict(checkpoint_path)
    try:
        incompatible = model.load_state_dict(checkpoint_state, strict=False)
        return DetrexLoadReport(
            mode="full",
            loaded_keys=len(checkpoint_state),
            missing_keys=list(incompatible.missing_keys),
            unexpected_keys=list(incompatible.unexpected_keys),
        )
    except RuntimeError:
        model_state = model.state_dict()
        filtered_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        incompatible = model.load_state_dict(filtered_state, strict=False)
        missing_keys = set(incompatible.missing_keys)
        missing_keys.update(set(model_state) - set(filtered_state))
        unexpected_keys = set(incompatible.unexpected_keys)
        unexpected_keys.update(set(checkpoint_state) - set(filtered_state))
        return DetrexLoadReport(
            mode="partial",
            loaded_keys=len(filtered_state),
            missing_keys=sorted(missing_keys),
            unexpected_keys=sorted(unexpected_keys),
        )


def load_daod_model_config(cfg: Any) -> DetrexAdapterConfig:
    detector_cfg = _get_detector_cfg(cfg)
    model_name = str(detector_cfg.model_name)
    return DetrexAdapterConfig(
        model_name=model_name,
        config_path=str(_find_config_path(model_name)),
        init_checkpoint=str(_get_checkpoint_path(detector_cfg, model_name)),
        num_classes=_get_num_classes(cfg),
        min_size_test=int(getattr(detector_cfg, "min_size_test", 800)),
        max_size_test=int(getattr(detector_cfg, "max_size_test", 1333)),
    )


def build_daod_model(cfg: Any, *, load_weights: bool = True) -> DetrexAdapter:
    sys.path.insert(0, str(_DETREX_ROOT))

    adapter_cfg = load_daod_model_config(cfg)
    lazy_cfg = LazyConfig.load(adapter_cfg.config_path)
    _apply_num_classes(lazy_cfg, adapter_cfg.num_classes)
    lazy_cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    lazy_cfg.model.device = lazy_cfg.train.device

    model = instantiate(lazy_cfg.model)
    model.to(lazy_cfg.train.device)
    model.eval()

    if load_weights:
        load_report = _load_model_weights(model, adapter_cfg.init_checkpoint)
    else:
        load_report = DetrexLoadReport(
            mode="none",
            loaded_keys=0,
            missing_keys=[],
            unexpected_keys=[],
        )

    aug = T.ResizeShortestEdge(
        [adapter_cfg.min_size_test, adapter_cfg.min_size_test],
        adapter_cfg.max_size_test,
    )
    return DetrexAdapter(
        config=adapter_cfg,
        lazy_cfg=lazy_cfg,
        model=model,
        aug=aug,
        load_report=load_report,
    )


def _load_image(file_name: str) -> np.ndarray:
    return np.asarray(Image.open(file_name).convert("RGB"))


def _sample_image(sample: dict[str, Any]) -> np.ndarray:
    """Resolve an input image from an in-memory override or from file_name."""
    if "image" not in sample:
        return _load_image(sample["file_name"])

    image = sample["image"]
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        return image
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.transpose(array, (1, 2, 0))
        return array
    raise TypeError(f"Unsupported in-memory image type: {type(image)}")


def _prepare_input(adapter: DetrexAdapter, sample: dict[str, Any]) -> dict[str, Any]:
    image = _sample_image(sample)
    image = adapter.aug.get_transform(image).apply_image(image)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    height = int(sample.get("height", image.shape[1]))
    width = int(sample.get("width", image.shape[2]))
    return {
        "image": torch.as_tensor(image, dtype=torch.float32),
        "height": height,
        "width": width,
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
