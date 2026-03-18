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
from detrex.layers import box_cxcywh_to_xyxy
from detrex.utils import inverse_sigmoid


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


def _run_dino_raw_outputs(adapter: DetrexAdapter, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the local detrex DINO model and keep raw decoder outputs.

    Inputs:
    - `inputs`: detectron2-style model inputs from `_prepare_input`

    Output:
    - one raw output dict per input image containing:
      `pred_logits`, `pred_boxes`, optional `aux_outputs`, and `enc_outputs`

    Assumptions:
    - this helper is intentionally DINO-specific
    - it mirrors the evaluation path inside detrex DINO before postprocessing
    """

    model = adapter.model
    if not all(hasattr(model, attr) for attr in ("preprocess_image", "backbone", "neck", "transformer")):
        raise TypeError("Raw-output extraction is only implemented for the local DINO-style detrex model path.")

    images = model.preprocess_image(inputs)
    batch_size, _, H, W = images.tensor.shape
    img_masks = images.tensor.new_zeros(batch_size, H, W)

    features = model.backbone(images.tensor)
    multi_level_feats = model.neck(features)
    multi_level_masks = []
    multi_level_position_embeddings = []
    for feat in multi_level_feats:
        multi_level_masks.append(
            torch.nn.functional.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
        )
        multi_level_position_embeddings.append(model.position_embedding(multi_level_masks[-1]))

    inter_states, init_reference, inter_references, enc_state, enc_reference = model.transformer(
        multi_level_feats,
        multi_level_masks,
        multi_level_position_embeddings,
        (None, None),
        attn_masks=[None, None],
    )

    outputs_classes = []
    outputs_coords = []
    for lvl in range(inter_states.shape[0]):
        reference = init_reference if lvl == 0 else inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = model.class_embed[lvl](inter_states[lvl])
        tmp = model.bbox_embed[lvl](inter_states[lvl])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
    outputs_class = torch.stack(outputs_classes)
    outputs_coord = torch.stack(outputs_coords)

    final_logits = outputs_class[-1]
    final_boxes = outputs_coord[-1]
    enc_logits = model.transformer.decoder.class_embed[-1](enc_state)

    raw_outputs: list[dict[str, Any]] = []
    aux_outputs = model._set_aux_loss(outputs_class, outputs_coord) if model.aux_loss else None
    for batch_index in range(batch_size):
        raw_output = {
            "pred_logits": final_logits[batch_index].detach().cpu(),
            "pred_boxes": final_boxes[batch_index].detach().cpu(),
            "enc_outputs": {
                "pred_logits": enc_logits[batch_index].detach().cpu(),
                "pred_boxes": enc_reference[batch_index].detach().cpu(),
            },
        }
        if aux_outputs is not None:
            raw_output["aux_outputs"] = [
                {
                    "pred_logits": aux_output["pred_logits"][batch_index].detach().cpu(),
                    "pred_boxes": aux_output["pred_boxes"][batch_index].detach().cpu(),
                }
                for aux_output in aux_outputs
            ]
        raw_outputs.append(raw_output)
    return raw_outputs


def _select_dino_topk(raw_output: dict[str, Any], image_size: tuple[int, int], topk: int) -> list[dict[str, Any]]:
    """Select final DINO detections together with raw logits and aux-layer traces.

    Inputs:
    - `raw_output`: one image's raw DINO output dict
    - `image_size`: `(height, width)` for the resized detector input space
    - `topk`: number of detections to keep, matching DINO inference

    Output:
    - one row per selected detection, including:
      query index, class index, score, selected logit, full class logits,
      final box, and optional aux-layer logits/boxes for the same query
    """

    pred_logits = raw_output["pred_logits"]
    pred_boxes = raw_output["pred_boxes"]
    num_queries, num_classes = pred_logits.shape
    probs = pred_logits.sigmoid()
    topk_values, topk_indices = torch.topk(probs.reshape(-1), k=min(topk, probs.numel()))
    query_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
    class_indices = topk_indices % num_classes

    selected_rows: list[dict[str, Any]] = []
    image_h, image_w = image_size
    scale = torch.tensor([image_w, image_h, image_w, image_h], dtype=pred_boxes.dtype)
    for score, query_idx, class_idx in zip(topk_values, query_indices, class_indices):
        query_idx_int = int(query_idx)
        class_idx_int = int(class_idx)
        box_cxcywh = pred_boxes[query_idx_int]
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh.unsqueeze(0))[0] * scale
        row = {
            "query_index": query_idx_int,
            "class_index": class_idx_int,
            "score": float(score),
            "selected_logit": float(pred_logits[query_idx_int, class_idx_int]),
            "class_logits": pred_logits[query_idx_int].clone(),
            "bbox_cxcywh": box_cxcywh.clone(),
            "bbox_xyxy": box_xyxy.clone(),
        }
        if "aux_outputs" in raw_output:
            row["aux_selected_logits"] = [
                float(aux_output["pred_logits"][query_idx_int, class_idx_int])
                for aux_output in raw_output["aux_outputs"]
            ]
            row["aux_class_logits"] = [
                aux_output["pred_logits"][query_idx_int].clone()
                for aux_output in raw_output["aux_outputs"]
            ]
            row["aux_bbox_xyxy"] = [
                (box_cxcywh_to_xyxy(aux_output["pred_boxes"][query_idx_int].unsqueeze(0))[0] * scale).clone()
                for aux_output in raw_output["aux_outputs"]
            ]
        selected_rows.append(row)
    return selected_rows


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


def run_daod_inference_with_raw(
    adapter: DetrexAdapter,
    batch: dict[str, Any] | Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run inference and keep both postprocessed detections and raw DINO outputs.

    Inputs:
    - one sample dict or an iterable of sample dicts

    Output:
    - list of per-sample dicts containing:
      `prediction`: detectron2 postprocessed output
      `raw_output`: raw DINO decoder output dict
      `selected_detections`: final top-k detections with logits and aux traces
    """

    samples = [batch] if isinstance(batch, dict) else list(batch)
    inputs = [_prepare_input(adapter, sample) for sample in samples]
    with torch.no_grad():
        processed_outputs = adapter.model(inputs)
        raw_outputs = _run_dino_raw_outputs(adapter, inputs)

    return [
        {
            "sample_id": sample["sample_id"],
            "file_name": sample["file_name"],
            "prediction": processed_output,
            "raw_output": raw_output,
            "selected_detections": _select_dino_topk(
                raw_output,
                (int(inp["height"]), int(inp["width"])),
                int(getattr(adapter.model, "select_box_nums_for_evaluation", 300)),
            ),
        }
        for sample, inp, processed_output, raw_output in zip(samples, inputs, processed_outputs, raw_outputs)
    ]
