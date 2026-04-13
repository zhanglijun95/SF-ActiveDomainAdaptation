"""DINO-specific helpers used only by the isolated FNP baseline."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterable

from detrex.layers import box_cxcywh_to_xyxy
import torch

from src.models import prepare_daod_inputs, run_daod_raw_outputs


def _as_list(batch: dict[str, Any] | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [batch] if isinstance(batch, dict) else list(batch)


def _abs_xyxy_boxes(pred_boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    image_h, image_w = [int(v) for v in image_size]
    scale = torch.tensor([image_w, image_h, image_w, image_h], dtype=pred_boxes.dtype, device=pred_boxes.device)
    return box_cxcywh_to_xyxy(pred_boxes) * scale


@contextmanager
def mc_dropout_enabled(model: torch.nn.Module, *, dropout_rate: float):
    saved_dropout_state: list[tuple[torch.nn.Module, float, bool]] = []
    saved_mha_state: list[tuple[torch.nn.MultiheadAttention, float, bool]] = []
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                saved_dropout_state.append((module, float(module.p), bool(module.training)))
                module.p = float(dropout_rate)
                module.train(True)
            elif isinstance(module, torch.nn.MultiheadAttention):
                saved_mha_state.append((module, float(module.dropout), bool(module.training)))
                module.dropout = float(dropout_rate)
                module.train(True)
        yield
    finally:
        for module, p_value, training in saved_dropout_state:
            module.p = p_value
            module.train(training)
        for module, p_value, training in saved_mha_state:
            module.dropout = p_value
            module.train(training)


def extract_pooled_backbone_features(
    adapter,
    batch: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    with_grad: bool,
) -> torch.Tensor:
    model = adapter.model
    if not all(hasattr(model, attr) for attr in ("preprocess_image", "backbone")):
        raise TypeError("Backbone feature extraction is only implemented for the local DINO-style detrex model path.")

    inputs = prepare_daod_inputs(adapter, batch)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    device_ctx = torch.cuda.device(model_device) if model_device.type == "cuda" else nullcontext()
    grad_ctx = nullcontext() if with_grad else torch.no_grad()

    with device_ctx, grad_ctx:
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        deepest_key = sorted(features.keys())[-1]
        pooled = features[deepest_key].mean(dim=(-2, -1))
        return pooled if with_grad else pooled.detach().cpu()


def mc_dropout_query_statistics(
    adapter,
    sample: dict[str, Any],
    *,
    num_passes: int,
    dropout_rate: float,
    score_floor: float,
    max_queries: int,
) -> dict[str, Any]:
    if num_passes <= 0:
        raise ValueError("num_passes must be positive.")

    raw_outputs = []
    with mc_dropout_enabled(adapter.model, dropout_rate=dropout_rate):
        for _ in range(int(num_passes)):
            raw_outputs.append(run_daod_raw_outputs(adapter, sample, with_grad=False)[0])

    pred_logits = torch.stack([raw_output["pred_logits"] for raw_output in raw_outputs], dim=0)
    pred_boxes = torch.stack([raw_output["pred_boxes"] for raw_output in raw_outputs], dim=0)

    probs = pred_logits.sigmoid()
    softmax_probs = torch.softmax(pred_logits, dim=-1)
    mean_probs = probs.mean(dim=0)
    mean_softmax_probs = softmax_probs.mean(dim=0)
    mean_boxes = pred_boxes.mean(dim=0)
    var_boxes = pred_boxes.var(dim=0, unbiased=False)

    abs_boxes = _abs_xyxy_boxes(mean_boxes, (int(sample["height"]), int(sample["width"])))
    mean_scores, category_ids = mean_probs.max(dim=-1)
    entropy = -(mean_softmax_probs.clamp(min=1e-12) * mean_softmax_probs.clamp(min=1e-12).log()).sum(dim=-1)
    entropy = entropy / max(float(torch.log(torch.tensor(mean_softmax_probs.shape[-1], dtype=torch.float32))), 1.0)
    box_var = var_boxes.mean(dim=-1)

    keep = torch.nonzero(mean_scores >= float(score_floor), as_tuple=False).flatten()
    if keep.numel() > int(max_queries):
        keep_scores = mean_scores[keep]
        topk = torch.topk(keep_scores, k=int(max_queries)).indices
        keep = keep[topk]

    rows = []
    for query_index in keep.tolist():
        rows.append(
            {
                "query_index": int(query_index),
                "category_id": int(category_ids[query_index].item()),
                "score": float(mean_scores[query_index].item()),
                "bbox": [float(v) for v in abs_boxes[query_index].tolist()],
                "entropy": float(entropy[query_index].item()),
                "box_var": float(box_var[query_index].item()),
            }
        )

    loc_score = float(torch.stack([box_var[idx] for idx in keep]).mean().item()) if keep.numel() else 0.0
    ent_score = float(torch.stack([entropy[idx] for idx in keep]).mean().item()) if keep.numel() else 0.0
    return {
        "rows": rows,
        "loc_score": loc_score,
        "ent_score": ent_score,
    }
