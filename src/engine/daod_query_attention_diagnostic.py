"""Inference-only DINO query-attention diagnostics for sparse target labels.

This module intentionally stays outside the training path.  It records the
decoder deformable cross-attention sampling locations/weights and measures how
well object queries sample inside GT or pseudo boxes.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class DeformableAttentionSnapshot:
    """One decoder cross-attention call captured from DINO."""

    name: str
    layer_index: int
    sampling_locations: torch.Tensor
    attention_weights: torch.Tensor
    valid_ratios: torch.Tensor | None


class DinoDecoderAttentionRecorder(AbstractContextManager["DinoDecoderAttentionRecorder"]):
    """Record decoder deformable cross-attention sampling tensors.

    detrex's DINO uses multi-scale deformable attention, so there is no dense
    attention map to hook.  We recompute the small offset/weight tensors from
    each decoder cross-attention module before the original forward executes.
    This is inference-only and does not change model outputs.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.records: list[DeformableAttentionSnapshot] = []
        self._patched: list[tuple[torch.nn.Module, Any]] = []

    def __enter__(self) -> "DinoDecoderAttentionRecorder":
        for name, module in self.model.named_modules():
            if not _is_decoder_deformable_cross_attention(name, module):
                continue
            original_forward = module.forward

            def wrapped_forward(*args, __name=name, __module=module, __original=original_forward, **kwargs):
                self._record_from_call(__name, __module, args, kwargs)
                return __original(*args, **kwargs)

            module.forward = wrapped_forward
            self._patched.append((module, original_forward))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for module, original_forward in self._patched:
            module.forward = original_forward
        self._patched.clear()

    @property
    def num_patched_modules(self) -> int:
        return len(self._patched)

    def clear(self) -> None:
        self.records.clear()

    def _record_from_call(
        self,
        name: str,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        query = args[0] if len(args) > 0 else kwargs.get("query")
        value = args[2] if len(args) > 2 else kwargs.get("value")
        query_pos = kwargs.get("query_pos")
        reference_points = kwargs.get("reference_points")
        spatial_shapes = kwargs.get("spatial_shapes")
        valid_ratios = kwargs.get("valid_ratios")
        if query is None or reference_points is None or spatial_shapes is None:
            return
        if value is None:
            value = query

        with torch.no_grad():
            q = query + query_pos if query_pos is not None else query
            v = value
            if not bool(getattr(module, "batch_first", False)):
                q = q.permute(1, 0, 2)
                v = v.permute(1, 0, 2)

            batch_size, num_queries, _ = q.shape
            num_heads = int(module.num_heads)
            num_levels = int(module.num_levels)
            num_points = int(module.num_points)

            # Keep this computation exactly aligned with detrex's
            # MultiScaleDeformableAttention forward implementation.
            sampling_offsets = module.sampling_offsets(q).view(
                batch_size,
                num_queries,
                num_heads,
                num_levels,
                num_points,
                2,
            )
            attention_weights = module.attention_weights(q).view(
                batch_size,
                num_queries,
                num_heads,
                num_levels * num_points,
            )
            attention_weights = attention_weights.softmax(-1).view(
                batch_size,
                num_queries,
                num_heads,
                num_levels,
                num_points,
            )

            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = (
                    reference_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                )
            elif reference_points.shape[-1] == 4:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets / num_points * reference_points[:, :, None, :, None, 2:] * 0.5
                )
            else:
                return

            self.records.append(
                DeformableAttentionSnapshot(
                    name=name,
                    layer_index=_layer_index_from_name(name, fallback=len(self.records)),
                    sampling_locations=sampling_locations.detach().cpu(),
                    attention_weights=attention_weights.detach().cpu(),
                    valid_ratios=valid_ratios.detach().cpu() if torch.is_tensor(valid_ratios) else None,
                )
            )


def _is_decoder_deformable_cross_attention(name: str, module: torch.nn.Module) -> bool:
    if module.__class__.__name__ != "MultiScaleDeformableAttention":
        return False
    normalized = name.replace("_checkpoint_wrapped_module.", "")
    return ".decoder.layers." in normalized and ".attentions.1" in normalized


def _layer_index_from_name(name: str, *, fallback: int) -> int:
    parts = name.replace("_checkpoint_wrapped_module.", "").split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return int(fallback)
    return int(fallback)


def resolve_sparse_target_split(
    target_train: list[dict[str, Any]],
    *,
    budget_total: float | int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], dict[str, Any]]:
    """Reproduce the DDT random sparse-label split for diagnostics."""

    total_count = len(target_train)
    if total_count <= 0:
        return [], [], set(), {"budget_k": 0, "target_total": 0, "selected_ids": []}
    if isinstance(budget_total, float) and 0.0 < float(budget_total) <= 1.0:
        budget_k = min(total_count, max(1, int(round(float(budget_total) * float(total_count)))))
    else:
        budget_k = min(total_count, max(0, int(budget_total)))

    rng = np.random.default_rng(int(seed))
    selected_indices = sorted(int(idx) for idx in rng.choice(total_count, size=budget_k, replace=False))
    selected_ids = {str(target_train[idx]["sample_id"]) for idx in selected_indices}
    selected_order = [str(target_train[idx]["sample_id"]) for idx in selected_indices]
    labeled = [sample for sample in target_train if str(sample["sample_id"]) in selected_ids]
    unlabeled = [sample for sample in target_train if str(sample["sample_id"]) not in selected_ids]
    return labeled, unlabeled, selected_ids, {
        "budget_k": int(budget_k),
        "target_total": int(total_count),
        "selected_ids": selected_order,
    }


def xyxy_iou(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def best_query_for_gt(
    query_rows: list[dict[str, Any]],
    annotation: dict[str, Any],
    *,
    require_class_match: bool,
    min_iou: float,
) -> tuple[dict[str, Any] | None, float]:
    gt_box = [float(v) for v in annotation["bbox"]]
    gt_class = int(annotation["category_id"])
    best_row = None
    best_iou = 0.0
    best_score = -1.0
    for row in query_rows:
        if require_class_match and int(row["category_id"]) != gt_class:
            continue
        iou = xyxy_iou(row["bbox"], gt_box)
        score = float(row.get("score", 0.0))
        if iou > best_iou or (math.isclose(iou, best_iou) and score > best_score):
            best_row = row
            best_iou = float(iou)
            best_score = score
    if best_row is None or best_iou < float(min_iou):
        return None, best_iou
    return best_row, best_iou


def best_gt_for_prediction(
    sample: dict[str, Any],
    pred_box: list[float],
    pred_class: int,
) -> dict[str, Any]:
    best_any_iou = 0.0
    best_any_class = None
    best_same_iou = 0.0
    for ann in sample.get("annotations", []):
        gt_class = int(ann["category_id"])
        iou = xyxy_iou(pred_box, ann["bbox"])
        if iou > best_any_iou:
            best_any_iou = float(iou)
            best_any_class = gt_class
        if gt_class == int(pred_class) and iou > best_same_iou:
            best_same_iou = float(iou)
    return {
        "best_gt_iou": float(best_any_iou),
        "best_gt_class": best_any_class,
        "best_same_class_iou": float(best_same_iou),
    }


def summarize_query_attention(
    snapshots: list[DeformableAttentionSnapshot],
    *,
    query_index: int,
    reference_box: list[float],
    image_height: int,
    image_width: int,
    topk_points: int = 8,
) -> dict[str, Any]:
    """Compute per-layer and aggregate attention localization metrics."""

    layer_summaries = []
    for snapshot in sorted(snapshots, key=lambda item: item.layer_index):
        if query_index < 0 or query_index >= snapshot.sampling_locations.shape[1]:
            continue
        layer_summary = _summarize_snapshot_query(
            snapshot,
            query_index=query_index,
            reference_box=reference_box,
            image_height=image_height,
            image_width=image_width,
            topk_points=topk_points,
        )
        layer_summaries.append(layer_summary)

    output: dict[str, Any] = {"num_attention_layers": len(layer_summaries)}
    if not layer_summaries:
        return output

    final = layer_summaries[-1]
    for key, value in final.items():
        if key == "layer_index":
            continue
        output[f"final_{key}"] = value

    metric_keys = [key for key in final if key != "layer_index"]
    for key in metric_keys:
        values = [float(layer[key]) for layer in layer_summaries if layer.get(key) is not None]
        output[f"mean_{key}"] = float(np.mean(values)) if values else None

    output["layers"] = layer_summaries
    return output


def _summarize_snapshot_query(
    snapshot: DeformableAttentionSnapshot,
    *,
    query_index: int,
    reference_box: list[float],
    image_height: int,
    image_width: int,
    topk_points: int,
) -> dict[str, float | int]:
    locations = snapshot.sampling_locations[0, query_index].float().numpy().copy()
    weights = snapshot.attention_weights[0, query_index].float().numpy().copy()
    if snapshot.valid_ratios is not None:
        valid_ratios = snapshot.valid_ratios[0].float().numpy()
        for level_idx in range(locations.shape[1]):
            vx = max(float(valid_ratios[level_idx, 0]), 1e-6)
            vy = max(float(valid_ratios[level_idx, 1]), 1e-6)
            locations[:, level_idx, :, 0] /= vx
            locations[:, level_idx, :, 1] /= vy

    norm_x = locations[..., 0]
    norm_y = locations[..., 1]
    x = norm_x * float(image_width)
    y = norm_y * float(image_height)
    x0, y0, x1, y1 = [float(v) for v in reference_box]
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    outside_image = (norm_x < 0.0) | (norm_x > 1.0) | (norm_y < 0.0) | (norm_y > 1.0)

    flat_weights = weights.reshape(-1).astype(float)
    flat_inside = inside.reshape(-1)
    flat_outside_image = outside_image.reshape(-1)
    total_weight = float(np.sum(flat_weights))
    if total_weight <= 1e-12:
        probs = np.ones_like(flat_weights, dtype=float) / max(len(flat_weights), 1)
        total_weight = 1.0
    else:
        probs = flat_weights / total_weight

    inside_mass = float(np.sum(flat_weights[flat_inside]) / total_weight)
    outside_image_mass = float(np.sum(flat_weights[flat_outside_image]) / total_weight)

    k = min(max(int(topk_points), 1), len(flat_weights))
    top_indices = np.argsort(flat_weights)[-k:]
    topk_inside_rate = float(np.mean(flat_inside[top_indices])) if len(top_indices) else 0.0

    weighted_x = float(np.sum(probs * x.reshape(-1)))
    weighted_y = float(np.sum(probs * y.reshape(-1)))
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    diag = max(float(np.hypot(x1 - x0, y1 - y0)), 1e-6)
    center_distance = float(np.hypot(weighted_x - cx, weighted_y - cy) / diag)

    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    entropy_norm = float(entropy / math.log(max(len(probs), 2)))

    return {
        "layer_index": int(snapshot.layer_index),
        "inside_box_mass": inside_mass,
        "topk_inside_rate": topk_inside_rate,
        "center_distance": center_distance,
        "attention_entropy": entropy_norm,
        "outside_image_mass": outside_image_mass,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate object-level diagnostic rows by object group and class."""

    metric_keys = [
        "final_inside_box_mass",
        "final_topk_inside_rate",
        "final_center_distance",
        "final_attention_entropy",
        "final_outside_image_mass",
        "mean_inside_box_mass",
        "mean_topk_inside_rate",
        "mean_center_distance",
        "mean_attention_entropy",
        "score",
        "match_iou",
        "best_same_class_iou",
    ]
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "groups": {},
        "classes": {},
    }

    def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        agg: dict[str, Any] = {"count": len(items)}
        for key in metric_keys:
            values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_median"] = float(np.median(values))
        return agg

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("group", "unknown"))].append(row)
        by_class[str(row.get("class_name", row.get("category_id", "unknown")))].append(row)
    summary["groups"] = {key: _aggregate(items) for key, items in sorted(by_group.items())}
    summary["classes"] = {key: _aggregate(items) for key, items in sorted(by_class.items())}
    return summary


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_for_csv(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not isinstance(value, (list, dict))}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value
