"""Shared utilities for label-guided teacher updates.

The helpers in this module are intentionally model-agnostic.  SFOD methods can
compute one or more gradient-importance maps however they like, then reuse the
same normalization and merge policy before calling their teacher update rule.
"""

from __future__ import annotations

from typing import Literal

import torch


ImportanceMerge = Literal["max", "add", "gt_only", "base_only"]


def collect_grad_importance(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect absolute parameter gradients as an importance map."""

    importance: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] = param.grad.detach().abs().clone()
    return importance


def _global_scale(importance: dict[str, torch.Tensor], *, eps: float) -> torch.Tensor | None:
    flat = [
        value.detach().float().reshape(-1)
        for value in importance.values()
        if value.numel() > 0
    ]
    if not flat:
        return None
    scale = torch.cat(flat).mean()
    if float(scale.detach().cpu()) <= eps:
        return None
    return scale


def normalize_importance_map(
    importance: dict[str, torch.Tensor],
    *,
    enabled: bool = True,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Normalize a map by global mean magnitude.

    AEMA only uses ranks, so scalar normalization does not change a single map.
    It does make two maps comparable before merging.
    """

    if not enabled or not importance:
        return {name: value.detach().clone() for name, value in importance.items()}
    scale = _global_scale(importance, eps=eps)
    if scale is None:
        return {name: value.detach().clone() for name, value in importance.items()}
    return {name: value.detach() / scale.to(value.device) for name, value in importance.items()}


def merge_importance_maps(
    base: dict[str, torch.Tensor],
    guidance: dict[str, torch.Tensor],
    *,
    merge: ImportanceMerge | str = "max",
    guidance_weight: float = 1.0,
    normalize: bool = True,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Merge an SFOD-native importance map with sparse-label guidance.

    ``base`` is the foundation method's own signal, while ``guidance`` is the
    sparse-GT signal.  The default ``max`` policy lets labels elevate
    target-important parameters without suppressing parameters the foundation
    already considers important.
    """

    merge = str(merge).strip().lower()
    if merge not in {"max", "add", "gt_only", "base_only"}:
        raise ValueError(f"Unsupported importance merge mode: {merge!r}")
    if merge == "base_only" or not guidance:
        return normalize_importance_map(base, enabled=normalize, eps=eps)
    if merge == "gt_only" or not base:
        guidance_norm = normalize_importance_map(guidance, enabled=normalize, eps=eps)
        return {name: float(guidance_weight) * value for name, value in guidance_norm.items()}

    base_norm = normalize_importance_map(base, enabled=normalize, eps=eps)
    guidance_norm = normalize_importance_map(guidance, enabled=normalize, eps=eps)
    merged: dict[str, torch.Tensor] = {}
    keys = set(base_norm).union(guidance_norm)
    weight = float(guidance_weight)
    for name in keys:
        base_value = base_norm.get(name)
        guidance_value = guidance_norm.get(name)
        if base_value is None and guidance_value is not None:
            merged[name] = weight * guidance_value
            continue
        if guidance_value is None and base_value is not None:
            merged[name] = base_value
            continue
        if base_value is None or guidance_value is None:
            continue
        guidance_value = guidance_value.to(base_value.device)
        if merge == "add":
            merged[name] = base_value + weight * guidance_value
        else:
            merged[name] = torch.maximum(base_value, weight * guidance_value)
    return merged


def importance_map_stats(importance: dict[str, torch.Tensor]) -> dict[str, float]:
    flat = [
        value.detach().float().reshape(-1)
        for value in importance.values()
        if value.numel() > 0
    ]
    if not flat:
        return {"num_tensors": 0.0, "num_values": 0.0, "mean": 0.0, "max": 0.0}
    all_values = torch.cat(flat)
    return {
        "num_tensors": float(len(flat)),
        "num_values": float(all_values.numel()),
        "mean": float(all_values.mean().detach().cpu()),
        "max": float(all_values.max().detach().cpu()),
    }
