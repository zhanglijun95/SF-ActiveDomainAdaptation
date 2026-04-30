"""Gradient surgery utilities for sparse human-label DAOD.

The trusted target-label gradient is treated as the anchor.  Auxiliary gradients
from pseudo-label objectives are only changed when they point against that anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


GradList = list[torch.Tensor | None]


@dataclass
class PCGradStats:
    cosine_before: float | None
    cosine_after: float | None
    projected: bool
    weight: float | None = None


def _dot(grads_a: GradList, grads_b: GradList) -> torch.Tensor | None:
    total: torch.Tensor | None = None
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        value = torch.sum(grad_a * grad_b)
        total = value if total is None else total + value
    return total


def _squared_norm(grads: GradList) -> torch.Tensor | None:
    total: torch.Tensor | None = None
    for grad in grads:
        if grad is None:
            continue
        value = torch.sum(grad * grad)
        total = value if total is None else total + value
    return total


def _cosine(grads_a: GradList, grads_b: GradList, *, eps: float) -> float | None:
    dot = _dot(grads_a, grads_b)
    norm_a = _squared_norm(grads_a)
    norm_b = _squared_norm(grads_b)
    if dot is None or norm_a is None or norm_b is None:
        return None
    if float(norm_a.detach().cpu()) <= float(eps) or float(norm_b.detach().cpu()) <= float(eps):
        return None
    denom = torch.sqrt(torch.clamp(norm_a * norm_b, min=float(eps)))
    return float((dot / denom).detach().cpu())


def target_anchored_pcgrad(
    *,
    anchor_grads: GradList,
    aux_grads: GradList,
    eps: float = 1e-12,
) -> tuple[GradList, PCGradStats]:
    """Project an auxiliary gradient away from a trusted anchor on conflict.

    If ``dot(aux, anchor) < 0``, remove the component of ``aux`` that lies along
    ``anchor``.  The anchor gradient is never modified.
    """

    cosine_before = _cosine(aux_grads, anchor_grads, eps=eps)
    dot = _dot(aux_grads, anchor_grads)
    anchor_norm = _squared_norm(anchor_grads)
    if dot is None or anchor_norm is None:
        return list(aux_grads), PCGradStats(
            cosine_before=cosine_before,
            cosine_after=cosine_before,
            projected=False,
            weight=1.0,
        )

    projected = bool(float(dot.detach().cpu()) < 0.0 and float(anchor_norm.detach().cpu()) > float(eps))
    if not projected:
        return list(aux_grads), PCGradStats(
            cosine_before=cosine_before,
            cosine_after=cosine_before,
            projected=False,
            weight=1.0,
        )

    scale = dot / torch.clamp(anchor_norm, min=float(eps))
    projected_grads: GradList = []
    for aux_grad, anchor_grad in zip(aux_grads, anchor_grads):
        if aux_grad is None:
            projected_grads.append(None)
        elif anchor_grad is None:
            projected_grads.append(aux_grad)
        else:
            projected_grads.append(aux_grad - scale * anchor_grad)

    return projected_grads, PCGradStats(
        cosine_before=cosine_before,
        cosine_after=_cosine(projected_grads, anchor_grads, eps=eps),
        projected=True,
        weight=1.0,
    )


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference)


def _first_tensor(*grad_lists: GradList) -> torch.Tensor | None:
    for grads in grad_lists:
        for grad in grads:
            if grad is not None:
                return grad
    return None


def _weighted_sum_grad_lists(
    grad_lists: list[GradList],
    weights: list[float | torch.Tensor],
) -> GradList:
    if len(grad_lists) != len(weights):
        raise ValueError("Gradient list and weight counts must match.")
    if not grad_lists:
        return []

    reference = _first_tensor(*grad_lists)
    if reference is None:
        return [None for _ in grad_lists[0]]

    combined: GradList = []
    for grads_for_param in zip(*grad_lists):
        total: torch.Tensor | None = None
        for grad, weight in zip(grads_for_param, weights):
            if grad is None:
                if total is None:
                    continue
                value = _zero_like(total)
            else:
                value = grad
            weighted = value * weight
            total = weighted if total is None else total + weighted
        combined.append(total)
    return combined


def scale_grad_list(grads: GradList, weight: float | torch.Tensor) -> GradList:
    """Scale a gradient list while preserving ``None`` entries."""

    return [None if grad is None else grad * weight for grad in grads]


def target_anchored_l2rw(
    *,
    anchor_grads: GradList,
    aux_grads: GradList,
    min_weight: float = 0.25,
    max_weight: float = 1.0,
    eps: float = 1e-12,
) -> tuple[GradList, PCGradStats]:
    """First-order target-anchored pseudo-example reweighting.

    The sparse target gradient is treated as the meta/validation signal.  A
    pseudo gradient receives high weight when it aligns with that signal and is
    downweighted when its first-order meta-gradient is conflicting.
    """

    if min_weight < 0.0 or max_weight < min_weight:
        raise ValueError("Require 0 <= min_weight <= max_weight for target_anchored_l2rw.")

    cosine_before = _cosine(aux_grads, anchor_grads, eps=eps)
    if cosine_before is None:
        weight = float(max_weight)
    else:
        alignment = max(0.0, min(1.0, float(cosine_before)))
        weight = float(min_weight) + (float(max_weight) - float(min_weight)) * alignment
    weighted_grads = scale_grad_list(aux_grads, weight)
    return weighted_grads, PCGradStats(
        cosine_before=cosine_before,
        cosine_after=_cosine(weighted_grads, anchor_grads, eps=eps),
        projected=abs(weight - 1.0) > 1e-12,
        weight=weight,
    )


def _cagrad_objective(
    *,
    anchor_norm: float,
    aux_norm: float,
    dot: float,
    g0_norm: float,
    c: float,
    anchor_weight: float,
    eps: float,
) -> float:
    aux_weight = 1.0 - anchor_weight
    gw_dot_g0 = 0.5 * (
        anchor_weight * (anchor_norm + dot)
        + aux_weight * (dot + aux_norm)
    )
    gw_norm_sq = (
        anchor_weight * anchor_weight * anchor_norm
        + 2.0 * anchor_weight * aux_weight * dot
        + aux_weight * aux_weight * aux_norm
    )
    return gw_dot_g0 + float(c) * g0_norm * float(max(gw_norm_sq, eps) ** 0.5)


def _minimize_cagrad_weight(
    *,
    anchor_norm: float,
    aux_norm: float,
    dot: float,
    g0_norm: float,
    c: float,
    eps: float,
    iterations: int = 32,
) -> float:
    """Solve the two-task CAGrad simplex problem by bounded golden search."""

    left = 0.0
    right = 1.0
    inv_phi = (5.0**0.5 - 1.0) / 2.0
    x1 = right - inv_phi * (right - left)
    x2 = left + inv_phi * (right - left)
    f1 = _cagrad_objective(
        anchor_norm=anchor_norm,
        aux_norm=aux_norm,
        dot=dot,
        g0_norm=g0_norm,
        c=c,
        anchor_weight=x1,
        eps=eps,
    )
    f2 = _cagrad_objective(
        anchor_norm=anchor_norm,
        aux_norm=aux_norm,
        dot=dot,
        g0_norm=g0_norm,
        c=c,
        anchor_weight=x2,
        eps=eps,
    )
    for _ in range(iterations):
        if f1 < f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - inv_phi * (right - left)
            f1 = _cagrad_objective(
                anchor_norm=anchor_norm,
                aux_norm=aux_norm,
                dot=dot,
                g0_norm=g0_norm,
                c=c,
                anchor_weight=x1,
                eps=eps,
            )
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + inv_phi * (right - left)
            f2 = _cagrad_objective(
                anchor_norm=anchor_norm,
                aux_norm=aux_norm,
                dot=dot,
                g0_norm=g0_norm,
                c=c,
                anchor_weight=x2,
                eps=eps,
            )
    candidates = [
        (0.0, _cagrad_objective(
            anchor_norm=anchor_norm,
            aux_norm=aux_norm,
            dot=dot,
            g0_norm=g0_norm,
            c=c,
            anchor_weight=0.0,
            eps=eps,
        )),
        (1.0, _cagrad_objective(
            anchor_norm=anchor_norm,
            aux_norm=aux_norm,
            dot=dot,
            g0_norm=g0_norm,
            c=c,
            anchor_weight=1.0,
            eps=eps,
        )),
        ((left + right) / 2.0, min(f1, f2)),
    ]
    return float(min(candidates, key=lambda item: item[1])[0])


def target_anchored_cagrad(
    *,
    anchor_grads: GradList,
    aux_grads: GradList,
    c: float = 0.4,
    rescale: int = 1,
    sum_scale: bool = True,
    eps: float = 1e-12,
) -> tuple[GradList, PCGradStats]:
    """Two-task CAGrad for sparse target supervision and pseudo loss.

    Returns a combined anchor+pseudo gradient.  ``sum_scale=True`` keeps the
    magnitude comparable to the usual ``anchor + pseudo`` loss sum.
    """

    cosine_before = _cosine(aux_grads, anchor_grads, eps=eps)
    dot_tensor = _dot(anchor_grads, aux_grads)
    anchor_norm_tensor = _squared_norm(anchor_grads)
    aux_norm_tensor = _squared_norm(aux_grads)
    if dot_tensor is None or anchor_norm_tensor is None or aux_norm_tensor is None:
        combined = combine_grad_lists(anchor_grads, aux_grads)
        return combined, PCGradStats(
            cosine_before=cosine_before,
            cosine_after=_cosine(combined, anchor_grads, eps=eps),
            projected=False,
            weight=1.0,
        )

    dot = float(dot_tensor.detach().cpu())
    anchor_norm = float(anchor_norm_tensor.detach().cpu())
    aux_norm = float(aux_norm_tensor.detach().cpu())
    g0_norm = max(0.25 * (anchor_norm + 2.0 * dot + aux_norm), 0.0) ** 0.5
    if anchor_norm <= eps or aux_norm <= eps or g0_norm <= eps:
        combined = combine_grad_lists(anchor_grads, aux_grads)
        return combined, PCGradStats(
            cosine_before=cosine_before,
            cosine_after=_cosine(combined, anchor_grads, eps=eps),
            projected=False,
            weight=1.0,
        )

    c = max(0.0, float(c))
    anchor_weight = _minimize_cagrad_weight(
        anchor_norm=anchor_norm,
        aux_norm=aux_norm,
        dot=dot,
        g0_norm=g0_norm,
        c=c,
        eps=eps,
    )
    aux_weight = 1.0 - anchor_weight
    gw_norm = max(
        anchor_weight * anchor_weight * anchor_norm
        + 2.0 * anchor_weight * aux_weight * dot
        + aux_weight * aux_weight * aux_norm,
        eps,
    ) ** 0.5
    lambda_value = c * g0_norm / gw_norm

    g0_grads = _weighted_sum_grad_lists([anchor_grads, aux_grads], [0.5, 0.5])
    gw_grads = _weighted_sum_grad_lists([anchor_grads, aux_grads], [anchor_weight, aux_weight])
    cagrad_grads = combine_grad_lists(g0_grads, scale_grad_list(gw_grads, lambda_value))
    if rescale == 1:
        cagrad_grads = scale_grad_list(cagrad_grads, 1.0 / (1.0 + c * c))
    elif rescale == 2:
        cagrad_grads = scale_grad_list(cagrad_grads, 1.0 / (1.0 + c))
    elif rescale != 0:
        raise ValueError("target_anchored_cagrad rescale must be 0, 1, or 2.")
    if sum_scale:
        cagrad_grads = scale_grad_list(cagrad_grads, 2.0)

    return cagrad_grads, PCGradStats(
        cosine_before=cosine_before,
        cosine_after=_cosine(cagrad_grads, anchor_grads, eps=eps),
        projected=True,
        weight=float(aux_weight),
    )


def combine_grad_lists(*grad_lists: GradList) -> GradList:
    """Sum gradient lists while preserving ``None`` for fully unused params."""

    if not grad_lists:
        return []
    combined: GradList = []
    for grads_for_param in zip(*grad_lists):
        total: torch.Tensor | None = None
        for grad in grads_for_param:
            if grad is None:
                continue
            total = grad.detach().clone() if total is None else total + grad.detach()
        combined.append(total)
    return combined


def clone_grad_list(grads: GradList) -> GradList:
    """Detach and clone a gradient list before assigning or accumulating it."""

    return [None if grad is None else grad.detach().clone() for grad in grads]


def add_grads_in_place(destination: GradList, source: GradList) -> GradList:
    """Accumulate ``source`` into ``destination`` without keeping extra lists."""

    if len(destination) != len(source):
        raise ValueError("Gradient lists must have the same length.")
    for idx, grad in enumerate(source):
        if grad is None:
            continue
        if destination[idx] is None:
            destination[idx] = grad.detach().clone()
        else:
            destination[idx].add_(grad.detach())
    return destination


def assign_grads(parameters: list[torch.nn.Parameter], grads: GradList) -> None:
    """Assign a detached gradient list to model parameters."""

    for parameter, grad in zip(parameters, grads):
        parameter.grad = None if grad is None else grad.detach().clone()
