"""Classification metrics."""

from __future__ import annotations

from typing import Any

import torch


def accuracy_top1(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if gt.numel() == 0:
        return 0.0
    return float((pred == gt).float().mean().item())


def per_class_accuracy(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    acc_per_class: list[float] = []
    support: list[int] = []
    for c in range(num_classes):
        mask = gt == c
        n = int(mask.sum().item())
        support.append(n)
        if n == 0:
            acc_per_class.append(0.0)
        else:
            acc_per_class.append(float((pred[mask] == gt[mask]).float().mean().item()))
    mean_acc = float(sum(acc_per_class) / max(num_classes, 1))
    return {"acc_per_class": acc_per_class, "mean_acc": mean_acc, "support": support}


def compute_classification_metrics(
    logits: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    with_per_class: bool = True,
) -> dict[str, Any]:
    pred = logits.argmax(dim=1)
    out: dict[str, Any] = {"acc_top1": accuracy_top1(pred, gt)}
    if with_per_class:
        out.update(per_class_accuracy(pred, gt, num_classes))
    return out
