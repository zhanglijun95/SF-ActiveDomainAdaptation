"""Method utilities for prior/margin/change/score."""

from __future__ import annotations

import torch


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def estimate_prior(prob: torch.Tensor) -> torch.Tensor:
    prior = prob.mean(dim=0)
    return prior / prior.sum().clamp(min=1e-12)


def debias_logits(logits: torch.Tensor, prior: torch.Tensor, lam: float) -> torch.Tensor:
    return logits - lam * torch.log(prior.clamp(min=1e-12))[None, :]


def margin_from_prob(prob: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(prob, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def change_l1(prob: torch.Tensor, prev_prob: torch.Tensor) -> torch.Tensor:
    return (prob - prev_prob).abs().sum(dim=1)


def rank_norm(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(torch.argsort(x))
    return order.float() / max(len(x) - 1, 1)
