"""Sparse target-label split helpers for SFOD baseline plugins."""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample["sample_id"])


def without_annotations(sample: dict[str, Any]) -> dict[str, Any]:
    """Return an unlabeled target view so GT cannot leak into pseudo training."""

    cloned = dict(sample)
    cloned["annotations"] = []
    return cloned


def resolve_budget_count(budget_cfg: Any, total_count: int) -> int:
    if total_count <= 0:
        return 0
    if isinstance(budget_cfg, float) and 0.0 < float(budget_cfg) <= 1.0:
        return min(total_count, max(1, int(round(float(budget_cfg) * float(total_count)))))
    return min(total_count, max(0, int(budget_cfg)))


def build_sparse_target_split(
    target_train: list[dict[str, Any]],
    active_cfg: Any,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], dict[str, Any]]:
    active_enabled = bool(getattr(active_cfg, "enabled", False))
    if not active_enabled:
        return [], [without_annotations(sample) for sample in target_train], set(), {
            "enabled": False,
            "selected_ids": [],
            "sample_plans": [
                {"sample_id": sample_id(sample), "selected": False, "role": "unlabeled"}
                for sample in target_train
            ],
        }

    strategy = str(getattr(active_cfg, "strategy", "random")).strip().lower()
    if strategy != "random":
        raise ValueError(f"SFOD sparse-label plugin currently supports only strategy=random, got {strategy!r}")

    budget_cfg = getattr(active_cfg, "budget_total", 0)
    budget_k = resolve_budget_count(budget_cfg, len(target_train))
    rng = np.random.default_rng(int(seed))
    if budget_k > 0:
        selected_indices = sorted(int(idx) for idx in rng.choice(len(target_train), size=budget_k, replace=False))
    else:
        selected_indices = []

    selected_ids = {sample_id(target_train[idx]) for idx in selected_indices}
    selected_order = [sample_id(target_train[idx]) for idx in selected_indices]
    target_labeled = [sample for sample in target_train if sample_id(sample) in selected_ids]
    target_unlabeled = [without_annotations(sample) for sample in target_train if sample_id(sample) not in selected_ids]
    sample_plans = [
        {
            "sample_id": sample_id(sample),
            "selected": sample_id(sample) in selected_ids,
            "role": "labeled" if sample_id(sample) in selected_ids else "unlabeled",
        }
        for sample in target_train
    ]
    plan = {
        "enabled": True,
        "strategy": strategy,
        "budget_total": budget_cfg,
        "budget_k": int(budget_k),
        "target_total": len(target_train),
        "selected_ids": selected_order,
        "sample_plans": sample_plans,
    }
    return target_labeled, target_unlabeled, selected_ids, plan

