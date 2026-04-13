"""Acquisition scoring for the isolated FNP baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


def clipped_gaussian_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    sigma = float(arr.std())
    if sigma <= 1e-12:
        if float(arr.max()) <= 0.0:
            return [0.0] * len(values)
        return [1.0 if float(value) > 0.0 else 0.0 for value in arr.tolist()]
    mu = float(arr.mean())
    normalized = np.maximum(0.0, (arr - (mu - 3.0 * sigma)) / (6.0 * sigma))
    return normalized.tolist()


def apply_acquisition(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    metric_keys = ("fn", "loc", "ent", "div")
    normalized_by_key = {
        key: clipped_gaussian_normalize([float(record["metrics"][key]) for record in records])
        for key in metric_keys
    }

    scored_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        normalized_metrics = {
            key: float(normalized_by_key[key][index])
            for key in metric_keys
        }
        acquisition_score = float(np.prod([normalized_metrics[key] for key in metric_keys], dtype=float))
        updated = dict(record)
        updated["normalized_metrics"] = normalized_metrics
        updated["acquisition_score"] = acquisition_score
        scored_records.append(updated)

    scored_records.sort(
        key=lambda record: (
            float(record["acquisition_score"]),
            float(record["metrics"]["fn"]),
            str(record["sample_id"]),
        ),
        reverse=True,
    )
    return scored_records
