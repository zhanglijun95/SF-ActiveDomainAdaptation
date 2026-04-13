"""Round-based DAOD method orchestration.

This module owns the round-planning layer for active DAOD:

- round state persistence
- target-train signal extraction on original / weak / strong views
- budgeted human-label selection
- handoff into the round trainer that performs teacher/student training

Important boundary:
- planning only decides which target images receive new human labels
- hard-vs-soft pseudo routing happens online inside the round trainer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from src.data.daod import build_daod_dataset, make_strong_view, make_weak_view, map_boxes_to_original_view
from src.data.daod.analysis import (
    percentile_rank_normalize,
    raw_output_to_query_rows,
    score_cross_view_support,
    score_geometry_structure,
    score_semantic_structure,
    summarize_scores,
)
from src.engine.daod_round_trainer import DAODMeanTeacherRoundTrainer
from src.engine.utils import resolve_daod_method_run_dir, resolve_daod_source_ckpt_path, save_json
from src.models.detrex_adapter import build_daod_model, run_daod_raw_outputs


@dataclass
class DAODRoundState:
    """Persistent state carried from one DAOD round to the next.

    Meanings:
    - `queried_ids`: target-train images that already received human labels in
      previous rounds
    - `teacher_checkpoint`: teacher weights used to generate pseudo targets in
      the next round
    - `student_checkpoint`: student weights used both for the next round's
      human-label selection pass and as the trainable model at round start
    """

    round_idx: int
    queried_ids: set[str]
    budget_total: int
    budget_used: int
    teacher_checkpoint: str
    student_checkpoint: str
    optimizer_checkpoint: str | None = None
    scheduler_checkpoint: str | None = None
    global_step: int = 0


@dataclass
class DAODSamplePlan:
    """Selection-side record for one target-train image.

    Meanings:
    - `selection_score`: final weighted score used to rank images for human
      labeling in this round
    - `selection_signals`: only the subset of features that actively contribute
      to `selection_score`
    - `features`: the full normalized feature table for this sample, saved for
      inspection/debugging even if a feature is not part of the current score
    """

    sample_id: str
    file_name: str
    selection_score: float
    selection_signals: dict[str, float]
    features: dict[str, float]


@dataclass
class DAODRoundPlan:
    """Plan produced before one round update.

    This plan only decides which new images become human labeled.
    Hard-vs-soft pseudo routing is intentionally *not* stored here, because it
    is decided online inside the round trainer from teacher outputs.
    """

    round_idx: int
    queried_ids: list[str]
    sample_plans: list[DAODSamplePlan]


def save_daod_round_state(path: str | Path, state: DAODRoundState) -> None:
    payload = asdict(state)
    payload["queried_ids"] = sorted(state.queried_ids)
    save_json(path, payload)


def _resolve_budget_total(cfg: Any, total_target: int) -> int:
    budget_cfg = getattr(cfg.method, "budget_total", 0)
    if isinstance(budget_cfg, float) and 0.0 < budget_cfg <= 1.0:
        return max(1, int(total_target * budget_cfg))
    return max(0, int(budget_cfg))


def _compute_round_budgets(budget_total: int, num_rounds: int) -> list[int]:
    if budget_total <= 0 or num_rounds <= 0:
        return []
    base = budget_total // num_rounds
    if base == 0:
        effective_rounds = min(num_rounds, budget_total)
        return [1] * effective_rounds
    budgets = [base] * num_rounds
    budgets[-1] += budget_total - sum(budgets)
    return budgets


def _remap_rows_to_original(query_rows: list[dict[str, Any]], view_meta: dict[str, Any]) -> list[dict[str, Any]]:
    remapped_boxes = map_boxes_to_original_view([row["bbox"] for row in query_rows], view_meta)
    remapped_rows = []
    for row, remapped_box in zip(query_rows, remapped_boxes):
        updated = dict(row)
        updated["bbox"] = remapped_box
        remapped_rows.append(updated)
    return remapped_rows


def _signal_specs(section: Any, default_specs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    signal_cfg = getattr(section, "signals", None)
    if signal_cfg is None:
        return default_specs
    specs: list[tuple[str, float]] = []
    for item in signal_cfg:
        name = str(item.name)
        weight = float(getattr(item, "weight", 1.0))
        specs.append((name, weight))
    return specs


def _signal_value(name: str, features: dict[str, float]) -> float:
    if name not in features:
        raise KeyError(f"Unknown DAOD signal: {name}")
    return float(features[name])


def _weighted_score(specs: list[tuple[str, float]], features: dict[str, float]) -> tuple[float, dict[str, float]]:
    parts: dict[str, float] = {}
    total = 0.0
    for name, weight in specs:
        value = _signal_value(name, features)
        parts[name] = value
        total += float(weight) * value
    return float(total), parts


def _latent_rows(
    query_rows: list[dict[str, Any]],
    *,
    latent_score_floor: float,
    prediction_score_thresh: float,
) -> list[dict[str, Any]]:
    """Select the current latent-confidence band from a full query list."""

    return [
        row
        for row in query_rows
        if latent_score_floor <= row["score"] < prediction_score_thresh
    ]


def _summary_feature_dict(scores: list[float], *, prefix: str, top_k: int) -> dict[str, float]:
    """Summarize one query-level score family into image-level features."""

    summary = summarize_scores(scores, top_k=top_k)
    return {
        f"{prefix}_mean_all": summary["mean_all"],
        f"{prefix}_topk_mean": summary["mean_topk"],
        f"{prefix}_max": summary["max"],
    }


def _latent_feature_bundle(
    original_query_rows: list[dict[str, Any]],
    *,
    latent_score_floor: float,
    prediction_score_thresh: float,
) -> dict[str, float]:
    """Image-level metadata about the current latent-confidence band."""

    latent_query_rows = _latent_rows(
        original_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    return {
        "latent_query_count": float(len(latent_query_rows)),
    }


def _semantic_feature_bundle(
    original_query_rows: list[dict[str, Any]],
    *,
    top_k: int,
    latent_score_floor: float,
    prediction_score_thresh: float,
) -> dict[str, float]:
    """Semantic-structure features from latent queries only."""

    latent_query_rows = _latent_rows(
        original_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    semantic_scores = [score_semantic_structure(row) for row in latent_query_rows]
    features = _summary_feature_dict(semantic_scores, prefix="semantic", top_k=top_k)
    ambiguity_scores = [1.0 - float(score) for score in semantic_scores]
    features.update(_summary_feature_dict(ambiguity_scores, prefix="semantic_ambiguity", top_k=top_k))
    return features


def _geometry_feature_bundle(
    original_query_rows: list[dict[str, Any]],
    *,
    top_k: int,
    latent_score_floor: float,
    prediction_score_thresh: float,
) -> dict[str, float]:
    """Geometry-stability features from latent queries only."""

    latent_query_rows = _latent_rows(
        original_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    geometry_scores = [score_geometry_structure(row) for row in latent_query_rows]
    features = _summary_feature_dict(geometry_scores, prefix="geometry", top_k=top_k)
    instability_scores = [1.0 - float(score) for score in geometry_scores]
    features.update(_summary_feature_dict(instability_scores, prefix="geometry_instability", top_k=top_k))
    return features


def _cross_view_feature_bundle(
    original_query_rows: list[dict[str, Any]],
    weak_query_rows: list[dict[str, Any]],
    strong_query_rows: list[dict[str, Any]],
    *,
    top_k: int,
    cross_view_iou_thresh: float,
    latent_score_floor: float,
    prediction_score_thresh: float,
) -> dict[str, float]:
    """Cross-view support features from latent queries only."""

    latent_query_rows = _latent_rows(
        original_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    weak_latent_rows = _latent_rows(
        weak_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    strong_latent_rows = _latent_rows(
        strong_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    cross_view_scores = []
    for row in latent_query_rows:
        weak_support = score_cross_view_support(row, weak_latent_rows, match_iou_thresh=cross_view_iou_thresh)
        strong_support = score_cross_view_support(row, strong_latent_rows, match_iou_thresh=cross_view_iou_thresh)
        cross_view_scores.append(max(weak_support, strong_support))
    features = _summary_feature_dict(cross_view_scores, prefix="cross_view", top_k=top_k)
    inconsistency_scores = [1.0 - float(score) for score in cross_view_scores]
    features.update(_summary_feature_dict(inconsistency_scores, prefix="cross_view_inconsistency", top_k=top_k))
    return features


def _teacher_student_feature_bundle(
    teacher_weak_query_rows: list[dict[str, Any]],
    student_strong_query_rows: list[dict[str, Any]],
    *,
    top_k: int,
    cross_view_iou_thresh: float,
    latent_score_floor: float,
    prediction_score_thresh: float,
) -> dict[str, float]:
    """Teacher-weak vs student-strong support/disagreement on latent queries."""

    teacher_latent_rows = _latent_rows(
        teacher_weak_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    student_latent_rows = _latent_rows(
        student_strong_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    support_scores = [
        score_cross_view_support(row, student_latent_rows, match_iou_thresh=cross_view_iou_thresh)
        for row in teacher_latent_rows
    ]
    features = _summary_feature_dict(support_scores, prefix="teacher_student", top_k=top_k)
    disagreement_scores = [1.0 - float(score) for score in support_scores]
    features.update(_summary_feature_dict(disagreement_scores, prefix="teacher_student_disagreement", top_k=top_k))
    return features


def _coverage_gap_feature_bundle(
    original_query_rows: list[dict[str, Any]],
    weak_query_rows: list[dict[str, Any]],
    strong_query_rows: list[dict[str, Any]],
    *,
    cross_view_iou_thresh: float,
    latent_score_floor: float,
    prediction_score_thresh: float,
) -> dict[str, float]:
    """Image-level gap between supported latent hypotheses and confident detections."""

    latent_query_rows = _latent_rows(
        original_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    weak_latent_rows = _latent_rows(
        weak_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    strong_latent_rows = _latent_rows(
        strong_query_rows,
        latent_score_floor=latent_score_floor,
        prediction_score_thresh=prediction_score_thresh,
    )
    confident_count = float(sum(1 for row in original_query_rows if row["score"] >= prediction_score_thresh))
    supported_latent_count = 0.0
    for row in latent_query_rows:
        weak_support = score_cross_view_support(row, weak_latent_rows, match_iou_thresh=cross_view_iou_thresh)
        strong_support = score_cross_view_support(row, strong_latent_rows, match_iou_thresh=cross_view_iou_thresh)
        if max(weak_support, strong_support) >= 0.5:
            supported_latent_count += 1.0
    coverage_gap_count = max(supported_latent_count - confident_count, 0.0)
    coverage_gap_ratio = supported_latent_count / max(confident_count, 1.0)
    return {
        "supported_latent_count": supported_latent_count,
        "coverage_gap_count": coverage_gap_count,
        "coverage_gap_ratio": float(coverage_gap_ratio),
    }


def _build_class_rarity_lookup(
    target_train: list[dict[str, Any]] | Any,
    labeled_ids: set[str],
    *,
    num_classes: int,
) -> dict[int, float]:
    """Inverse-frequency rarity weights from the currently labeled target set."""

    class_counts = np.zeros(max(int(num_classes), 1), dtype=float)
    for sample in target_train:
        if sample["sample_id"] not in labeled_ids:
            continue
        annotations = sample.get("annotations", [])
        for ann in annotations:
            category_id = int(ann.get("category_id", -1))
            if 0 <= category_id < len(class_counts):
                class_counts[category_id] += 1.0
    rarity = 1.0 / np.sqrt(class_counts + 1.0)
    return {idx: float(value) for idx, value in enumerate(rarity.tolist())}


def _class_rarity_feature_bundle(
    original_query_rows: list[dict[str, Any]],
    *,
    class_rarity_lookup: dict[int, float],
    latent_score_floor: float,
) -> dict[str, float]:
    """Rarity of classes predicted in this image relative to current labeled set."""

    class_scores: dict[int, float] = {}
    for row in original_query_rows:
        score = float(row["score"])
        if score < latent_score_floor:
            continue
        category_id = int(row["category_id"])
        class_scores[category_id] = max(class_scores.get(category_id, 0.0), score)
    if not class_scores:
        return {
            "class_rarity_mean": 0.0,
            "class_rarity_max": 0.0,
        }
    weighted_values = []
    weights = []
    max_value = 0.0
    for category_id, score in class_scores.items():
        rarity = float(class_rarity_lookup.get(category_id, 1.0))
        weighted_values.append(rarity * score)
        weights.append(score)
        max_value = max(max_value, rarity)
    return {
        "class_rarity_mean": float(sum(weighted_values) / max(sum(weights), 1e-12)),
        "class_rarity_max": float(max_value),
    }


def _confident_feature_bundle(
    original_query_rows: list[dict[str, Any]],
    *,
    prediction_score_thresh: float,
) -> dict[str, float]:
    """Features from confident detections only."""

    confident_rows = [row for row in original_query_rows if row["score"] >= prediction_score_thresh]
    confident_scores = np.asarray([row["score"] for row in confident_rows], dtype=float)
    return {
        "confident_count": float(len(confident_rows)),
        "confident_mean_score": float(confident_scores.mean()) if len(confident_scores) else 0.0,
        "confident_max_score": float(confident_scores.max()) if len(confident_scores) else 0.0,
    }


def _required_feature_bundles(signal_names: list[str]) -> set[str]:
    """Map configured signal names to the feature bundles they need."""

    bundles: set[str] = set()
    for name in signal_names:
        if name == "latent_query_count":
            bundles.add("latent")
        elif name.startswith("coverage_gap_") or name == "supported_latent_count":
            bundles.add("coverage_gap")
        elif name.startswith("class_rarity_"):
            bundles.add("class_rarity")
        elif name.startswith("teacher_student_disagreement_"):
            bundles.add("teacher_student")
        elif name.startswith("teacher_student_"):
            bundles.add("teacher_student")
        elif name.startswith("semantic_"):
            bundles.add("semantic")
        elif name.startswith("semantic_ambiguity_"):
            bundles.add("semantic")
        elif name.startswith("geometry_"):
            bundles.add("geometry")
        elif name.startswith("geometry_instability_"):
            bundles.add("geometry")
        elif name.startswith("cross_view_"):
            bundles.add("cross_view")
        elif name.startswith("cross_view_inconsistency_"):
            bundles.add("cross_view")
        elif name.startswith("confident_"):
            bundles.add("confident")
        else:
            raise KeyError(
                f"Cannot infer feature bundle for signal '{name}'. "
                "Add an explicit bundle mapping before using this signal."
            )
    return bundles


def _normalize_values(values: list[float], *, norm_type: str) -> list[float]:
    """Normalize one feature column with the configured selection normalization."""

    kind = str(norm_type).strip().lower()
    if kind == "rank":
        return percentile_rank_normalize(values).tolist()
    if kind in {"minmax", "min-max", "min_max"}:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return []
        min_value = float(arr.min())
        max_value = float(arr.max())
        scale = max_value - min_value
        if scale <= 1e-12:
            return [0.0] * len(values)
        return ((arr - min_value) / scale).tolist()
    if kind in {"zscore", "z-score", "z_score"}:
        arr = np.asarray(values, dtype=float)
        std = float(arr.std())
        if std <= 1e-12:
            return [0.0] * len(values)
        mean = float(arr.mean())
        return ((arr - mean) / std).tolist()
    raise ValueError(f"Unsupported selection normalization: {norm_type}. Use 'rank', 'minmax', or 'zscore'.")


class BaseDAODRoundTrainer:
    """Interface for one DAOD round update."""

    def fit_round(
        self,
        *,
        cfg: Any,
        round_dir: Path,
        state_in: DAODRoundState,
        plan: DAODRoundPlan,
    ) -> dict[str, Any]:
        round_dir.mkdir(parents=True, exist_ok=True)
        return {
            "teacher_checkpoint": state_in.teacher_checkpoint,
            "student_checkpoint": state_in.student_checkpoint,
            "optimizer_checkpoint": state_in.optimizer_checkpoint,
            "scheduler_checkpoint": state_in.scheduler_checkpoint,
            "global_step": state_in.global_step,
            "status": "no_training",
        }


class DAODRoundMethod:
    def __init__(self, cfg: Any, device: torch.device, trainer: BaseDAODRoundTrainer | None = None) -> None:
        self.cfg = cfg
        self.device = device
        self.run_dir = resolve_daod_method_run_dir(cfg)
        self.target_train = build_daod_dataset(cfg, split="target_train", transform=None)
        self.trainer = trainer or DAODMeanTeacherRoundTrainer(cfg=cfg, device=device)

        selection_cfg = getattr(getattr(cfg, "method", object()), "selection", object())
        selection_features_cfg = getattr(selection_cfg, "features", object())

        # `top_k` controls image-level summarization of latent-query scores.
        # Example: `geometry_topk_mean` is the mean over the best `top_k`
        # low-confidence geometry scores in the image.
        self.top_k = int(getattr(selection_features_cfg, "top_k", 5))
        # Queries above this score are treated as normal detections, not latent
        # weak evidence.
        self.prediction_score_thresh = float(getattr(selection_features_cfg, "confident_score_thresh", 0.30))
        # Queries below this floor are ignored as too weak/noisy for selection
        # features. So the latent band is:
        #   latent_score_floor <= query_score < prediction_score_thresh
        self.latent_score_floor = float(getattr(selection_features_cfg, "latent_score_floor", 0.05))
        # Minimum box IoU used when checking whether weak/strong views support
        # the same latent hypothesis.
        self.cross_view_iou_thresh = float(getattr(selection_features_cfg, "cross_view_iou_thresh", 0.30))
        # Feature normalization used before weighted score combination.
        # - `rank`: percentile-rank normalization in roughly [0, 1]
        # - `zscore`: standard score normalization with mean 0 and std 1
        self.selection_norm = str(getattr(selection_cfg, "norm", "rank"))

        self.selection_specs = _signal_specs(
            selection_cfg,
            default_specs=[("geometry_mean_all", 1.0)],
        )
        self.selection_strategy = str(getattr(selection_cfg, "strategy", "score")).strip().lower()
        self.selection_signal_names = [name for name, _ in self.selection_specs]
        self.required_bundles = _required_feature_bundles(self.selection_signal_names)
        self.selection_needs_aug_views = bool(
            {"cross_view", "teacher_student", "coverage_gap"} & self.required_bundles
        )
        self.selection_batch_size = int(getattr(selection_cfg, "batch_size", 8))
        self.selection_log_period = int(getattr(selection_cfg, "log_period", 50))

    def _build_adapter(self, checkpoint_path: str):
        adapter = build_daod_model(self.cfg, load_weights=False)
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(adapter.model).load(str(ckpt))
        adapter.model.eval()
        return adapter

    def _feature_row_from_raw_views(
        self,
        sample: dict[str, Any],
        *,
        original_raw: dict[str, Any],
        teacher_weak_raw: dict[str, Any] | None,
        weak_raw: dict[str, Any] | None,
        strong_raw: dict[str, Any] | None,
        weak_meta: dict[str, Any] | None,
        strong_meta: dict[str, Any] | None,
        class_rarity_lookup: dict[int, float] | None = None,
    ) -> dict[str, float]:
        """Build one image's selection features from already-computed raw views."""

        image_size = (sample["height"], sample["width"])
        original_rows = raw_output_to_query_rows(original_raw, image_size=image_size)
        teacher_weak_rows: list[dict[str, Any]] = []
        weak_rows: list[dict[str, Any]] = []
        strong_rows: list[dict[str, Any]] = []
        if teacher_weak_raw is not None and weak_meta is not None:
            teacher_weak_rows = _remap_rows_to_original(
                raw_output_to_query_rows(teacher_weak_raw, image_size=image_size),
                weak_meta,
            )
        if weak_raw is not None and weak_meta is not None:
            weak_rows = _remap_rows_to_original(
                raw_output_to_query_rows(weak_raw, image_size=image_size),
                weak_meta,
            )
        if strong_raw is not None and strong_meta is not None:
            strong_rows = _remap_rows_to_original(
                raw_output_to_query_rows(strong_raw, image_size=image_size),
                strong_meta,
            )

        features: dict[str, float] = {}
        if "latent" in self.required_bundles:
            features.update(
                _latent_feature_bundle(
                    original_rows,
                    latent_score_floor=self.latent_score_floor,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        if "semantic" in self.required_bundles:
            features.update(
                _semantic_feature_bundle(
                    original_rows,
                    top_k=self.top_k,
                    latent_score_floor=self.latent_score_floor,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        if "geometry" in self.required_bundles:
            features.update(
                _geometry_feature_bundle(
                    original_rows,
                    top_k=self.top_k,
                    latent_score_floor=self.latent_score_floor,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        if "cross_view" in self.required_bundles:
            features.update(
                _cross_view_feature_bundle(
                    original_rows,
                    weak_rows,
                    strong_rows,
                    top_k=self.top_k,
                    cross_view_iou_thresh=self.cross_view_iou_thresh,
                    latent_score_floor=self.latent_score_floor,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        if "teacher_student" in self.required_bundles:
            features.update(
                _teacher_student_feature_bundle(
                    teacher_weak_rows,
                    strong_rows,
                    top_k=self.top_k,
                    cross_view_iou_thresh=self.cross_view_iou_thresh,
                    latent_score_floor=self.latent_score_floor,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        if "coverage_gap" in self.required_bundles:
            features.update(
                _coverage_gap_feature_bundle(
                    original_rows,
                    weak_rows,
                    strong_rows,
                    cross_view_iou_thresh=self.cross_view_iou_thresh,
                    latent_score_floor=self.latent_score_floor,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        if "confident" in self.required_bundles:
            features.update(
                _confident_feature_bundle(
                    original_rows,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        if "class_rarity" in self.required_bundles and class_rarity_lookup is not None:
            features.update(
                _class_rarity_feature_bundle(
                    original_rows,
                    class_rarity_lookup=class_rarity_lookup,
                    latent_score_floor=self.latent_score_floor,
                )
            )
        return features

    def _selection_batches(self) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for index in range(len(self.target_train)):
            current.append(self.target_train[index])
            if len(current) >= self.selection_batch_size:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    def infer_target_train(self, state_in: DAODRoundState) -> list[DAODSamplePlan]:
        """Score every target-train image for human-label selection.

        Steps:
        1. run the current student checkpoint on original / weak / strong views
        2. build normalized image-level features
        3. combine the configured selection signals into one selection score
        """

        student_adapter = self._build_adapter(state_in.student_checkpoint)
        teacher_adapter = self._build_adapter(state_in.teacher_checkpoint) if "teacher_student" in self.required_bundles else None
        try:
            sample_plans: list[DAODSamplePlan] = []
            raw_feature_rows: list[dict[str, Any]] = []
            total_samples = len(self.target_train)
            batches = self._selection_batches()
            started_at = time.time()
            class_rarity_lookup = (
                _build_class_rarity_lookup(
                    self.target_train,
                    set(state_in.queried_ids),
                    num_classes=int(getattr(self.cfg.data, "num_classes", 1)),
                )
                if "class_rarity" in self.required_bundles
                else None
            )

            for batch_idx, sample_batch in enumerate(batches, start=1):
                original_batch: list[dict[str, Any]] = []
                weak_batch: list[dict[str, Any]] = []
                strong_batch: list[dict[str, Any]] = []
                batch_context: list[dict[str, Any]] = []

                for sample in sample_batch:
                    original_image = Image.open(sample["file_name"]).convert("RGB")
                    weak_meta = None
                    strong_meta = None

                    original_item = dict(sample)
                    original_item["image"] = original_image

                    original_batch.append(original_item)
                    if self.selection_needs_aug_views:
                        weak_image, weak_meta = make_weak_view(original_image.copy())
                        strong_image, strong_meta = make_strong_view(original_image.copy())

                        weak_item = dict(sample)
                        weak_item["image"] = weak_image
                        weak_item["sample_id"] = f"{sample['sample_id']}::weak"

                        strong_item = dict(sample)
                        strong_item["image"] = strong_image
                        strong_item["sample_id"] = f"{sample['sample_id']}::strong"

                        weak_batch.append(weak_item)
                        strong_batch.append(strong_item)
                    batch_context.append(
                        {
                            "sample": sample,
                            "weak_meta": weak_meta,
                            "strong_meta": strong_meta,
                        }
                    )

                original_raw_outputs = run_daod_raw_outputs(student_adapter, original_batch, with_grad=False)
                weak_raw_outputs = (
                    run_daod_raw_outputs(student_adapter, weak_batch, with_grad=False)
                    if weak_batch
                    else [None] * len(sample_batch)
                )
                strong_raw_outputs = (
                    run_daod_raw_outputs(student_adapter, strong_batch, with_grad=False)
                    if strong_batch
                    else [None] * len(sample_batch)
                )
                teacher_weak_raw_outputs = (
                    run_daod_raw_outputs(teacher_adapter, weak_batch, with_grad=False)
                    if teacher_adapter is not None and weak_batch
                    else [None] * len(sample_batch)
                )

                for context, original_raw, teacher_weak_raw, weak_raw, strong_raw in zip(
                    batch_context, original_raw_outputs, teacher_weak_raw_outputs, weak_raw_outputs, strong_raw_outputs
                ):
                    sample = context["sample"]
                    features = self._feature_row_from_raw_views(
                        sample,
                        original_raw=original_raw,
                        teacher_weak_raw=teacher_weak_raw,
                        weak_raw=weak_raw,
                        strong_raw=strong_raw,
                        weak_meta=context["weak_meta"],
                        strong_meta=context["strong_meta"],
                        class_rarity_lookup=class_rarity_lookup,
                    )
                    raw_feature_rows.append(
                        {
                            "sample_id": sample["sample_id"],
                            "file_name": sample["file_name"],
                            **features,
                        }
                    )

                if self.selection_log_period > 0 and (
                    batch_idx == 1
                    or batch_idx % self.selection_log_period == 0
                    or batch_idx == len(batches)
                ):
                    done = min(batch_idx * self.selection_batch_size, total_samples)
                    elapsed = time.time() - started_at
                    print(
                        f"[DAOD][selection] {done}/{total_samples} images "
                        f"({batch_idx}/{len(batches)} batches, bs={self.selection_batch_size}) "
                        f"elapsed={elapsed:.1f}s"
                    )

            # Normalize every numeric feature according to `selection.norm` so
            # weighted sums across different signal families stay on a comparable
            # scale.
            numeric_keys = [key for key in raw_feature_rows[0].keys() if key not in {"sample_id", "file_name"}] if raw_feature_rows else []
            for key in numeric_keys:
                values = [float(row[key]) for row in raw_feature_rows]
                normalized = _normalize_values(values, norm_type=self.selection_norm)
                for row, value in zip(raw_feature_rows, normalized):
                    row[key] = float(value)

            for row in raw_feature_rows:
                selection_score, selection_parts = _weighted_score(self.selection_specs, row)
                sample_plans.append(
                    DAODSamplePlan(
                        sample_id=str(row["sample_id"]),
                        file_name=str(row["file_name"]),
                        selection_score=selection_score,
                        selection_signals=selection_parts,
                        features={key: float(value) for key, value in row.items() if key not in {"sample_id", "file_name"}},
                    )
                )
            return sample_plans
        finally:
            for adapter in (student_adapter, teacher_adapter):
                if adapter is None:
                    continue
                if hasattr(adapter, "model"):
                    try:
                        adapter.model.cpu()
                    except Exception:
                        pass
                del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _select_top_scored(self, candidates: list[DAODSamplePlan], budget_k: int) -> list[str]:
        if budget_k <= 0 or not candidates:
            return []
        return [plan.sample_id for plan in candidates[:budget_k]]

    def _random_sample_plans(self) -> list[DAODSamplePlan]:
        """Create placeholder sample plans for random-selection baselines."""

        sample_plans: list[DAODSamplePlan] = []
        for sample in self.target_train:
            sample_plans.append(
                DAODSamplePlan(
                    sample_id=str(sample["sample_id"]),
                    file_name=str(sample["file_name"]),
                    selection_score=0.0,
                    selection_signals={},
                    features={},
                )
            )
        return sample_plans

    def _select_random(self, candidates: list[DAODSamplePlan], budget_k: int, *, round_idx: int) -> list[str]:
        if budget_k <= 0 or not candidates:
            return []
        rng = random.Random(int(getattr(self.cfg, "seed", 42)) + int(round_idx))
        candidate_ids = [plan.sample_id for plan in candidates]
        rng.shuffle(candidate_ids)
        return candidate_ids[:budget_k]

    def plan_round(self, state_in: DAODRoundState, budget_k: int) -> DAODRoundPlan:
        if self.selection_strategy == "random":
            sample_plans = self._random_sample_plans()
        else:
            sample_plans = self.infer_target_train(state_in)
            sample_plans.sort(key=lambda plan: plan.selection_score, reverse=True)

        available = [plan for plan in sample_plans if plan.sample_id not in state_in.queried_ids]
        if self.selection_strategy == "random":
            queried_ids = self._select_random(available, budget_k, round_idx=state_in.round_idx)
        else:
            queried_ids = self._select_top_scored(available, budget_k)

        return DAODRoundPlan(
            round_idx=state_in.round_idx,
            queried_ids=queried_ids,
            sample_plans=sample_plans,
        )

    def _round_dir(self, round_idx: int) -> Path:
        return self.run_dir / f"round_{round_idx}"

    def run_round(self, state_in: DAODRoundState, budget_k: int) -> DAODRoundState:
        plan = self.plan_round(state_in, budget_k)
        round_dir = self._round_dir(state_in.round_idx)
        round_dir.mkdir(parents=True, exist_ok=True)

        save_json(
            round_dir / "plan.json",
            {
                "round_idx": plan.round_idx,
                "queried_ids": plan.queried_ids,
                "sample_plans": [asdict(sample_plan) for sample_plan in plan.sample_plans],
            },
        )

        trainer_summary = self.trainer.fit_round(
            cfg=self.cfg,
            round_dir=round_dir,
            state_in=state_in,
            plan=plan,
        )
        state_out = DAODRoundState(
            round_idx=state_in.round_idx + 1,
            queried_ids=set(state_in.queried_ids).union(plan.queried_ids),
            budget_total=state_in.budget_total,
            budget_used=state_in.budget_used + len(plan.queried_ids),
            teacher_checkpoint=str(trainer_summary["teacher_checkpoint"]),
            student_checkpoint=str(trainer_summary["student_checkpoint"]),
            optimizer_checkpoint=trainer_summary.get("optimizer_checkpoint"),
            scheduler_checkpoint=trainer_summary.get("scheduler_checkpoint"),
            global_step=int(trainer_summary.get("global_step", state_in.global_step)),
        )
        save_daod_round_state(self.run_dir / "state_last.json", state_out)
        save_json(
            round_dir / "summary.json",
            {
                "round_idx": state_in.round_idx,
                "budget_k": budget_k,
                "budget_used": state_out.budget_used,
                "budget_total": state_out.budget_total,
                "selected_count": len(plan.queried_ids),
                "trainer": trainer_summary,
            },
        )
        return state_out

    def run_all_rounds(self, source_ckpt: str, state_init: DAODRoundState | None = None) -> DAODRoundState:
        budget_total = _resolve_budget_total(self.cfg, total_target=len(self.target_train))
        num_rounds = int(getattr(self.cfg.method, "num_rounds", 1))
        budget_schedule = _compute_round_budgets(budget_total, num_rounds)
        if state_init is None:
            state = DAODRoundState(
                round_idx=0,
                queried_ids=set(),
                budget_total=budget_total,
                budget_used=0,
                teacher_checkpoint=str(source_ckpt),
                student_checkpoint=str(source_ckpt),
                optimizer_checkpoint=None,
                scheduler_checkpoint=None,
                global_step=0,
            )
        else:
            state = state_init

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for budget_k in budget_schedule:
            state = self.run_round(state, budget_k)
        return state


def build_default_daod_round_state(cfg: Any) -> DAODRoundState:
    source_ckpt = resolve_daod_source_ckpt_path(cfg, which=str(getattr(getattr(cfg, "detector", object()), "source_ckpt", "best")))
    return DAODRoundState(
        round_idx=0,
        queried_ids=set(),
        budget_total=_resolve_budget_total(cfg, total_target=len(build_daod_dataset(cfg, split="target_train", transform=None))),
        budget_used=0,
        teacher_checkpoint=str(source_ckpt),
        student_checkpoint=str(source_ckpt),
        optimizer_checkpoint=None,
        scheduler_checkpoint=None,
        global_step=0,
    )
