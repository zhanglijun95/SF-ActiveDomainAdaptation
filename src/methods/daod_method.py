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
from src.models.detrex_adapter import build_daod_model, run_daod_inference_with_raw


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
    return _summary_feature_dict(semantic_scores, prefix="semantic", top_k=top_k)


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
    return _summary_feature_dict(geometry_scores, prefix="geometry", top_k=top_k)


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
    return _summary_feature_dict(cross_view_scores, prefix="cross_view", top_k=top_k)


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
        elif name.startswith("semantic_"):
            bundles.add("semantic")
        elif name.startswith("geometry_"):
            bundles.add("geometry")
        elif name.startswith("cross_view_"):
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
    if kind in {"zscore", "z-score", "z_score"}:
        arr = np.asarray(values, dtype=float)
        std = float(arr.std())
        if std <= 1e-12:
            return [0.0] * len(values)
        mean = float(arr.mean())
        return ((arr - mean) / std).tolist()
    raise ValueError(f"Unsupported selection normalization: {norm_type}. Use 'rank' or 'zscore'.")


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
        self.selection_signal_names = [name for name, _ in self.selection_specs]
        self.required_bundles = _required_feature_bundles(self.selection_signal_names)

    def _build_adapter(self, checkpoint_path: str):
        adapter = build_daod_model(self.cfg, load_weights=False)
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(adapter.model).load(str(ckpt))
        adapter.model.eval()
        return adapter

    def _sample_feature_row(self, adapter, sample: dict[str, Any]) -> dict[str, float]:
        """Compute all configured selection features for one target-train image.

        Important design:
        - the planner always keeps all original / weak / strong query rows
        - each bundle decides internally which confidence region it uses
        - the active bundles are chosen from the configured selection signals
        """

        original_image = Image.open(sample["file_name"]).convert("RGB")
        weak_image, weak_meta = make_weak_view(original_image.copy())
        strong_image, strong_meta = make_strong_view(original_image.copy())

        def _clone_with_image(image: Image.Image, suffix: str) -> dict[str, Any]:
            cloned = dict(sample)
            cloned["image"] = image
            cloned["sample_id"] = f"{sample['sample_id']}::{suffix}"
            return cloned

        original_output = run_daod_inference_with_raw(adapter, sample)[0]
        weak_output = run_daod_inference_with_raw(adapter, _clone_with_image(weak_image, "weak"))[0]
        strong_output = run_daod_inference_with_raw(adapter, _clone_with_image(strong_image, "strong"))[0]

        image_size = (sample["height"], sample["width"])
        original_rows = raw_output_to_query_rows(original_output["raw_output"], image_size=image_size)
        weak_rows = _remap_rows_to_original(
            raw_output_to_query_rows(weak_output["raw_output"], image_size=image_size),
            weak_meta,
        )
        strong_rows = _remap_rows_to_original(
            raw_output_to_query_rows(strong_output["raw_output"], image_size=image_size),
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
        if "confident" in self.required_bundles:
            features.update(
                _confident_feature_bundle(
                    original_rows,
                    prediction_score_thresh=self.prediction_score_thresh,
                )
            )
        return features

    def infer_target_train(self, checkpoint_path: str) -> list[DAODSamplePlan]:
        """Score every target-train image for human-label selection.

        Steps:
        1. run the current student checkpoint on original / weak / strong views
        2. build normalized image-level features
        3. combine the configured selection signals into one selection score
        """

        adapter = self._build_adapter(checkpoint_path)
        sample_plans: list[DAODSamplePlan] = []
        raw_feature_rows: list[dict[str, Any]] = []
        for index in range(len(self.target_train)):
            sample = self.target_train[index]
            features = self._sample_feature_row(adapter, sample)
            raw_feature_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "file_name": sample["file_name"],
                    **features,
                }
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

    def _select_top_scored(self, candidates: list[DAODSamplePlan], budget_k: int) -> list[str]:
        if budget_k <= 0 or not candidates:
            return []
        return [plan.sample_id for plan in candidates[:budget_k]]

    def plan_round(self, state_in: DAODRoundState, budget_k: int) -> DAODRoundPlan:
        sample_plans = self.infer_target_train(state_in.student_checkpoint)
        sample_plans.sort(key=lambda plan: plan.selection_score, reverse=True)

        available = [plan for plan in sample_plans if plan.sample_id not in state_in.queried_ids]
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
    )
