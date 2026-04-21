from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.config import AttrDict
from src.engine.daod_pseudo_recalibration import (
    _coverage_precision,
    _label_prior_quota,
    _label_prior_ratio,
    _selected_fbeta,
    compute_pseudo_recalibration,
)
from src.engine.daod_stepwise_injection_trainer import _injection_points
from src.methods.daod_method import DAODRoundState
from src.methods.daod_stepwise_injection_method import DAODStepwiseInjectionMethod
from src.methods.daod_run_rounds import build_daod_method_from_cfg


def _to_attr(value):
    if isinstance(value, dict):
        return AttrDict({k: _to_attr(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr(v) for v in value]
    return value


def _cfg(training_scheme: str = "stepwise_injection") -> AttrDict:
    return _to_attr(
        {
            "run": {"root_dir": "runs"},
            "data": {
                "root": "/tmp/data",
                "source_domain": "cityscapes",
                "target_domain": "foggy_cityscapes",
                "num_classes": 8,
            },
            "detector": {
                "model_name": "dino_r50_4scale_12ep",
                "source_ckpt": "final",
                "min_size_test": 800,
                "max_size_test": 1333,
            },
            "method": {
                "training_scheme": training_scheme,
                "exp_name": "unit_test",
                "num_rounds": 2,
                "budget_total": 2,
                "selection": {
                    "strategy": "random",
                    "norm": "rank",
                    "features": {
                        "confident_score_thresh": 0.3,
                        "latent_score_floor": 0.05,
                        "top_k": 5,
                        "cross_view_iou_thresh": 0.3,
                    },
                },
                "train": {
                    "labeled_only_warmup_steps": 5,
                },
            },
        }
    )


class FakeStepwiseTrainer:
    def fit_with_injections(self, *, cfg, run_dir, state_in, initial_plan, remaining_budgets, plan_callback):
        return {
            "teacher_checkpoint": str(run_dir / "teacher_last.pth"),
            "student_checkpoint": str(run_dir / "student_last.pth"),
            "optimizer_checkpoint": str(run_dir / "optimizer_last.pt"),
            "scheduler_checkpoint": None,
            "global_step": 123,
            "queried_ids": list(initial_plan.queried_ids),
            "source_val_metrics": {},
            "target_val_metrics": {},
            "train_history": [],
        }


class DAODStepwiseTests(unittest.TestCase):
    def test_entrypoint_selects_stepwise_method(self):
        cfg = _cfg(training_scheme="stepwise_injection")
        device = torch.device("cpu")
        with patch("src.methods.daod_method.build_daod_dataset", return_value=[]), patch(
            "src.methods.daod_method.resolve_daod_method_run_dir",
            return_value=Path(tempfile.mkdtemp()),
        ):
            method = build_daod_method_from_cfg(cfg, device)
        self.assertIsInstance(method, DAODStepwiseInjectionMethod)

    def test_stepwise_method_runs_single_injected_session(self):
        cfg = _cfg(training_scheme="stepwise_injection")
        run_dir = Path(tempfile.mkdtemp())
        with patch("src.methods.daod_method.build_daod_dataset", return_value=[
            {"sample_id": "a", "file_name": "/tmp/a.jpg"},
            {"sample_id": "b", "file_name": "/tmp/b.jpg"},
        ]), patch(
            "src.methods.daod_method.resolve_daod_method_run_dir",
            return_value=run_dir,
        ):
            method = DAODStepwiseInjectionMethod(cfg=cfg, device=torch.device("cpu"), trainer=FakeStepwiseTrainer())
        state = DAODRoundState(
            round_idx=0,
            queried_ids=set(),
            budget_total=2,
            budget_used=0,
            teacher_checkpoint="teacher0.pth",
            student_checkpoint="student0.pth",
        )
        with patch.object(
            method,
            "plan_round",
            return_value=AttrDict({"round_idx": 0, "queried_ids": ["a"], "sample_plans": []}),
        ):
            state_out = method.run_all_rounds(source_ckpt=state.student_checkpoint, state_init=state)
        self.assertEqual(state_out.global_step, 123)
        self.assertEqual(state_out.queried_ids, {"a"})

    def test_injection_points_evenly_split_total_steps(self):
        self.assertEqual(_injection_points(2800, 1), [])
        self.assertEqual(_injection_points(2800, 2), [1400])
        self.assertEqual(_injection_points(2800, 3), [933, 1867])

    def test_label_prior_ratio_rebalances_against_pseudo_prior(self):
        thresholds, stats = _label_prior_ratio(
            [1, 9],
            [[0.95, 0.90, 0.85, 0.80], [0.45, 0.40, 0.35, 0.30]],
            base_score_min=0.5,
            cfg=_to_attr(
                {
                    "min_score_min": 0.35,
                    "max_score_min": 0.65,
                    "max_delta": 0.12,
                    "ratio_temperature": 0.75,
                    "smoothing": 1.0,
                }
            ),
        )
        self.assertGreater(thresholds[0], 0.5)
        self.assertLess(thresholds[1], 0.5)
        self.assertEqual(stats["base_pseudo_counts"], [4, 0])

    def test_label_prior_quota_maps_selected_label_prior_to_pseudo_counts(self):
        thresholds, stats = _label_prior_quota(
            [1, 9],
            [[0.95, 0.90, 0.85, 0.80], [0.45, 0.40, 0.35, 0.30]],
            base_score_min=0.5,
            cfg=_to_attr(
                {
                    "min_score_min": 0.30,
                    "max_score_min": 0.70,
                    "target_total_scale": 1.0,
                    "smoothing": 1.0,
                }
            ),
        )
        self.assertGreater(thresholds[0], 0.5)
        self.assertLess(thresholds[1], 0.5)
        self.assertGreater(stats["target_pseudo_counts"][1], stats["target_pseudo_counts"][0])

    def test_label_rarity_stage_scaled_uses_weaker_early_threshold_shift(self):
        target_dicts = [
            {
                "sample_id": "a",
                "annotations": [{"category_id": 0, "bbox": [0, 0, 1, 1]} for _ in range(10)],
            }
        ]
        cfg = _to_attr(
            {
                "method": "label_rarity_stage_scaled",
                "min_score_min": 0.35,
                "max_delta": 0.10,
                "smoothing": 0.0,
                "stage_scales": [0.25, 1.0],
            }
        )
        early_thresholds, early_stats = compute_pseudo_recalibration(
            target_dicts,
            {"a"},
            num_classes=2,
            base_score_min=0.5,
            recalibration_cfg=cfg,
            stage_idx=0,
        )
        late_thresholds, late_stats = compute_pseudo_recalibration(
            target_dicts,
            {"a"},
            num_classes=2,
            base_score_min=0.5,
            recalibration_cfg=cfg,
            stage_idx=1,
        )
        self.assertAlmostEqual(early_thresholds[1], 0.475)
        self.assertAlmostEqual(late_thresholds[1], 0.4)
        self.assertEqual(early_stats["method"], "label_rarity_stage_scaled")
        self.assertLess(early_stats["effective_max_delta"], late_stats["effective_max_delta"])

    def test_label_rarity_exp_is_smooth_and_bounded(self):
        target_dicts = [
            {
                "sample_id": "a",
                "annotations": [{"category_id": 0, "bbox": [0, 0, 1, 1]} for _ in range(10)]
                + [{"category_id": 1, "bbox": [0, 0, 1, 1]} for _ in range(5)],
            }
        ]
        thresholds, stats = compute_pseudo_recalibration(
            target_dicts,
            {"a"},
            num_classes=3,
            base_score_min=0.5,
            recalibration_cfg=_to_attr(
                {
                    "method": "label_rarity_exp",
                    "min_score_min": 0.38,
                    "max_delta": 0.10,
                    "smoothing": 0.0,
                    "exp_gamma": 1.0,
                }
            ),
            stage_idx=0,
        )
        self.assertEqual(stats["method"], "label_rarity_exp")
        self.assertAlmostEqual(thresholds[0], 0.5)
        self.assertLess(thresholds[1], 0.45)
        self.assertAlmostEqual(thresholds[2], 0.4)
        self.assertGreaterEqual(min(thresholds.values()), 0.38)

    def test_coverage_precision_lowers_under_confident_class_threshold(self):
        teacher_items = [
            {
                "sample": {
                    "sample_id": "a",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                },
                "query_rows": [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.42}],
            }
        ]
        thresholds, stats = _coverage_precision(
            [1, 0],
            teacher_items,
            num_classes=2,
            base_score_min=0.5,
            cfg=_to_attr(
                {
                    "min_score_min": 0.35,
                    "max_score_min": 0.65,
                    "lower_delta": 0.15,
                    "raise_delta": 0.10,
                    "precision_target": 0.7,
                }
            ),
        )
        self.assertLess(thresholds[0], 0.5)
        self.assertEqual(stats["gt_counts"][0], 1)

    def test_selected_fbeta_raises_threshold_to_remove_false_positive(self):
        teacher_items = [
            {
                "sample": {
                    "sample_id": "a",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                },
                "query_rows": [
                    {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.60},
                    {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.55},
                ],
            }
        ]
        thresholds, stats = _selected_fbeta(
            [1, 0],
            teacher_items,
            num_classes=2,
            base_score_min=0.5,
            cfg=_to_attr(
                {
                    "min_score_min": 0.35,
                    "max_score_min": 0.70,
                    "candidate_score_floor": 0.01,
                    "beta": 1.0,
                    "precision_floor": 0.75,
                    "precision_penalty": 0.5,
                }
            ),
        )
        self.assertAlmostEqual(thresholds[0], 0.60)
        self.assertEqual(stats["candidate_counts"][0], 2)


if __name__ == "__main__":
    unittest.main()
