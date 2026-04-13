from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.config import AttrDict
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


if __name__ == "__main__":
    unittest.main()
