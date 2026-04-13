from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import AttrDict

from baselines.fnp_daod.acquisition import apply_acquisition, clipped_gaussian_normalize
from baselines.fnp_daod.metrics import count_false_negatives
from baselines.fnp_daod.state import FNPDAODState, load_fnp_state, save_fnp_state


def _to_attr(value):
    if isinstance(value, dict):
        return AttrDict({k: _to_attr(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr(v) for v in value]
    return value


class AcquisitionTests(unittest.TestCase):
    def test_clipped_gaussian_normalize_handles_constant_positive_values(self):
        self.assertEqual(clipped_gaussian_normalize([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0])

    def test_apply_acquisition_multiplies_normalized_metrics(self):
        records = [
            {
                "sample_id": "a",
                "file_name": "a.jpg",
                "metrics": {"fn": 1.0, "loc": 1.0, "ent": 1.0, "div": 1.0},
            },
            {
                "sample_id": "b",
                "file_name": "b.jpg",
                "metrics": {"fn": 2.0, "loc": 2.0, "ent": 2.0, "div": 2.0},
            },
        ]
        scored = apply_acquisition(records)
        self.assertEqual(scored[0]["sample_id"], "b")
        self.assertGreater(scored[0]["acquisition_score"], scored[1]["acquisition_score"])


class MetricTests(unittest.TestCase):
    def test_false_negative_count_requires_class_aware_match(self):
        gt = [
            {"bbox": [0, 0, 10, 10], "category_id": 0},
            {"bbox": [20, 20, 30, 30], "category_id": 1},
        ]
        preds = [
            {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9},
            {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.9},
        ]
        self.assertEqual(count_false_negatives(gt, preds, iou_thresh=0.5, score_floor=0.1), 1)


class StateTests(unittest.TestCase):
    def test_state_roundtrip_preserves_sets(self):
        state = FNPDAODState(
            round_idx=1,
            queried_ids={"a", "b"},
            budget_total=10,
            budget_used=2,
            student_checkpoint="student.pth",
            teacher_checkpoint="teacher.pth",
            discriminator_checkpoint="disc.pt",
            teacher_discriminator_checkpoint="teacher_disc.pt",
            initialized=True,
            global_step=12,
        )
        tmp_dir = Path(tempfile.mkdtemp())
        state_path = tmp_dir / "state.json"
        save_fnp_state(state_path, state)
        loaded = load_fnp_state(state_path)
        self.assertEqual(loaded.queried_ids, {"a", "b"})
        self.assertTrue(loaded.initialized)


if __name__ == "__main__":
    unittest.main()
