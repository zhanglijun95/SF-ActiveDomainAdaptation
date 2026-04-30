from __future__ import annotations

import unittest
from types import SimpleNamespace

from baselines.sfod_common.active import build_sparse_target_split, resolve_budget_count


class SFODSparseLabelSplitTests(unittest.TestCase):
    def test_random_sparse_split_is_deterministic(self):
        target_train = [
            {"sample_id": f"target-{idx}", "annotations": [{"category_id": idx % 8}]}
            for idx in range(20)
        ]
        active_cfg = SimpleNamespace(enabled=True, strategy="random", budget_total=0.05)

        labeled_a, unlabeled_a, selected_a, plan_a = build_sparse_target_split(target_train, active_cfg, seed=42)
        labeled_b, unlabeled_b, selected_b, plan_b = build_sparse_target_split(target_train, active_cfg, seed=42)

        self.assertEqual(plan_a["selected_ids"], plan_b["selected_ids"])
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(len(labeled_a), 1)
        self.assertEqual(len(unlabeled_a), 19)
        self.assertEqual(plan_a["budget_k"], 1)

    def test_unlabeled_target_has_no_gt_annotations(self):
        target_train = [
            {"sample_id": f"target-{idx}", "annotations": [{"category_id": idx % 8, "bbox": [0, 0, 1, 1]}]}
            for idx in range(10)
        ]
        active_cfg = SimpleNamespace(enabled=True, strategy="random", budget_total=0.2)

        labeled, unlabeled, selected_ids, plan = build_sparse_target_split(target_train, active_cfg, seed=7)

        self.assertEqual(len(labeled), 2)
        self.assertEqual(len(selected_ids), 2)
        self.assertTrue(all(sample["annotations"] for sample in labeled))
        self.assertTrue(all(sample.get("annotations") == [] for sample in unlabeled))
        self.assertTrue(all(item["role"] in {"labeled", "unlabeled"} for item in plan["sample_plans"]))

    def test_disabled_active_still_strips_unlabeled_annotations(self):
        target_train = [
            {"sample_id": f"target-{idx}", "annotations": [{"category_id": idx % 8}]}
            for idx in range(5)
        ]
        active_cfg = SimpleNamespace(enabled=False)

        labeled, unlabeled, selected_ids, plan = build_sparse_target_split(target_train, active_cfg, seed=42)

        self.assertEqual(labeled, [])
        self.assertEqual(selected_ids, set())
        self.assertEqual(len(unlabeled), 5)
        self.assertTrue(all(sample.get("annotations") == [] for sample in unlabeled))
        self.assertFalse(plan["enabled"])

    def test_fractional_budget_uses_at_least_one_when_nonzero(self):
        self.assertEqual(resolve_budget_count(0.05, 10), 1)
        self.assertEqual(resolve_budget_count(0.05, 100), 5)
        self.assertEqual(resolve_budget_count(3, 100), 3)


if __name__ == "__main__":
    unittest.main()

