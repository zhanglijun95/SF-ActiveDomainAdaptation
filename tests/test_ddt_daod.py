from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from baselines.ddt_daod.masking import apply_block_mask
from baselines.ddt_daod.pseudo import filter_pseudo_rows, update_dynamic_thresholds
from baselines.ddt_daod.trainer import _build_sparse_target_split, _effective_thresholds
from src.engine.daod_gradient_surgery import (
    target_anchored_cagrad,
    target_anchored_l2rw,
    target_anchored_pcgrad,
)
from src.engine.daod_teacher_guidance import merge_importance_maps


class DDTMaskingTests(unittest.TestCase):
    def test_apply_block_mask_preserves_shape(self):
        image = torch.ones((3, 32, 48), dtype=torch.float32)
        masked = apply_block_mask(image, block_size=8, masked_ratio=0.5)
        self.assertEqual(tuple(masked.shape), tuple(image.shape))
        self.assertLessEqual(float(masked.max()), 1.0)


class DDTPseudoTests(unittest.TestCase):
    def test_filter_pseudo_rows_uses_class_thresholds(self):
        rows = [
            {"category_id": 0, "score": 0.39, "bbox": [0, 0, 10, 10]},
            {"category_id": 0, "score": 0.41, "bbox": [0, 0, 10, 10]},
            {"category_id": 1, "score": 0.45, "bbox": [20, 20, 30, 30]},
        ]
        kept = filter_pseudo_rows(rows, thresholds=[0.4, 0.5], dedup_iou_thresh=0.7)
        self.assertEqual(len(kept), 1)
        self.assertEqual(int(kept[0]["category_id"]), 0)

    def test_update_dynamic_thresholds_clips_values(self):
        updated = update_dynamic_thresholds(
            [0.4, 0.4],
            [100.0, 0.0],
            [1, 0],
            alpha_dt=0.5,
            gamma_dt=0.9,
            max_dt=0.45,
            min_dt=0.25,
        )
        self.assertEqual(updated[0], 0.45)
        self.assertGreaterEqual(updated[1], 0.25)


class DDTActiveSparseLabelTests(unittest.TestCase):
    def test_random_split_is_deterministic_and_strips_unlabeled_annotations(self):
        target_train = [
            {"sample_id": f"sample-{idx}", "annotations": [{"category_id": idx % 2}]}
            for idx in range(10)
        ]
        active_cfg = SimpleNamespace(enabled=True, strategy="random", budget_total=0.2)

        labeled_a, unlabeled_a, selected_a, plan_a = _build_sparse_target_split(target_train, active_cfg, seed=42)
        labeled_b, unlabeled_b, selected_b, plan_b = _build_sparse_target_split(target_train, active_cfg, seed=42)

        self.assertEqual(plan_a["selected_ids"], plan_b["selected_ids"])
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(len(labeled_a), 2)
        self.assertEqual(len(unlabeled_a), 8)
        self.assertTrue(all(sample.get("annotations") == [] for sample in unlabeled_a))

    def test_effective_thresholds_apply_offsets_with_bounds(self):
        pseudo_cfg = SimpleNamespace(min_dt=0.25, max_dt=0.45)
        recalibration_cfg = SimpleNamespace(min_score_min=0.30)
        effective = _effective_thresholds(
            [0.40, 0.45, 0.32],
            [0.05, 0.20, 0.10],
            pseudo_cfg=pseudo_cfg,
            recalibration_cfg=recalibration_cfg,
            base_threshold=0.40,
        )

        for actual, expected in zip(effective, [0.35, 0.30, 0.30]):
            self.assertAlmostEqual(actual, expected)


class DDTLabelGuidedAEMATests(unittest.TestCase):
    def test_merge_importance_maps_can_elevate_gt_signal_without_suppressing_base(self):
        base = {
            "a": torch.tensor([1.0, 4.0]),
            "b": torch.tensor([2.0]),
        }
        guidance = {
            "a": torch.tensor([10.0, 1.0]),
            "b": torch.tensor([1.0]),
        }

        merged = merge_importance_maps(
            base,
            guidance,
            merge="max",
            guidance_weight=1.0,
            normalize=True,
        )

        self.assertGreater(float(merged["a"][0]), float(merged["b"][0]))
        self.assertGreater(float(merged["a"][1]), float(merged["b"][0]))
        self.assertEqual(set(merged), {"a", "b"})


class DDTGradientSurgeryTests(unittest.TestCase):
    def test_target_anchored_pcgrad_removes_negative_anchor_component(self):
        anchor = [torch.tensor([1.0, 0.0])]
        aux = [torch.tensor([-2.0, 3.0])]

        projected, stats = target_anchored_pcgrad(anchor_grads=anchor, aux_grads=aux)

        self.assertTrue(stats.projected)
        self.assertAlmostEqual(float(torch.dot(projected[0], anchor[0])), 0.0, places=6)
        self.assertLess(stats.cosine_before, 0.0)
        self.assertAlmostEqual(stats.cosine_after, 0.0, places=6)

    def test_target_anchored_pcgrad_keeps_aligned_gradient(self):
        anchor = [torch.tensor([1.0, 0.0])]
        aux = [torch.tensor([2.0, 3.0])]

        projected, stats = target_anchored_pcgrad(anchor_grads=anchor, aux_grads=aux)

        self.assertFalse(stats.projected)
        self.assertTrue(torch.equal(projected[0], aux[0]))
        self.assertGreater(stats.cosine_before, 0.0)
        self.assertEqual(stats.cosine_before, stats.cosine_after)

    def test_target_anchored_l2rw_downweights_conflicting_gradient(self):
        anchor = [torch.tensor([1.0, 0.0])]
        aux = [torch.tensor([-2.0, 3.0])]

        weighted, stats = target_anchored_l2rw(
            anchor_grads=anchor,
            aux_grads=aux,
            min_weight=0.25,
            max_weight=1.0,
        )

        self.assertTrue(stats.projected)
        self.assertAlmostEqual(stats.weight, 0.25)
        self.assertTrue(torch.allclose(weighted[0], 0.25 * aux[0]))

    def test_target_anchored_cagrad_returns_anchor_pseudo_direction(self):
        anchor = [torch.tensor([1.0, 0.0])]
        aux = [torch.tensor([-0.5, 1.0])]

        combined, stats = target_anchored_cagrad(
            anchor_grads=anchor,
            aux_grads=aux,
            c=0.4,
            rescale=1,
            sum_scale=True,
        )

        self.assertTrue(stats.projected)
        self.assertIsNotNone(stats.weight)
        self.assertEqual(tuple(combined[0].shape), tuple(anchor[0].shape))
        self.assertGreater(stats.cosine_after, stats.cosine_before)


if __name__ == "__main__":
    unittest.main()
