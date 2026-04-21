from __future__ import annotations

import unittest

import torch

from baselines.ddt_daod.masking import apply_block_mask
from baselines.ddt_daod.pseudo import filter_pseudo_rows, update_dynamic_thresholds


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


if __name__ == "__main__":
    unittest.main()
