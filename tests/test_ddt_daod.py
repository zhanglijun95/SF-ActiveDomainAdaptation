from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from baselines.ddt_daod.masking import apply_block_mask
from baselines.ddt_daod.pseudo import filter_pseudo_rows, update_dynamic_thresholds
from baselines.ddt_daod.trainer import (
    _build_sparse_target_split,
    _effective_thresholds,
    _student_eval_patience_update,
)
from src.engine.daod_gradient_surgery import (
    target_anchored_cagrad,
    target_anchored_l2rw,
    target_anchored_pcgrad,
)
from src.engine.daod_latent_query_activation import LatentQueryActivator
from src.engine.daod_oracle_pseudo import ORACLE_ANNOTATION_KEY, apply_oracle_pseudo_intervention
from src.engine.daod_query_recovery import fit_query_recovery_scorer, merge_multiview_teacher_items
from src.engine.daod_query_revival import query_revival_loss
from src.engine.daod_soft_query_activation import fit_benefit_risk_class_gate, soft_query_activation_loss
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
        self.assertEqual(updated[1], 0.4)

    def test_update_dynamic_thresholds_keeps_classes_without_pseudo_evidence(self):
        updated = update_dynamic_thresholds(
            [0.4, 0.35],
            [0.0, 0.0],
            [0, 0],
            alpha_dt=0.5,
            gamma_dt=0.9,
            max_dt=0.45,
            min_dt=0.25,
        )
        self.assertEqual(updated, [0.4, 0.35])


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
        self.assertTrue(all(sample.get(ORACLE_ANNOTATION_KEY) for sample in unlabeled_a))

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


class DDTIntermediateEvalEarlyStopTests(unittest.TestCase):
    def test_consecutive_drop_stops_after_three_student_ap50_drops(self):
        previous_ap50 = None
        best_ap50 = None
        no_improve_count = 0
        drop_count = 0
        stopped = False

        for ap50 in [40.0, 39.5, 39.0, 38.5]:
            update = _student_eval_patience_update(
                ap50=ap50,
                previous_ap50=previous_ap50,
                best_ap50=best_ap50,
                no_improve_count=no_improve_count,
                consecutive_drop_count=drop_count,
                min_delta=0.0,
                mode="consecutive_drop",
                patience=3,
            )
            if update["improved"]:
                best_ap50 = ap50
            previous_ap50 = update["previous_ap50"]
            no_improve_count = update["no_improve_count"]
            drop_count = update["consecutive_drop_count"]
            stopped = update["should_stop"]

        self.assertTrue(stopped)
        self.assertEqual(drop_count, 3)
        self.assertAlmostEqual(best_ap50, 40.0)

    def test_consecutive_drop_resets_when_student_ap50_recovers(self):
        previous_ap50 = None
        best_ap50 = None
        no_improve_count = 0
        drop_count = 0

        for ap50 in [40.0, 39.5, 39.7, 39.2]:
            update = _student_eval_patience_update(
                ap50=ap50,
                previous_ap50=previous_ap50,
                best_ap50=best_ap50,
                no_improve_count=no_improve_count,
                consecutive_drop_count=drop_count,
                min_delta=0.0,
                mode="consecutive_drop",
                patience=3,
            )
            if update["improved"]:
                best_ap50 = ap50
            previous_ap50 = update["previous_ap50"]
            no_improve_count = update["no_improve_count"]
            drop_count = update["consecutive_drop_count"]

        self.assertFalse(update["should_stop"])
        self.assertEqual(drop_count, 1)
        self.assertAlmostEqual(best_ap50, 40.0)


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


class DDTOraclePseudoTests(unittest.TestCase):
    def test_oracle_filter_keeps_only_one_to_one_correct_pseudo_boxes(self):
        sample = {
            "sample_id": "target-0",
            ORACLE_ANNOTATION_KEY: [
                {"bbox": [0, 0, 10, 10], "category_id": 0},
                {"bbox": [50, 50, 70, 70], "category_id": 1},
            ],
        }
        rows = [
            {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9, "query_index": 0},
            {"bbox": [1, 1, 11, 11], "category_id": 0, "score": 0.8, "query_index": 1},
            {"bbox": [50, 50, 70, 70], "category_id": 0, "score": 0.7, "query_index": 2},
            {"bbox": [50, 50, 70, 70], "category_id": 1, "score": 0.6, "query_index": 3},
        ]

        result = apply_oracle_pseudo_intervention(
            sample=sample,
            pseudo_rows=rows,
            cfg=SimpleNamespace(enabled=True, mode="filter", match_iou=0.5),
            num_classes=2,
            class_names=("person", "rider"),
        )

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.stats.kept, 2)
        self.assertEqual(result.stats.dropped, 2)
        self.assertEqual(result.stats.recovered, 0)
        self.assertEqual([int(row["query_index"]) for row in result.threshold_rows], [0, 3])

    def test_oracle_recovery_adds_missed_gt_without_threshold_stats(self):
        sample = {
            "sample_id": "target-0",
            ORACLE_ANNOTATION_KEY: [
                {"bbox": [0, 0, 10, 10], "category_id": 0},
                {"bbox": [50, 50, 70, 70], "category_id": 1},
            ],
        }
        rows = [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9, "query_index": 0}]

        result = apply_oracle_pseudo_intervention(
            sample=sample,
            pseudo_rows=rows,
            cfg=SimpleNamespace(enabled=True, mode="recover", match_iou=0.5, recovery_score=0.95),
            num_classes=2,
            class_names=("person", "rider"),
        )

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(len(result.threshold_rows), 1)
        self.assertEqual(result.stats.kept, 1)
        self.assertEqual(result.stats.recovered, 1)
        recovered = [row for row in result.rows if row.get("_oracle_recovered")]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(int(recovered[0]["category_id"]), 1)
        self.assertAlmostEqual(float(recovered[0]["score"]), 0.95)

    def test_classwise_policy_can_filter_one_class_and_recover_another(self):
        sample = {
            "sample_id": "target-0",
            ORACLE_ANNOTATION_KEY: [
                {"bbox": [0, 0, 10, 10], "category_id": 0},
                {"bbox": [50, 50, 70, 70], "category_id": 1},
            ],
        }
        rows = [
            {"bbox": [80, 80, 90, 90], "category_id": 0, "score": 0.9, "query_index": 0},
            {"bbox": [0, 0, 10, 10], "category_id": 1, "score": 0.8, "query_index": 1},
        ]

        result = apply_oracle_pseudo_intervention(
            sample=sample,
            pseudo_rows=rows,
            cfg=SimpleNamespace(
                enabled=True,
                mode="classwise",
                match_iou=0.5,
                default_policy="none",
                policies={"person": "filter", "rider": "recover"},
            ),
            num_classes=2,
            class_names=("person", "rider"),
        )

        self.assertEqual(result.stats.dropped_by_class[0], 1)
        self.assertEqual(result.stats.recovered_by_class[1], 1)
        self.assertEqual(result.stats.kept_by_class[1], 1)
        self.assertEqual(result.stats.output_by_class[0], 0)
        self.assertEqual(result.stats.output_by_class[1], 2)


class DDTQueryRecoveryTests(unittest.TestCase):
    def test_recovery_scorer_selects_query_for_missed_gt_object(self):
        sample = {
            "sample_id": "target-0",
            "height": 100,
            "width": 100,
            "annotations": [
                {"bbox": [0, 0, 10, 10], "category_id": 0},
                {"bbox": [50, 50, 70, 70], "category_id": 1},
            ],
        }
        query_rows = [
            {
                "bbox": [0, 0, 10, 10],
                "category_id": 0,
                "score": 0.90,
                "query_index": 0,
                "softmax_margin": 0.8,
                "softmax_entropy": 0.1,
                "decoder_box_iou_gap": 0.0,
                "decoder_center_shift": 0.0,
            },
            {
                "bbox": [50, 50, 70, 70],
                "category_id": 1,
                "score": 0.20,
                "query_index": 1,
                "softmax_margin": 0.8,
                "softmax_entropy": 0.1,
                "decoder_box_iou_gap": 0.0,
                "decoder_center_shift": 0.0,
            },
            {
                "bbox": [80, 80, 90, 90],
                "category_id": 1,
                "score": 0.20,
                "query_index": 2,
                "softmax_margin": 0.1,
                "softmax_entropy": 0.9,
                "decoder_box_iou_gap": 1.0,
                "decoder_center_shift": 1.0,
            },
        ]
        cfg = SimpleNamespace(
            enabled=True,
            min_score=0.01,
            below_threshold_only=True,
            positive_iou=0.5,
            negative_iou=0.3,
            miss_iou=0.5,
            precision_floor=0.5,
            f_beta=2.0,
            min_class_positives=1,
            min_class_candidates=1,
            max_per_image=5,
            per_class_max=3,
            train_steps=60,
            lr=0.1,
            l2=0.0,
            max_negative_records=100,
            max_pos_weight=5.0,
            _resolved_num_views=1,
        )

        scorer = fit_query_recovery_scorer(
            [{"sample": sample, "query_rows": query_rows, "primary_query_rows": query_rows}],
            thresholds=[0.4, 0.4],
            num_classes=2,
            recovery_cfg=cfg,
            seed=42,
            dedup_iou_thresh=0.7,
        )
        selected, stats = scorer.select(
            query_rows,
            thresholds=[0.4, 0.4],
            dedup_iou_thresh=0.7,
            sample=sample,
            existing_rows=[query_rows[0]],
        )

        self.assertGreaterEqual(scorer.summary()["fit_positive"], 1)
        self.assertEqual(stats.selected, 1)
        self.assertEqual(int(selected[0]["query_index"]), 1)
        self.assertTrue(selected[0]["_query_recovery"])

    def test_multiview_merge_adds_support_features(self):
        sample = {"sample_id": "target-0", "height": 100, "width": 100, "annotations": []}
        primary = [
            {
                "sample": sample,
                "query_rows": [
                    {"bbox": [10, 10, 30, 30], "category_id": 0, "score": 0.2, "query_index": 0}
                ],
            }
        ]
        extra = [
            [
                {
                    "sample": sample,
                    "query_rows": [
                        {"bbox": [11, 11, 31, 31], "category_id": 0, "score": 0.3, "query_index": 0}
                    ],
                }
            ]
        ]

        merged = merge_multiview_teacher_items(primary, extra, support_iou=0.5)

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["query_rows"]), 2)
        self.assertEqual(merged[0]["num_views"], 2)
        self.assertTrue(all(row["_mv_support_views"] == 2 for row in merged[0]["query_rows"]))
        self.assertTrue(all(row["_mv_support_frac"] == 1.0 for row in merged[0]["query_rows"]))

    def test_recovery_risk_gate_disables_unsupported_class(self):
        sample = {
            "sample_id": "target-0",
            "height": 100,
            "width": 100,
            "annotations": [
                {"bbox": [0, 0, 10, 10], "category_id": 0},
                {"bbox": [50, 50, 70, 70], "category_id": 1},
            ],
        }
        query_rows = [
            {
                "bbox": [0, 0, 10, 10],
                "category_id": 0,
                "score": 0.90,
                "query_index": 0,
                "softmax_margin": 0.8,
                "softmax_entropy": 0.1,
                "decoder_box_iou_gap": 0.0,
                "decoder_center_shift": 0.0,
            },
            {
                "bbox": [50, 50, 70, 70],
                "category_id": 1,
                "score": 0.20,
                "query_index": 1,
                "softmax_margin": 0.8,
                "softmax_entropy": 0.1,
                "decoder_box_iou_gap": 0.0,
                "decoder_center_shift": 0.0,
            },
            {
                "bbox": [80, 80, 90, 90],
                "category_id": 1,
                "score": 0.20,
                "query_index": 2,
                "softmax_margin": 0.1,
                "softmax_entropy": 0.9,
                "decoder_box_iou_gap": 1.0,
                "decoder_center_shift": 1.0,
            },
        ]
        cfg = SimpleNamespace(
            enabled=True,
            min_score=0.01,
            below_threshold_only=True,
            positive_iou=0.5,
            negative_iou=0.3,
            miss_iou=0.5,
            precision_floor=0.5,
            f_beta=2.0,
            min_class_positives=1,
            min_class_candidates=1,
            max_per_image=5,
            per_class_max=3,
            train_steps=60,
            lr=0.1,
            l2=0.0,
            max_negative_records=100,
            max_pos_weight=5.0,
            _resolved_num_views=1,
            risk_gate=SimpleNamespace(
                enabled=True,
                min_precision=0.5,
                min_recall=0.01,
                min_total_positive=1,
                min_selected=1,
                precision_power=1.0,
                recall_power=0.5,
                normalize=True,
                gate_floor=0.0,
                gate_max=1.0,
                budget=SimpleNamespace(enabled=True, scale=0.25, min_budget=0.0, max_budget=1.0),
            ),
        )

        scorer = fit_query_recovery_scorer(
            [{"sample": sample, "query_rows": query_rows, "primary_query_rows": query_rows}],
            thresholds=[0.4, 0.4],
            num_classes=2,
            recovery_cfg=cfg,
            seed=42,
            dedup_iou_thresh=0.7,
        )

        summary = scorer.summary()
        self.assertEqual(summary["risk_gate"]["enabled_classes"], [1])
        self.assertEqual(summary["class_gates"][0], 0.0)
        self.assertGreater(summary["class_gates"][1], 0.0)
        self.assertIsNotNone(summary["class_budgets"])


class DDTSoftQueryActivationTests(unittest.TestCase):
    def test_class_bce_query_index_loss_pushes_selected_class_up(self):
        student_logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        soft_items = [
            {
                "sample": {"height": 100, "width": 100},
                "teacher_raw": {"pred_logits": torch.zeros((1, 3), dtype=torch.float32)},
                "student_raw": {"pred_logits": student_logits},
                "teacher_rows": [
                    {
                        "query_index": 0,
                        "category_id": 2,
                        "bbox": [10, 10, 30, 30],
                        "score": 0.20,
                        "_latent_activation_score": 0.80,
                    }
                ],
            }
        ]

        loss, stats = soft_query_activation_loss(
            soft_items,
            objective="class_bce",
            loss_weight=0.05,
            match_mode="query_index",
            min_match_iou=0.4,
            match_class_aware=False,
            positive_target=0.8,
            margin=0.3,
            activation_weight_power=1.0,
            min_activation_weight=0.25,
            distill_temperature=1.0,
            distill_negative_weight=0.2,
            distill_boost_selected=True,
        )
        loss.backward()

        self.assertEqual(stats.targets, 1)
        self.assertEqual(stats.matched, 1)
        self.assertLess(float(student_logits.grad[0, 2]), 0.0)

    def test_box_iou_match_uses_student_box_alignment(self):
        student_logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        student_boxes = torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=torch.float32)
        soft_items = [
            {
                "sample": {"height": 100, "width": 100},
                "teacher_raw": {"pred_logits": torch.zeros((1, 3), dtype=torch.float32)},
                "student_raw": {"pred_logits": student_logits, "pred_boxes": student_boxes},
                "teacher_rows": [
                    {
                        "query_index": 0,
                        "category_id": 2,
                        "bbox": [10, 10, 30, 30],
                        "score": 0.20,
                        "_latent_activation_score": 0.80,
                    }
                ],
            }
        ]

        loss, stats = soft_query_activation_loss(
            soft_items,
            objective="margin",
            loss_weight=0.05,
            match_mode="box_iou",
            min_match_iou=0.5,
            match_class_aware=False,
            positive_target=0.8,
            margin=0.3,
            activation_weight_power=1.0,
            min_activation_weight=0.25,
            distill_temperature=1.0,
            distill_negative_weight=0.2,
            distill_boost_selected=True,
        )
        loss.backward()

        self.assertEqual(stats.targets, 1)
        self.assertEqual(stats.matched, 1)
        self.assertGreater(stats.as_dict()["mean_match_iou"], 0.99)

    def test_benefit_risk_gate_prefers_recovered_missed_class(self):
        activator = LatentQueryActivator(
            method="precision_rule",
            num_classes=2,
            min_score=0.01,
            max_per_image=5,
            class_thresholds={0: 0.0, 1: 0.0},
            global_threshold=0.0,
            precision_target=0.95,
            positive_iou=0.5,
            negative_iou=0.3,
            quality_weights={
                "score": 1.0,
                "margin": 0.0,
                "confidence": 0.0,
                "box_stability": 0.0,
                "center_stability": 0.0,
            },
            summary={},
        )
        sample = {
            "sample_id": "target-0",
            "height": 100,
            "width": 100,
            "annotations": [
                {"bbox": [10, 10, 30, 30], "category_id": 0},
                {"bbox": [60, 60, 80, 80], "category_id": 1},
            ],
        }
        rows = [
            {
                "query_index": 0,
                "category_id": 0,
                "score": 0.20,
                "bbox": [10, 10, 30, 30],
                "softmax_margin": 0.9,
                "softmax_entropy": 0.1,
                "decoder_box_iou_gap": 0.0,
                "decoder_center_shift": 0.0,
            },
            {
                "query_index": 1,
                "category_id": 1,
                "score": 0.80,
                "bbox": [60, 60, 80, 80],
                "softmax_margin": 0.9,
                "softmax_entropy": 0.1,
                "decoder_box_iou_gap": 0.0,
                "decoder_center_shift": 0.0,
            },
            {
                "query_index": 2,
                "category_id": 1,
                "score": 0.20,
                "bbox": [2, 2, 15, 15],
                "softmax_margin": 0.9,
                "softmax_entropy": 0.1,
                "decoder_box_iou_gap": 0.0,
                "decoder_center_shift": 0.0,
            },
        ]
        cfg = SimpleNamespace(
            enabled=True,
            match_iou=0.5,
            confidence_z=1.0,
            normalize=True,
            gate_floor=0.0,
            gate_max=1.0,
            budget=SimpleNamespace(enabled=True, scale=2.0, need_power=0.5, min_budget=0.0),
        )

        gate = fit_benefit_risk_class_gate(
            [{"sample": sample, "query_rows": rows}],
            activator=activator,
            thresholds=[0.4, 0.4],
            num_classes=2,
            gate_cfg=cfg,
            dedup_iou_thresh=0.7,
        )

        self.assertGreater(gate.gates[0], gate.gates[1])
        self.assertGreater(gate.gates[0], 0.0)
        self.assertEqual(gate.gates[1], 0.0)
        self.assertIsNotNone(gate.budgets)

    def test_query_risk_weight_downweights_unstable_query(self):
        student_logits = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
        soft_items = [
            {
                "sample": {"height": 100, "width": 100},
                "teacher_raw": {"pred_logits": torch.zeros((2, 3), dtype=torch.float32)},
                "student_raw": {"pred_logits": student_logits},
                "teacher_rows": [
                    {
                        "query_index": 0,
                        "category_id": 1,
                        "bbox": [10, 10, 30, 30],
                        "score": 0.20,
                        "_latent_activation_score": 1.00,
                        "decoder_box_iou_gap": 0.0,
                    },
                    {
                        "query_index": 1,
                        "category_id": 1,
                        "bbox": [40, 40, 60, 60],
                        "score": 0.20,
                        "_latent_activation_score": 1.00,
                        "decoder_box_iou_gap": 1.0,
                    },
                ],
            }
        ]

        loss, stats = soft_query_activation_loss(
            soft_items,
            objective="class_bce",
            loss_weight=1.0,
            match_mode="query_index",
            min_match_iou=0.4,
            match_class_aware=False,
            positive_target=0.8,
            margin=0.3,
            activation_weight_power=1.0,
            min_activation_weight=0.25,
            distill_temperature=1.0,
            distill_negative_weight=0.2,
            distill_boost_selected=True,
            query_risk_cfg=SimpleNamespace(
                enabled=True,
                aggregate="weighted_mean",
                min_weight=0.5,
                max_weight=1.0,
                power=1.0,
                weights={
                    "activation": 0.0,
                    "score_proximity": 0.0,
                    "margin": 0.0,
                    "confidence": 0.0,
                    "box_stability": 1.0,
                    "center_stability": 0.0,
                },
            ),
        )
        loss.backward()

        self.assertEqual(stats.matched, 2)
        self.assertAlmostEqual(stats.weight_sum, 1.5)
        self.assertAlmostEqual(stats.as_dict()["mean_risk_weight"], 0.75)
        self.assertIsNotNone(student_logits.grad)


class DDTQueryRevivalTests(unittest.TestCase):
    def test_foreground_revival_loss_pushes_query_activation_up(self):
        student_logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        revival_items = [
            {
                "sample": {"height": 100, "width": 100},
                "student_raw": {"pred_logits": student_logits},
                "teacher_rows": [
                    {
                        "query_index": 0,
                        "category_id": 1,
                        "bbox": [10, 10, 30, 30],
                        "score": 0.20,
                        "_query_recovery_score": 0.90,
                        "_query_recovery_gate": 0.50,
                    }
                ],
            }
        ]

        loss, stats = query_revival_loss(
            revival_items,
            loss_weight=1.0,
            match_mode="query_index",
            min_match_iou=0.4,
            match_class_aware=False,
            positive_target=0.8,
            foreground_pool="mean_logsumexp",
            foreground_temperature=1.0,
            recovery_weight_power=1.0,
            min_candidate_weight=0.1,
            class_budgets={1: 0.2},
        )
        loss.backward()

        self.assertEqual(stats.targets, 1)
        self.assertEqual(stats.matched, 1)
        self.assertAlmostEqual(stats.weight_sum, 0.2)
        self.assertAlmostEqual(stats.as_dict()["mean_gate"], 0.5)
        self.assertLess(float(student_logits.grad.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
