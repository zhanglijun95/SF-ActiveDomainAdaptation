from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.engine.daod_label_guided import build_label_guided_hook, summarize_label_guided_components


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


class LabelGuidedComponentsTest(unittest.TestCase):
    def test_random_supervised_anchor_is_baseline_not_legacy(self):
        summary = summarize_label_guided_components(
            ns(active=ns(enabled=True, strategy="random", budget_total=0.05))
        )

        self.assertEqual(summary["enabled_component_names"], ["random_supervised_target_loss"])
        self.assertEqual(summary["enabled_categories"], ["baseline"])
        self.assertFalse(summary["has_legacy_live_prototype"])

    def test_oracle_is_diagnostic_not_legacy(self):
        summary = summarize_label_guided_components(
            ns(
                active=ns(enabled=True, strategy="random", budget_total=0.05),
                oracle_pseudo=ns(enabled=True, mode="recover"),
            )
        )

        self.assertTrue(summary["has_oracle_diagnostic"])
        self.assertFalse(summary["has_legacy_live_prototype"])
        self.assertIn("oracle_pseudo_intervention", summary["enabled_component_names"])

    def test_query_recovery_is_clean_completion_representative(self):
        summary = summarize_label_guided_components(
            ns(query_recovery=ns(enabled=True, train_as="hard_pseudo"))
        )

        self.assertFalse(summary["has_legacy_live_prototype"])
        self.assertEqual(summary["enabled_component_names"], ["query_recovery"])
        self.assertEqual(summary["legacy_live_component_names"], [])
        self.assertEqual(summary["enabled_categories"], ["completion"])

    def test_query_revival_name_follows_train_as(self):
        summary = summarize_label_guided_components(
            ns(query_recovery=ns(enabled=True, train_as="revival_loss"))
        )

        self.assertFalse(summary["has_legacy_live_prototype"])
        self.assertEqual(summary["enabled_component_names"], ["query_revival"])
        self.assertEqual(summary["legacy_live_component_names"], [])

    def test_selection_blocks_inside_train_are_classified(self):
        summary = summarize_label_guided_components(
            ns(train=ns(pseudo_score_calibration=ns(enabled=True)))
        )

        self.assertEqual(summary["enabled_categories"], ["selection"])
        self.assertEqual(summary["legacy_live_component_names"], ["pseudo_score_calibration"])

    def test_clean_label_guided_threshold_calibration_is_not_legacy(self):
        summary = summarize_label_guided_components(
            ns(label_guided=ns(enabled=True, method="score_threshold_calibration"))
        )

        self.assertEqual(summary["enabled_categories"], ["selection"])
        self.assertEqual(summary["enabled_component_names"], ["score_threshold_calibration"])
        self.assertFalse(summary["has_legacy_live_prototype"])

    def test_clean_label_guided_threshold_mapping_is_selection(self):
        summary = summarize_label_guided_components(
            ns(label_guided=ns(enabled=True, method="threshold_mapping"))
        )

        self.assertEqual(summary["enabled_categories"], ["selection"])
        self.assertEqual(summary["enabled_component_names"], ["threshold_mapping"])
        self.assertFalse(summary["has_legacy_live_prototype"])

    def test_clean_label_guided_score_reweight_is_selection(self):
        summary = summarize_label_guided_components(
            ns(label_guided=ns(enabled=True, method="pseudo_score_reweight"))
        )

        self.assertEqual(summary["enabled_categories"], ["selection"])
        self.assertEqual(summary["enabled_component_names"], ["pseudo_score_reweight"])
        self.assertFalse(summary["has_legacy_live_prototype"])

    def test_clean_label_guided_loss_balance_is_optimization_control(self):
        summary = summarize_label_guided_components(
            ns(label_guided=ns(enabled=True, method="sparse_loss_balance"))
        )

        self.assertEqual(summary["enabled_categories"], ["optimization_control"])
        self.assertEqual(summary["enabled_component_names"], ["sparse_loss_balance"])
        self.assertFalse(summary["has_legacy_live_prototype"])

    def test_restored_gradient_surgery_is_clean_optimization_control(self):
        summary = summarize_label_guided_components(
            ns(gradient_surgery=ns(enabled=True, method="target_anchored_pcgrad"))
        )

        self.assertEqual(summary["enabled_categories"], ["optimization_control"])
        self.assertEqual(summary["enabled_component_names"], ["target_anchored_gradient_surgery"])
        self.assertFalse(summary["has_legacy_live_prototype"])

    def test_restored_label_guided_aema_is_clean_optimization_control(self):
        summary = summarize_label_guided_components(
            ns(label_guided_aema=ns(enabled=True, merge="max"))
        )

        self.assertEqual(summary["enabled_categories"], ["optimization_control"])
        self.assertEqual(summary["enabled_component_names"], ["label_guided_teacher_update"])
        self.assertFalse(summary["has_legacy_live_prototype"])

    def test_noop_hook_preserves_rows_and_reports_state(self):
        hook = build_label_guided_hook(ns(active=ns(enabled=True, strategy="random", budget_total=0.05)))
        teacher_items = [{"query_rows": [{"score": 0.9}]}]
        pseudo_rows = [{"score": 0.9}]
        threshold_rows = [{"score": 0.9}]

        self.assertIs(hook.before_pseudo_filter(teacher_items, thresholds=[0.4], global_step=0), teacher_items)
        self.assertEqual(
            hook.after_pseudo_filter(
                sample={},
                pseudo_rows=pseudo_rows,
                threshold_rows=threshold_rows,
                global_step=0,
            ),
            (pseudo_rows, threshold_rows),
        )
        self.assertEqual(hook.extra_loss_terms(global_step=0), ([], {}))
        self.assertEqual(hook.adjust_thresholds([0.4], global_step=0), [0.4])
        self.assertIn("component_summary", hook.state.as_dict())

    def test_threshold_calibration_hook_fits_offsets_from_sparse_labels(self):
        method_cfg = ns(
            label_guided=ns(
                enabled=True,
                method="score_threshold_calibration",
                score_threshold_calibration=ns(
                    target_precision=0.75,
                    match_iou=0.5,
                    min_score=0.01,
                    min_selected=1,
                    min_positives=1,
                    min_threshold=0.25,
                    max_threshold=0.55,
                    max_delta_down=0.10,
                    max_delta_up=0.15,
                ),
            )
        )
        teacher_items = [
            {
                "sample": {
                    "annotations": [
                        {"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0},
                    ]
                },
                "query_rows": [
                    {"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0, "score": 0.50},
                    {"bbox": [20.0, 20.0, 30.0, 30.0], "category_id": 0, "score": 0.30},
                    {"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 1, "score": 0.90},
                ],
            }
        ]

        hook = build_label_guided_hook(
            method_cfg,
            fit_teacher_items=teacher_items,
            base_thresholds=[0.4, 0.4],
            num_classes=2,
        )

        state = hook.state.as_dict()["step_stats"]["score_threshold_calibration"]
        self.assertEqual(state["adjusted_classes"], [0])
        self.assertAlmostEqual(state["offsets"][0], 0.1)
        self.assertAlmostEqual(state["offsets"][1], 0.0)
        self.assertEqual(hook.adjust_thresholds([0.4, 0.4], global_step=10), [0.5, 0.4])

    def test_threshold_mapping_hook_moves_thresholds_by_sparse_prior(self):
        method_cfg = ns(
            label_guided=ns(
                enabled=True,
                method="threshold_mapping",
                threshold_mapping=ns(
                    min_score=0.01,
                    smoothing=1.0,
                    ratio_temperature=1.0,
                    min_threshold=0.25,
                    max_threshold=0.55,
                    max_delta_down=0.10,
                    max_delta_up=0.10,
                ),
            )
        )
        teacher_items = [
            {
                "sample": {
                    "annotations": [
                        {"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0},
                    ]
                },
                "query_rows": [
                    {"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0, "score": 0.30},
                    {"bbox": [20.0, 20.0, 30.0, 30.0], "category_id": 1, "score": 0.90},
                    {"bbox": [40.0, 40.0, 50.0, 50.0], "category_id": 1, "score": 0.80},
                ],
            }
        ]

        hook = build_label_guided_hook(
            method_cfg,
            fit_teacher_items=teacher_items,
            base_thresholds=[0.4, 0.4],
            num_classes=2,
        )

        adjusted = hook.adjust_thresholds([0.4, 0.4], global_step=10)
        state = hook.state.as_dict()["step_stats"]["threshold_mapping"]
        self.assertEqual(state["method"], "threshold_mapping")
        self.assertLess(adjusted[0], 0.4)
        self.assertGreater(adjusted[1], 0.4)
        self.assertEqual(state["adjusted_classes"], [0, 1])

    def test_pseudo_score_reweight_hook_reweights_scores_before_filtering(self):
        method_cfg = ns(
            label_guided=ns(
                enabled=True,
                method="pseudo_score_reweight",
                pseudo_score_reweight=ns(
                    match_iou=0.5,
                    min_score=0.01,
                    target_precision=0.75,
                    min_candidates=1,
                    min_positives=0,
                    min_weight=0.50,
                    max_weight=1.00,
                    power=1.0,
                ),
            )
        )
        teacher_items = [
            {
                "sample": {
                    "annotations": [
                        {"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0},
                    ]
                },
                "query_rows": [
                    {"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0, "score": 0.80},
                    {"bbox": [20.0, 20.0, 30.0, 30.0], "category_id": 0, "score": 0.70},
                    {"bbox": [40.0, 40.0, 50.0, 50.0], "category_id": 1, "score": 0.90},
                    {"bbox": [60.0, 60.0, 70.0, 70.0], "category_id": 1, "score": 0.80},
                ],
            }
        ]

        hook = build_label_guided_hook(
            method_cfg,
            fit_teacher_items=teacher_items,
            base_thresholds=[0.4, 0.4],
            num_classes=2,
        )
        weighted = hook.before_pseudo_filter(teacher_items, thresholds=[0.4, 0.4], global_step=10)
        state = hook.state.as_dict()["step_stats"]["pseudo_score_reweight"]

        self.assertEqual(state["method"], "pseudo_score_reweight")
        self.assertLess(weighted[0]["query_rows"][0]["score"], 0.80)
        self.assertLess(weighted[0]["query_rows"][2]["score"], weighted[0]["query_rows"][0]["score"])
        self.assertEqual(weighted[0]["query_rows"][2]["raw_score"], 0.90)

    def test_sparse_loss_balance_hook_scales_pseudo_with_supervised_anchor(self):
        method_cfg = ns(
            label_guided=ns(
                enabled=True,
                method="sparse_loss_balance",
                sparse_loss_balance=ns(
                    warmup_steps=0,
                    ema_momentum=0.0,
                    alpha=1.0,
                    target_ratio=1.0,
                    min_pseudo_scale=0.5,
                    max_pseudo_scale=1.5,
                    apply_to_masked=True,
                ),
            )
        )
        hook = build_label_guided_hook(method_cfg)

        scales = hook.loss_scales(
            {"pseudo": 4.0, "masked": 2.0, "supervised": 3.0},
            global_step=10,
        )

        self.assertEqual(scales["pseudo"], 0.5)
        self.assertEqual(scales["masked"], 0.5)
        self.assertEqual(scales["supervised"], 1.0)
        state = hook.state.as_dict()["step_stats"]["sparse_loss_balance"]
        self.assertEqual(state["updates"], 1)


if __name__ == "__main__":
    unittest.main()
