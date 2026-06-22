"""Label-guided hook interface for clean DAOD plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .components import summarize_label_guided_components
from .selection import (
    apply_class_score_weights,
    apply_threshold_offsets,
    fit_label_prior_threshold_mapping,
    fit_pseudo_score_reweight,
    fit_score_threshold_calibration,
)


THRESHOLD_CALIBRATION_METHODS = {
    "score_threshold_calibration",
    "threshold_calibration",
    "classwise_threshold_calibration",
}
THRESHOLD_MAPPING_METHODS = {
    "threshold_mapping",
    "label_prior_threshold_mapping",
    "class_distribution_threshold_mapping",
}
SCORE_REWEIGHT_METHODS = {
    "pseudo_score_reweight",
    "pseudo_loss_reweight",
    "score_reweight",
    "reliability_score_reweight",
}
LOSS_BALANCE_METHODS = {
    "sparse_loss_balance",
    "loss_balance",
    "supervised_pseudo_loss_balance",
}


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _label_guided_cfg(method_cfg: Any) -> Any:
    return _cfg_get(method_cfg, "label_guided", object())


def _label_guided_method(method_cfg: Any) -> str:
    label_cfg = _label_guided_cfg(method_cfg)
    return str(_cfg_get(label_cfg, "method", "")).strip().lower()


def label_guided_hook_requires_teacher_fit(method_cfg: Any) -> bool:
    """Return whether the enabled hook needs teacher outputs on sparse labels."""

    label_cfg = _label_guided_cfg(method_cfg)
    return bool(_cfg_get(label_cfg, "enabled", False)) and _label_guided_method(method_cfg) in (
        THRESHOLD_CALIBRATION_METHODS | THRESHOLD_MAPPING_METHODS | SCORE_REWEIGHT_METHODS
    )


def _method_specific_cfg(method_cfg: Any, method_name: str) -> Any:
    label_cfg = _label_guided_cfg(method_cfg)
    nested = _cfg_get(label_cfg, method_name, None)
    return nested if nested is not None else label_cfg


@dataclass
class LabelGuidedHookState:
    """Serializable hook state saved with a run."""

    component_summary: dict[str, Any]
    step_stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_summary": self.component_summary,
            "step_stats": self.step_stats,
        }


class LabelGuidedHook(Protocol):
    """Minimal interface clean sparse-label plugins should implement."""

    @property
    def state(self) -> LabelGuidedHookState:
        """Current serializable state."""

    def before_pseudo_filter(
        self,
        teacher_items: list[dict[str, Any]],
        *,
        thresholds: list[float],
        global_step: int,
    ) -> list[dict[str, Any]]:
        """Optionally modify teacher query rows before baseline pseudo filtering."""

    def after_pseudo_filter(
        self,
        *,
        sample: dict[str, Any],
        pseudo_rows: list[dict[str, Any]],
        threshold_rows: list[dict[str, Any]],
        global_step: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Optionally modify pseudo rows after baseline filtering."""

    def extra_loss_terms(self, *, global_step: int) -> tuple[list[Any], dict[str, Any]]:
        """Return optional loss terms and log fields."""

    def adjust_thresholds(self, thresholds: list[float], *, global_step: int) -> list[float]:
        """Optionally adjust per-class pseudo-label thresholds."""

    def loss_scales(self, raw_losses: dict[str, float], *, global_step: int) -> dict[str, float]:
        """Optionally scale loss branches before summing."""


class NoOpLabelGuidedHook:
    """Default hook used by baseline and oracle-only runs."""

    def __init__(self, method_cfg: Any) -> None:
        self._state = LabelGuidedHookState(
            component_summary=summarize_label_guided_components(method_cfg),
            step_stats={},
        )

    @property
    def state(self) -> LabelGuidedHookState:
        return self._state

    def before_pseudo_filter(
        self,
        teacher_items: list[dict[str, Any]],
        *,
        thresholds: list[float],
        global_step: int,
    ) -> list[dict[str, Any]]:
        return teacher_items

    def after_pseudo_filter(
        self,
        *,
        sample: dict[str, Any],
        pseudo_rows: list[dict[str, Any]],
        threshold_rows: list[dict[str, Any]],
        global_step: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return pseudo_rows, threshold_rows

    def extra_loss_terms(self, *, global_step: int) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def adjust_thresholds(self, thresholds: list[float], *, global_step: int) -> list[float]:
        return [float(value) for value in thresholds]

    def loss_scales(self, raw_losses: dict[str, float], *, global_step: int) -> dict[str, float]:
        return {"pseudo": 1.0, "masked": 1.0, "supervised": 1.0}


class ScoreThresholdCalibrationHook:
    """Classwise threshold-offset hook fitted from sparse target labels."""

    def __init__(
        self,
        method_cfg: Any,
        *,
        fit_teacher_items: list[dict[str, Any]] | None,
        base_thresholds: list[float] | None,
        num_classes: int | None,
    ) -> None:
        component_summary = summarize_label_guided_components(method_cfg)
        if fit_teacher_items is None or base_thresholds is None or num_classes is None:
            fit_result = {
                "enabled": True,
                "method": "score_threshold_calibration",
                "fit_images": 0,
                "reason": "not_fitted_yet",
                "offsets": [],
            }
        elif not fit_teacher_items:
            fit_result = {
                "enabled": True,
                "method": "score_threshold_calibration",
                "fit_images": 0,
                "reason": "no_labeled_target_images",
                "base_thresholds": [float(value) for value in base_thresholds],
                "calibrated_thresholds": [float(value) for value in base_thresholds],
                "offsets": [0.0 for _ in base_thresholds],
            }
        else:
            fit_result = fit_score_threshold_calibration(
                method_cfg,
                teacher_items=fit_teacher_items,
                base_thresholds=base_thresholds,
                num_classes=int(num_classes),
            )
        self._state = LabelGuidedHookState(
            component_summary=component_summary,
            step_stats={"score_threshold_calibration": fit_result},
        )

    @property
    def state(self) -> LabelGuidedHookState:
        return self._state

    def before_pseudo_filter(
        self,
        teacher_items: list[dict[str, Any]],
        *,
        thresholds: list[float],
        global_step: int,
    ) -> list[dict[str, Any]]:
        return teacher_items

    def after_pseudo_filter(
        self,
        *,
        sample: dict[str, Any],
        pseudo_rows: list[dict[str, Any]],
        threshold_rows: list[dict[str, Any]],
        global_step: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return pseudo_rows, threshold_rows

    def extra_loss_terms(self, *, global_step: int) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def adjust_thresholds(self, thresholds: list[float], *, global_step: int) -> list[float]:
        fit_result = self._state.step_stats.get("score_threshold_calibration", {})
        return apply_threshold_offsets([float(value) for value in thresholds], fit_result)

    def loss_scales(self, raw_losses: dict[str, float], *, global_step: int) -> dict[str, float]:
        return {"pseudo": 1.0, "masked": 1.0, "supervised": 1.0}


class ThresholdMappingHook(ScoreThresholdCalibrationHook):
    """Class-prior threshold mapping fitted from sparse target labels."""

    def __init__(
        self,
        method_cfg: Any,
        *,
        fit_teacher_items: list[dict[str, Any]] | None,
        base_thresholds: list[float] | None,
        num_classes: int | None,
    ) -> None:
        component_summary = summarize_label_guided_components(method_cfg)
        if fit_teacher_items is None or base_thresholds is None or num_classes is None:
            fit_result = {
                "enabled": True,
                "method": "threshold_mapping",
                "fit_images": 0,
                "reason": "not_fitted_yet",
                "offsets": [],
            }
        elif not fit_teacher_items:
            fit_result = {
                "enabled": True,
                "method": "threshold_mapping",
                "fit_images": 0,
                "reason": "no_labeled_target_images",
                "base_thresholds": [float(value) for value in base_thresholds],
                "calibrated_thresholds": [float(value) for value in base_thresholds],
                "offsets": [0.0 for _ in base_thresholds],
            }
        else:
            fit_result = fit_label_prior_threshold_mapping(
                method_cfg,
                teacher_items=fit_teacher_items,
                base_thresholds=base_thresholds,
                num_classes=int(num_classes),
            )
        self._state = LabelGuidedHookState(
            component_summary=component_summary,
            step_stats={"threshold_mapping": fit_result},
        )

    def adjust_thresholds(self, thresholds: list[float], *, global_step: int) -> list[float]:
        fit_result = self._state.step_stats.get("threshold_mapping", {})
        return apply_threshold_offsets([float(value) for value in thresholds], fit_result)


class PseudoScoreReweightHook(ScoreThresholdCalibrationHook):
    """Classwise score-weight hook fitted from sparse target pseudo precision."""

    def __init__(
        self,
        method_cfg: Any,
        *,
        fit_teacher_items: list[dict[str, Any]] | None,
        base_thresholds: list[float] | None,
        num_classes: int | None,
    ) -> None:
        component_summary = summarize_label_guided_components(method_cfg)
        if fit_teacher_items is None or base_thresholds is None or num_classes is None:
            fit_result = {
                "enabled": True,
                "method": "pseudo_score_reweight",
                "fit_images": 0,
                "reason": "not_fitted_yet",
                "class_weights": [],
            }
        elif not fit_teacher_items:
            fit_result = {
                "enabled": True,
                "method": "pseudo_score_reweight",
                "fit_images": 0,
                "reason": "no_labeled_target_images",
                "class_weights": [1.0 for _ in base_thresholds],
            }
        else:
            fit_result = fit_pseudo_score_reweight(
                method_cfg,
                teacher_items=fit_teacher_items,
                base_thresholds=base_thresholds,
                num_classes=int(num_classes),
            )
        self._state = LabelGuidedHookState(
            component_summary=component_summary,
            step_stats={"pseudo_score_reweight": fit_result},
        )

    def before_pseudo_filter(
        self,
        teacher_items: list[dict[str, Any]],
        *,
        thresholds: list[float],
        global_step: int,
    ) -> list[dict[str, Any]]:
        fit_result = self._state.step_stats.get("pseudo_score_reweight", {})
        return apply_class_score_weights(teacher_items, fit_result)

    def adjust_thresholds(self, thresholds: list[float], *, global_step: int) -> list[float]:
        return [float(value) for value in thresholds]


class SparseLossBalanceHook:
    """Scale pseudo branches using sparse supervised loss as a stable anchor."""

    def __init__(self, method_cfg: Any) -> None:
        self._method_cfg = method_cfg
        self._cfg = _method_specific_cfg(method_cfg, "sparse_loss_balance")
        self._ema: dict[str, float] = {}
        self._state = LabelGuidedHookState(
            component_summary=summarize_label_guided_components(method_cfg),
            step_stats={
                "sparse_loss_balance": {
                    "enabled": True,
                    "method": "sparse_loss_balance",
                    "updates": 0,
                    "ema": {},
                    "last_raw_losses": {},
                    "last_scales": {"pseudo": 1.0, "masked": 1.0, "supervised": 1.0},
                }
            },
        )

    @property
    def state(self) -> LabelGuidedHookState:
        return self._state

    def before_pseudo_filter(
        self,
        teacher_items: list[dict[str, Any]],
        *,
        thresholds: list[float],
        global_step: int,
    ) -> list[dict[str, Any]]:
        return teacher_items

    def after_pseudo_filter(
        self,
        *,
        sample: dict[str, Any],
        pseudo_rows: list[dict[str, Any]],
        threshold_rows: list[dict[str, Any]],
        global_step: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return pseudo_rows, threshold_rows

    def extra_loss_terms(self, *, global_step: int) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def adjust_thresholds(self, thresholds: list[float], *, global_step: int) -> list[float]:
        return [float(value) for value in thresholds]

    def loss_scales(self, raw_losses: dict[str, float], *, global_step: int) -> dict[str, float]:
        warmup_steps = int(_cfg_get(self._cfg, "warmup_steps", 0))
        momentum = float(_cfg_get(self._cfg, "ema_momentum", 0.95))
        alpha = float(_cfg_get(self._cfg, "alpha", 0.5))
        target_ratio = float(_cfg_get(self._cfg, "target_ratio", 1.0))
        min_pseudo_scale = float(_cfg_get(self._cfg, "min_pseudo_scale", 0.5))
        max_pseudo_scale = float(_cfg_get(self._cfg, "max_pseudo_scale", 1.5))
        apply_to_masked = bool(_cfg_get(self._cfg, "apply_to_masked", True))
        eps = float(_cfg_get(self._cfg, "eps", 1e-6))

        observed = {
            key: max(float(raw_losses.get(key, 0.0)), 0.0)
            for key in ("pseudo", "masked", "supervised")
        }
        for key, value in observed.items():
            if key not in self._ema:
                self._ema[key] = float(value)
            else:
                self._ema[key] = float(momentum) * self._ema[key] + (1.0 - float(momentum)) * float(value)

        pseudo_anchor = self._ema.get("pseudo", 0.0)
        if apply_to_masked:
            pseudo_anchor += self._ema.get("masked", 0.0)
        supervised_anchor = self._ema.get("supervised", 0.0)

        if int(global_step) < warmup_steps or supervised_anchor <= eps or pseudo_anchor <= eps:
            pseudo_scale = 1.0
        else:
            ratio = float(target_ratio) * float(supervised_anchor) / max(float(pseudo_anchor), eps)
            pseudo_scale = float(ratio) ** float(alpha)
            pseudo_scale = float(min(max(pseudo_scale, min_pseudo_scale), max_pseudo_scale))

        scales = {
            "pseudo": float(pseudo_scale),
            "masked": float(pseudo_scale if apply_to_masked else 1.0),
            "supervised": 1.0,
        }
        stats = self._state.step_stats["sparse_loss_balance"]
        stats["updates"] = int(stats.get("updates", 0)) + 1
        stats["ema"] = {key: float(value) for key, value in self._ema.items()}
        stats["last_raw_losses"] = observed
        stats["last_scales"] = scales
        stats["config"] = {
            "warmup_steps": int(warmup_steps),
            "ema_momentum": float(momentum),
            "alpha": float(alpha),
            "target_ratio": float(target_ratio),
            "min_pseudo_scale": float(min_pseudo_scale),
            "max_pseudo_scale": float(max_pseudo_scale),
            "apply_to_masked": bool(apply_to_masked),
        }
        return scales


def build_label_guided_hook(
    method_cfg: Any,
    *,
    fit_teacher_items: list[dict[str, Any]] | None = None,
    base_thresholds: list[float] | None = None,
    num_classes: int | None = None,
) -> LabelGuidedHook:
    """Build a label-guided hook.

    Unsupported or disabled label-guided configs intentionally fall back to the
    no-op hook so baseline behavior remains unchanged.
    """

    method_name = _label_guided_method(method_cfg)
    if bool(_cfg_get(_label_guided_cfg(method_cfg), "enabled", False)) and method_name in THRESHOLD_CALIBRATION_METHODS:
        return ScoreThresholdCalibrationHook(
            method_cfg,
            fit_teacher_items=fit_teacher_items,
            base_thresholds=base_thresholds,
            num_classes=num_classes,
        )
    if bool(_cfg_get(_label_guided_cfg(method_cfg), "enabled", False)) and method_name in THRESHOLD_MAPPING_METHODS:
        return ThresholdMappingHook(
            method_cfg,
            fit_teacher_items=fit_teacher_items,
            base_thresholds=base_thresholds,
            num_classes=num_classes,
        )
    if bool(_cfg_get(_label_guided_cfg(method_cfg), "enabled", False)) and method_name in SCORE_REWEIGHT_METHODS:
        return PseudoScoreReweightHook(
            method_cfg,
            fit_teacher_items=fit_teacher_items,
            base_thresholds=base_thresholds,
            num_classes=num_classes,
        )
    label_cfg = _label_guided_cfg(method_cfg)
    if bool(_cfg_get(label_cfg, "enabled", False)) and method_name in LOSS_BALANCE_METHODS:
        return SparseLossBalanceHook(method_cfg)
    return NoOpLabelGuidedHook(method_cfg)
