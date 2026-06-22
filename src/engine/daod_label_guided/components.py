"""Classify sparse-label-guided DAOD components for clean experiments.

This module intentionally has no training side effects. It gives every run a
paper-facing description of which sparse-label component is enabled, how it
fits the negative-results taxonomy, and whether the component is a stable
baseline/diagnostic path or an old prototype that should be reimplemented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BASELINE = "baseline"
SELECTION = "selection"
COMPLETION = "completion"
OPTIMIZATION_CONTROL = "optimization_control"
ORACLE_DIAGNOSTIC = "oracle_diagnostic"

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


@dataclass(frozen=True)
class LabelGuidedComponent:
    """Paper-facing metadata for one enabled or available component."""

    name: str
    category: str
    role: str
    status: str
    config_key: str
    enabled: bool
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_enabled(cfg: Any) -> bool:
    return bool(_cfg_get(cfg, "enabled", False))


def _cfg_present(parent: Any, name: str) -> bool:
    if isinstance(parent, dict):
        return name in parent
    return hasattr(parent, name)


def _append_if_present(
    components: list[LabelGuidedComponent],
    *,
    parent: Any,
    key: str,
    name: str,
    category: str,
    role: str,
    status: str,
    notes: str,
) -> None:
    if not _cfg_present(parent, key):
        return
    cfg = _cfg_get(parent, key, object())
    components.append(
        LabelGuidedComponent(
            name=name,
            category=category,
            role=role,
            status=status,
            config_key=key,
            enabled=_cfg_enabled(cfg),
            notes=notes,
        )
    )


def classify_label_guided_components(method_cfg: Any) -> list[LabelGuidedComponent]:
    """Return taxonomy metadata for sparse-label-related method config blocks."""

    components: list[LabelGuidedComponent] = []

    active_cfg = _cfg_get(method_cfg, "active", object())
    if _cfg_present(method_cfg, "active"):
        strategy = str(_cfg_get(active_cfg, "strategy", "random")).strip().lower()
        budget = _cfg_get(active_cfg, "budget_total", None)
        components.append(
            LabelGuidedComponent(
                name="random_supervised_target_loss",
                category=BASELINE,
                role="random_supervised_anchor",
                status="keep_live",
                config_key="active",
                enabled=_cfg_enabled(active_cfg),
                notes=f"Direct supervised target loss from {strategy} sparse labels; budget={budget}.",
            )
        )

    label_guided_cfg = _cfg_get(method_cfg, "label_guided", object())
    if _cfg_present(method_cfg, "label_guided"):
        method_name = str(_cfg_get(label_guided_cfg, "method", "unknown")).strip().lower()
        if method_name in THRESHOLD_CALIBRATION_METHODS:
            components.append(
                LabelGuidedComponent(
                    name="score_threshold_calibration",
                    category=SELECTION,
                    role="realistic_method",
                    status="keep_live",
                    config_key="label_guided",
                    enabled=_cfg_enabled(label_guided_cfg),
                    notes=(
                        "Clean selection representative: sparse target labels fit conservative "
                        "classwise pseudo-label score-threshold offsets."
                    ),
                )
            )
        elif method_name in THRESHOLD_MAPPING_METHODS:
            components.append(
                LabelGuidedComponent(
                    name="threshold_mapping",
                    category=SELECTION,
                    role="realistic_method",
                    status="keep_live",
                    config_key="label_guided",
                    enabled=_cfg_enabled(label_guided_cfg),
                    notes=(
                        "Clean selection representative: sparse target class priors "
                        "fit bounded classwise pseudo-label threshold offsets."
                    ),
                )
            )
        elif method_name in SCORE_REWEIGHT_METHODS:
            components.append(
                LabelGuidedComponent(
                    name="pseudo_score_reweight",
                    category=SELECTION,
                    role="realistic_method",
                    status="keep_live",
                    config_key="label_guided",
                    enabled=_cfg_enabled(label_guided_cfg),
                    notes=(
                        "Clean selection representative: sparse target labels estimate "
                        "classwise pseudo-label reliability and reweight pseudo scores."
                    ),
                )
            )
        elif method_name in LOSS_BALANCE_METHODS:
            components.append(
                LabelGuidedComponent(
                    name="sparse_loss_balance",
                    category=OPTIMIZATION_CONTROL,
                    role="realistic_method",
                    status="keep_live",
                    config_key="label_guided",
                    enabled=_cfg_enabled(label_guided_cfg),
                    notes=(
                        "Clean optimization-control representative: sparse supervised target "
                        "loss anchors conservative pseudo/masked loss scaling."
                    ),
                )
            )
        else:
            components.append(
                LabelGuidedComponent(
                    name=f"label_guided_{method_name or 'unknown'}",
                    category=SELECTION,
                    role="realistic_method",
                    status="planned_reimplement",
                    config_key="label_guided",
                    enabled=_cfg_enabled(label_guided_cfg),
                    notes="Generic label_guided block is present but not yet mapped to a clean plugin.",
                )
            )

    _append_if_present(
        components,
        parent=method_cfg,
        key="oracle_pseudo",
        name="oracle_pseudo_intervention",
        category=ORACLE_DIAGNOSTIC,
        role="oracle_upper_bound",
        status="keep_diagnostic",
        notes="Uses hidden target GT for filtering/recovery diagnostics; never a deployable method.",
    )

    if _cfg_present(method_cfg, "query_recovery"):
        cfg = _cfg_get(method_cfg, "query_recovery", object())
        train_as = str(_cfg_get(cfg, "train_as", "hard_pseudo")).strip().lower()
        component_name = "query_revival" if train_as == "revival_loss" else "query_recovery"
        components.append(
            LabelGuidedComponent(
                name=component_name,
                category=COMPLETION,
                role="realistic_method",
                status="keep_live",
                config_key="query_recovery",
                enabled=_cfg_enabled(cfg),
                notes=(
                    "Clean DDT-specific completion representative fitted from sparse labels."
                ),
            )
        )

    _append_if_present(
        components,
        parent=method_cfg,
        key="soft_query_activation",
        name="soft_query_activation",
        category=COMPLETION,
        role="realistic_method",
        status="reimplement_clean",
        notes=(
            "Historical low-confidence query activation/revival prototype. "
            "Collapse variants into one clean representative completion plugin."
        ),
    )

    train_cfg = _cfg_get(method_cfg, "train", object())
    _append_if_present(
        components,
        parent=train_cfg,
        key="pseudo_score_calibration",
        name="pseudo_score_calibration",
        category=SELECTION,
        role="realistic_method",
        status="reimplement_clean",
        notes="Historical pseudo-label selection/calibration branch; reimplement as a small representative plugin.",
    )
    _append_if_present(
        components,
        parent=train_cfg,
        key="pseudo_recalibration",
        name="pseudo_recalibration",
        category=SELECTION,
        role="realistic_method",
        status="archive_reference",
        notes="Older threshold/remapping branch; keep as reference unless selected for clean reimplementation.",
    )

    _append_if_present(
        components,
        parent=method_cfg,
        key="gradient_surgery",
        name="target_anchored_gradient_surgery",
        category=OPTIMIZATION_CONTROL,
        role="realistic_method",
        status="keep_live",
        notes="Clean restored optimization/control method from the historical target-anchored PCGrad/CAGrad/L2RW branch.",
    )
    _append_if_present(
        components,
        parent=method_cfg,
        key="label_guided_aema",
        name="label_guided_teacher_update",
        category=OPTIMIZATION_CONTROL,
        role="realistic_method",
        status="keep_live",
        notes="Clean restored optimization/control method from the historical guide-teacher-update/label-guided-AEMA branch.",
    )

    for key, name in (
        ("mtl_optimization", "mtl_optimization"),
        ("teacher_guidance", "teacher_guidance"),
        ("pseudo_bias", "pseudo_bias_correction"),
    ):
        _append_if_present(
            components,
            parent=method_cfg,
            key=key,
            name=name,
            category=OPTIMIZATION_CONTROL,
            role="realistic_method",
            status="reimplement_clean",
            notes="Historical optimization/control branch; keep only representative clean variants.",
        )

    aema_cfg = _cfg_get(method_cfg, "aema", object())
    if bool(_cfg_get(aema_cfg, "use_label_guidance", False)):
        components.append(
            LabelGuidedComponent(
                name="label_guided_teacher_update",
                category=OPTIMIZATION_CONTROL,
                role="realistic_method",
                status="keep_live",
                config_key="aema.use_label_guidance",
                enabled=True,
                notes="Sparse-label teacher-update control is treated as the restored label-guided-AEMA method.",
            )
        )

    return components


def summarize_label_guided_components(method_cfg: Any) -> dict[str, Any]:
    """Return a serializable summary for run metadata and audits."""

    components = classify_label_guided_components(method_cfg)
    enabled_components = [component for component in components if component.enabled]
    categories = sorted({component.category for component in enabled_components})
    legacy_components = [
        component
        for component in enabled_components
        if component.status in {"archive_reference", "reimplement_clean"}
    ]
    return {
        "components": [component.as_dict() for component in components],
        "enabled_components": [component.as_dict() for component in enabled_components],
        "enabled_component_names": [component.name for component in enabled_components],
        "enabled_categories": categories,
        "has_label_guided_enhancement": any(
            component.category != BASELINE for component in enabled_components
        ),
        "has_oracle_diagnostic": any(
            component.category == ORACLE_DIAGNOSTIC for component in enabled_components
        ),
        "has_legacy_live_prototype": bool(legacy_components),
        "legacy_live_component_names": [component.name for component in legacy_components],
        "cleanup_note": (
            "Baseline/random/oracle paths are stable; enabled legacy prototypes "
            "should be rerun through clean representative plugins before final paper use."
            if legacy_components
            else "No legacy label-guided prototype is enabled."
        ),
    }
