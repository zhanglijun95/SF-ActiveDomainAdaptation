"""Shared label-guided DAOD experiment metadata helpers."""

from .components import (
    LabelGuidedComponent,
    classify_label_guided_components,
    summarize_label_guided_components,
)
from .hooks import (
    LabelGuidedHook,
    LabelGuidedHookState,
    NoOpLabelGuidedHook,
    build_label_guided_hook,
    label_guided_hook_requires_teacher_fit,
)

__all__ = [
    "LabelGuidedComponent",
    "LabelGuidedHook",
    "LabelGuidedHookState",
    "NoOpLabelGuidedHook",
    "build_label_guided_hook",
    "classify_label_guided_components",
    "label_guided_hook_requires_teacher_fit",
    "summarize_label_guided_components",
]
