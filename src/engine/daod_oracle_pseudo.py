"""Oracle-only pseudo-label interventions for DAOD diagnostics.

These helpers intentionally use hidden target GT. They are not meant to be a
deployable method; they measure whether noisy pseudo labels or missed objects
are real bottlenecks beyond sparse random target supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ORACLE_ANNOTATION_KEY = "_oracle_annotations"


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _as_policy(policy: Any) -> str:
    normalized = str(policy).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "keep": "none",
        "filter": "filter",
        "filter_only": "filter",
        "recover": "recover",
        "recovery": "recover",
        "recover_only": "recover",
        "filter_recover": "filter_recover",
        "filter+recover": "filter_recover",
        "filter_and_recover": "filter_recover",
        "both": "filter_recover",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported oracle pseudo policy "
            f"{policy!r}; use one of none/filter/recover/filter_recover"
        )
    return aliases[normalized]


def _policy_filters(policy: str) -> bool:
    return policy in {"filter", "filter_recover"}


def _policy_recovers(policy: str) -> bool:
    return policy in {"recover", "filter_recover"}


def _valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    x0, y0, x1, y1 = [float(value) for value in box]
    return x1 > x0 and y1 > y0


def _normalize_annotations(sample: dict[str, Any], *, num_classes: int) -> list[dict[str, Any]]:
    annotations = sample.get(ORACLE_ANNOTATION_KEY, sample.get("annotations", []))
    normalized: list[dict[str, Any]] = []
    for ann in annotations or []:
        class_id = int(ann.get("category_id", -1))
        if class_id < 0 or class_id >= int(num_classes):
            continue
        bbox = ann.get("bbox")
        if not _valid_box(bbox):
            continue
        normalized.append(
            {
                "bbox": [float(value) for value in bbox],
                "category_id": class_id,
            }
        )
    return normalized


def _normalize_rows(rows: list[dict[str, Any]], *, num_classes: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        class_id = int(row.get("category_id", -1))
        if class_id < 0 or class_id >= int(num_classes):
            continue
        bbox = row.get("bbox")
        if not _valid_box(bbox):
            continue
        normalized.append(dict(row))
    return normalized


def _class_name_to_id(class_names: list[str] | tuple[str, ...]) -> dict[str, int]:
    return {str(name).strip().lower(): idx for idx, name in enumerate(class_names)}


def _resolve_class_id(value: Any, *, class_names: list[str] | tuple[str, ...]) -> int:
    if isinstance(value, int):
        return int(value)
    value_str = str(value).strip()
    if value_str.lstrip("-").isdigit():
        return int(value_str)
    name_to_id = _class_name_to_id(class_names)
    key = value_str.lower()
    if key not in name_to_id:
        raise ValueError(f"Unknown DAOD class {value!r}; known classes={list(class_names)}")
    return int(name_to_id[key])


def _resolve_class_set(
    values: Any,
    *,
    num_classes: int,
    class_names: list[str] | tuple[str, ...],
) -> set[int] | None:
    if values is None:
        return None
    if isinstance(values, str) and values.strip().lower() in {"all", "*"}:
        return None
    if isinstance(values, (str, int)):
        values = [values]
    resolved: set[int] = set()
    for value in values:
        class_id = _resolve_class_id(value, class_names=class_names)
        if 0 <= class_id < int(num_classes):
            resolved.add(class_id)
    return resolved


def _policy_map_from_cfg(
    cfg: Any,
    *,
    num_classes: int,
    class_names: list[str] | tuple[str, ...],
) -> list[str]:
    mode = str(_cfg_get(cfg, "mode", "filter")).strip().lower().replace("-", "_")
    selected_classes = _resolve_class_set(
        _cfg_get(cfg, "classes", None),
        num_classes=num_classes,
        class_names=class_names,
    )

    if mode == "filter":
        policies = ["filter"] * int(num_classes)
    elif mode in {"recover", "recovery"}:
        policies = ["recover"] * int(num_classes)
    elif mode in {"filter_recover", "filter+recover", "both"}:
        policies = ["filter_recover"] * int(num_classes)
    elif mode == "classwise":
        default_policy = _as_policy(_cfg_get(cfg, "default_policy", _cfg_get(cfg, "default", "none")))
        policies = [default_policy] * int(num_classes)
        policy_cfg = _cfg_get(cfg, "policies", {})
        for class_key, policy_value in dict(policy_cfg).items():
            class_id = _resolve_class_id(class_key, class_names=class_names)
            if 0 <= class_id < int(num_classes):
                policies[class_id] = _as_policy(policy_value)
    else:
        raise ValueError(
            "method.oracle_pseudo.mode must be filter, recover, filter_recover, or classwise; "
            f"got {mode!r}"
        )

    if selected_classes is not None:
        policies = [policy if idx in selected_classes else "none" for idx, policy in enumerate(policies)]
    return policies


@dataclass
class OraclePseudoResult:
    rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    stats: "OraclePseudoStats"


class OraclePseudoStats:
    """Small aggregate used for step/epoch logs."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.images = 0
        self.gt = 0
        self.input_pseudo = 0
        self.kept = 0
        self.dropped = 0
        self.recovered = 0
        self.output_pseudo = 0
        self.gt_by_class = [0] * self.num_classes
        self.input_by_class = [0] * self.num_classes
        self.kept_by_class = [0] * self.num_classes
        self.dropped_by_class = [0] * self.num_classes
        self.recovered_by_class = [0] * self.num_classes
        self.output_by_class = [0] * self.num_classes
        self.missed_before_recovery_by_class = [0] * self.num_classes

    def add(self, other: "OraclePseudoStats") -> None:
        if self.num_classes != other.num_classes:
            raise ValueError("Cannot add oracle pseudo stats with different class counts")
        self.images += other.images
        self.gt += other.gt
        self.input_pseudo += other.input_pseudo
        self.kept += other.kept
        self.dropped += other.dropped
        self.recovered += other.recovered
        self.output_pseudo += other.output_pseudo
        for idx in range(self.num_classes):
            self.gt_by_class[idx] += other.gt_by_class[idx]
            self.input_by_class[idx] += other.input_by_class[idx]
            self.kept_by_class[idx] += other.kept_by_class[idx]
            self.dropped_by_class[idx] += other.dropped_by_class[idx]
            self.recovered_by_class[idx] += other.recovered_by_class[idx]
            self.output_by_class[idx] += other.output_by_class[idx]
            self.missed_before_recovery_by_class[idx] += other.missed_before_recovery_by_class[idx]

    def as_dict(self, *, class_names: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        per_class: dict[str, dict[str, int]] = {}
        for class_id in range(self.num_classes):
            label = str(class_names[class_id]) if class_id < len(class_names) else str(class_id)
            per_class[label] = {
                "gt": int(self.gt_by_class[class_id]),
                "input_pseudo": int(self.input_by_class[class_id]),
                "kept": int(self.kept_by_class[class_id]),
                "dropped": int(self.dropped_by_class[class_id]),
                "recovered": int(self.recovered_by_class[class_id]),
                "output_pseudo": int(self.output_by_class[class_id]),
                "missed_before_recovery": int(self.missed_before_recovery_by_class[class_id]),
            }
        return {
            "images": int(self.images),
            "gt": int(self.gt),
            "input_pseudo": int(self.input_pseudo),
            "kept": int(self.kept),
            "dropped": int(self.dropped),
            "recovered": int(self.recovered),
            "output_pseudo": int(self.output_pseudo),
            "drop_rate": float(self.dropped / max(self.input_pseudo, 1)),
            "recovery_per_gt": float(self.recovered / max(self.gt, 1)),
            "per_class": per_class,
        }


def _count_by_class(items: list[dict[str, Any]], *, num_classes: int) -> list[int]:
    counts = [0] * int(num_classes)
    for item in items:
        class_id = int(item.get("category_id", -1))
        if 0 <= class_id < int(num_classes):
            counts[class_id] += 1
    return counts


def _best_unmatched_gt(
    row: dict[str, Any],
    annotations: list[dict[str, Any]],
    matched_gt: set[int],
    *,
    match_iou: float,
) -> tuple[int | None, float]:
    class_id = int(row["category_id"])
    best_index: int | None = None
    best_iou = float(match_iou)
    for gt_index, ann in enumerate(annotations):
        if gt_index in matched_gt:
            continue
        if int(ann["category_id"]) != class_id:
            continue
        iou = xyxy_iou(row["bbox"], ann["bbox"])
        if iou >= best_iou:
            best_iou = float(iou)
            best_index = int(gt_index)
    return best_index, best_iou


def _missed_gt_indices(
    rows: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    policies: list[str],
    match_iou: float,
) -> list[int]:
    missed: list[int] = []
    for gt_index, ann in enumerate(annotations):
        class_id = int(ann["category_id"])
        if class_id < 0 or class_id >= len(policies):
            continue
        if not _policy_recovers(policies[class_id]):
            continue
        covered = False
        for row in rows:
            if int(row["category_id"]) != class_id:
                continue
            if xyxy_iou(row["bbox"], ann["bbox"]) >= float(match_iou):
                covered = True
                break
        if not covered:
            missed.append(int(gt_index))
    return missed


def apply_oracle_pseudo_intervention(
    *,
    sample: dict[str, Any],
    pseudo_rows: list[dict[str, Any]],
    cfg: Any,
    num_classes: int,
    class_names: list[str] | tuple[str, ...] = (),
) -> OraclePseudoResult:
    """Apply hidden-GT pseudo filtering/recovery to one unlabeled target image."""

    stats = OraclePseudoStats(num_classes)
    rows = _normalize_rows(pseudo_rows, num_classes=num_classes)
    annotations = _normalize_annotations(sample, num_classes=num_classes)
    policies = _policy_map_from_cfg(cfg, num_classes=num_classes, class_names=class_names)
    match_iou = float(_cfg_get(cfg, "match_iou", 0.5))
    recovery_score = float(_cfg_get(cfg, "recovery_score", _cfg_get(cfg, "score", 1.0)))

    stats.images = 1
    stats.gt = len(annotations)
    stats.input_pseudo = len(rows)
    stats.gt_by_class = _count_by_class(annotations, num_classes=num_classes)
    stats.input_by_class = _count_by_class(rows, num_classes=num_classes)

    kept_rows: list[dict[str, Any]] = []
    matched_gt: set[int] = set()
    for row in sorted(rows, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        class_id = int(row["category_id"])
        if not _policy_filters(policies[class_id]):
            kept_rows.append(dict(row))
            continue

        gt_index, best_iou = _best_unmatched_gt(row, annotations, matched_gt, match_iou=match_iou)
        if gt_index is None:
            stats.dropped += 1
            stats.dropped_by_class[class_id] += 1
            continue
        matched_gt.add(int(gt_index))
        kept_row = dict(row)
        kept_row["_oracle_policy"] = "filter"
        kept_row["_oracle_matched_iou"] = float(best_iou)
        kept_rows.append(kept_row)

    kept_rows = sorted(
        kept_rows,
        key=lambda item: (int(item.get("query_index", 10**9)), -float(item.get("score", 0.0))),
    )
    threshold_rows = [dict(row) for row in kept_rows if not bool(row.get("_oracle_recovered", False))]
    missed_indices = _missed_gt_indices(
        threshold_rows,
        annotations,
        policies=policies,
        match_iou=match_iou,
    )
    for gt_index in missed_indices:
        class_id = int(annotations[gt_index]["category_id"])
        stats.missed_before_recovery_by_class[class_id] += 1

    output_rows = [dict(row) for row in kept_rows]
    for gt_index in missed_indices:
        ann = annotations[gt_index]
        class_id = int(ann["category_id"])
        recovered_row = {
            "bbox": [float(value) for value in ann["bbox"]],
            "category_id": class_id,
            "score": recovery_score,
            "query_index": -1,
            "_oracle_policy": "recover",
            "_oracle_recovered": True,
            "_oracle_gt_index": int(gt_index),
        }
        output_rows.append(recovered_row)
        stats.recovered += 1
        stats.recovered_by_class[class_id] += 1

    stats.kept = len(kept_rows)
    stats.output_pseudo = len(output_rows)
    stats.kept_by_class = _count_by_class(kept_rows, num_classes=num_classes)
    stats.output_by_class = _count_by_class(output_rows, num_classes=num_classes)

    return OraclePseudoResult(rows=output_rows, threshold_rows=threshold_rows, stats=stats)
