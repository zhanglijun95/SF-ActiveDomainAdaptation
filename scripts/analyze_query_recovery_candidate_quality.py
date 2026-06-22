"""Analyze query-recovery candidate quality for DDT + sparse target labels.

Two modes are intentionally separated:

1. `summary`: no model inference. Reads finished run artifacts and estimates
   how much noisy revival pressure the sparse-fit recovery rule likely added.
2. `full-audit`: GPU/slow. Re-fits the recovery scorer from the sparse selected
   labels, applies it to target-train images, and matches selected candidates to
   hidden GT for an offline oracle audit.

The full audit is diagnostic only. It must never be used during training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import torch

from baselines.ddt_daod.config import resolve_ddt_daod_run_dir, resolve_ddt_daod_source_ckpt_path
from baselines.ddt_daod.pseudo import filter_pseudo_rows
from baselines.ddt_daod.trainer import (
    _build_query_recovery_teacher_items,
    _build_sparse_target_split,
    _query_recovery_num_views,
)
from src.config import load_config
from src.data.daod import get_daod_thing_classes
from src.data.daod.detectron2 import materialize_daod_dicts
from src.engine.daod_oracle_pseudo import xyxy_iou
from src.engine.daod_query_recovery import fit_query_recovery_scorer
from src.engine.daod_round_trainer import _limit_samples, _teacher_outputs_for_unlabeled
from src.models import build_daod_model


DEFAULT_BASELINE_RUN = Path(
    "runs/baselines/ddt_daod/"
    "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_budget005_seed42/"
    "cityscapes__to__foggy_cityscapes/dino_r50_4scale_12ep"
)
DEFAULT_RECOVERY_RUNS = [
    Path(
        "runs/baselines/ddt_daod/"
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_query_revival_scorer_foreground_budget005_seed42/"
        "cityscapes__to__foggy_cityscapes/dino_r50_4scale_12ep"
    ),
    Path(
        "runs/baselines/ddt_daod/"
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_query_revival_multiview_foreground_budget005_seed42/"
        "cityscapes__to__foggy_cityscapes/dino_r50_4scale_12ep"
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    summary = subparsers.add_parser("summary", help="Summarize finished query-revival runs.")
    summary.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    summary.add_argument("--run", action="append", type=Path, default=None, help="Run dir to summarize.")
    summary.add_argument("--output", type=Path, default=Path("runs/analysis/query_recovery_candidate_quality_summary.json"))

    audit = subparsers.add_parser("full-audit", help="Regenerate and GT-audit recovery candidates.")
    audit.add_argument("--config", required=True, type=Path)
    audit.add_argument(
        "--checkpoint",
        default="source",
        help="source, teacher_last, student_last, or explicit checkpoint path. Default: source.",
    )
    audit.add_argument("--split", choices=["all", "unlabeled", "selected"], default="all")
    audit.add_argument("--max-images", type=int, default=0, help="0 means no limit.")
    audit.add_argument("--batch-size", type=int, default=1)
    audit.add_argument("--match-iou", type=float, default=0.5)
    audit.add_argument("--seed", type=int, default=None)
    audit.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.mode == "summary":
        run_summary_mode(args)
    elif args.mode == "full-audit":
        run_full_audit_mode(args)


def run_summary_mode(args: argparse.Namespace) -> None:
    baseline = _load_run_summary(args.baseline_run)
    run_dirs = args.run if args.run else DEFAULT_RECOVERY_RUNS
    payload = {
        "baseline_run": str(args.baseline_run),
        "baseline": _metric_record("DDT+random", baseline, baseline_ap50=None),
        "runs": [],
        "note": (
            "Estimated false revival targets use sparse-fit threshold precision from the selected "
            "5% labeled target audit. This is not a hidden-GT full audit."
        ),
    }
    baseline_ap50 = payload["baseline"]["AP50"]
    baseline_ap = payload["baseline"]["AP"]
    print("\nSummary from finished run artifacts")
    print(f"Baseline DDT+random: AP50={baseline_ap50:.3f} AP={baseline_ap:.3f}")

    for run_dir in run_dirs:
        summary = _load_run_summary(run_dir)
        record = _metric_record(_run_label(run_dir), summary, baseline_ap50=baseline_ap50)
        recovery = summary.get("query_recovery", {})
        fit = recovery.get("fit", {})
        global_stats = fit.get("global_threshold_stats", {})
        sparse_precision = _safe_float(global_stats.get("precision"))
        sparse_recall = _safe_float(global_stats.get("recall"))
        revival_history = summary.get("query_revival", {}).get("history", [])
        total_targets = int(sum(int(entry.get("loss_targets", 0) or 0) for entry in revival_history))
        total_matched = int(sum(int(entry.get("matched_targets", 0) or 0) for entry in revival_history))
        estimated_false = None
        if sparse_precision is not None:
            estimated_false = float(total_targets * max(0.0, 1.0 - sparse_precision))
        record.update(
            {
                "run_dir": str(run_dir),
                "sparse_fit_precision": sparse_precision,
                "sparse_fit_recall": sparse_recall,
                "fit_candidates": fit.get("fit_candidates"),
                "fit_positive": fit.get("fit_positive"),
                "class_gates": fit.get("class_gates"),
                "class_budgets": fit.get("class_budgets"),
                "revival_targets": total_targets,
                "revival_matched_targets": total_matched,
                "estimated_false_revival_targets_from_sparse_precision": estimated_false,
                "query_recovery_history": recovery.get("history", []),
                "query_revival_history": revival_history,
                "per_class_delta_AP": _per_class_delta_ap(summary, baseline),
            }
        )
        payload["runs"].append(record)
        print(
            f"{record['name']}: AP50={record['AP50']:.3f} "
            f"delta={record['delta_AP50']:.3f} AP={record['AP']:.3f} "
            f"sparse_precision={_fmt(sparse_precision)} revival_targets={total_targets} "
            f"est_false={_fmt(estimated_false)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")


def run_full_audit_mode(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else getattr(cfg, "seed", 42))
    method_cfg = getattr(cfg, "method", object())
    train_cfg = getattr(method_cfg, "train", object())
    pseudo_cfg = getattr(method_cfg, "pseudo", object())
    active_cfg = getattr(method_cfg, "active", object())
    recovery_cfg = getattr(method_cfg, "query_recovery", object())
    if not bool(getattr(recovery_cfg, "enabled", False)):
        raise ValueError("Config must have method.query_recovery.enabled=true for full-audit mode.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] CUDA is not available. Full audit will be very slow on CPU.", flush=True)
    teacher_adapter = build_daod_model(cfg, load_weights=False, device=device)
    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint)
    DetectionCheckpointer(teacher_adapter.model).load(str(checkpoint_path))
    teacher_adapter.model.eval()

    target_train = materialize_daod_dicts(cfg, "target_train")
    target_train = _limit_samples(target_train, int(getattr(train_cfg, "max_target_samples", 0)))
    target_labeled, target_unlabeled, selected_ids, active_plan = _build_sparse_target_split(
        target_train,
        active_cfg,
        seed=seed,
    )
    if not target_labeled:
        raise ValueError("No sparse selected labeled target images found; cannot fit recovery scorer.")

    num_classes = int(cfg.data.num_classes)
    class_names = list(get_daod_thing_classes(cfg))
    base_threshold = float(getattr(pseudo_cfg, "threshold", 0.4))
    thresholds = [base_threshold] * num_classes
    dedup_iou_thresh = float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7))

    num_views = _query_recovery_num_views(recovery_cfg)
    recovery_cfg["_resolved_num_views"] = int(num_views)
    print(
        f"[full-audit] fitting scorer on {len(target_labeled)} selected images "
        f"views={num_views} checkpoint={checkpoint_path}",
        flush=True,
    )
    with torch.no_grad():
        primary_fit_items = _teacher_outputs_for_unlabeled(
            teacher_adapter,
            target_labeled,
            weak_view_rng=np.random.default_rng(seed + 8181),
        )
        fit_items = _build_query_recovery_teacher_items(
            teacher_adapter=teacher_adapter,
            primary_items=primary_fit_items,
            source_batch=target_labeled,
            recovery_cfg=recovery_cfg,
            teacher_device=device,
            seed=seed + 9191,
            step_offset=0,
        )
    scorer = fit_query_recovery_scorer(
        fit_items,
        thresholds=thresholds,
        num_classes=num_classes,
        recovery_cfg=recovery_cfg,
        seed=seed,
        dedup_iou_thresh=dedup_iou_thresh,
    )

    if args.split == "selected":
        audit_samples = list(target_labeled)
    elif args.split == "unlabeled":
        audit_samples = [sample for sample in target_train if str(sample["sample_id"]) not in selected_ids]
    else:
        audit_samples = list(target_train)
    if args.max_images > 0:
        audit_samples = audit_samples[: int(args.max_images)]

    print(f"[full-audit] auditing {len(audit_samples)} images split={args.split}", flush=True)
    aggregate = _new_audit_counts(num_classes)
    sample_rows = []
    for start in range(0, len(audit_samples), max(int(args.batch_size), 1)):
        batch = audit_samples[start : start + max(int(args.batch_size), 1)]
        with torch.no_grad():
            primary_items = _teacher_outputs_for_unlabeled(
                teacher_adapter,
                batch,
                weak_view_rng=np.random.default_rng(seed + 31337 + start),
            )
            recovery_items = _build_query_recovery_teacher_items(
                teacher_adapter=teacher_adapter,
                primary_items=primary_items,
                source_batch=batch,
                recovery_cfg=recovery_cfg,
                teacher_device=device,
                seed=seed + 41443,
                step_offset=start,
            )
        for teacher_item, recovery_item in zip(primary_items, recovery_items):
            pseudo_rows = filter_pseudo_rows(
                teacher_item["query_rows"],
                thresholds=thresholds,
                dedup_iou_thresh=dedup_iou_thresh,
            )
            recovered_rows, selection_stats = scorer.select(
                recovery_item["query_rows"],
                thresholds=thresholds,
                dedup_iou_thresh=dedup_iou_thresh,
                sample=recovery_item["sample"],
                existing_rows=pseudo_rows,
            )
            sample_payload = _audit_one_sample(
                sample=recovery_item["sample"],
                pseudo_rows=pseudo_rows,
                recovered_rows=recovered_rows,
                num_classes=num_classes,
                match_iou=float(args.match_iou),
            )
            sample_payload["candidate_count"] = int(selection_stats.candidates)
            _accumulate_counts(aggregate, sample_payload, num_classes=num_classes)
            sample_rows.append(sample_payload)
        if (start + len(batch)) % 50 == 0 or start + len(batch) >= len(audit_samples):
            print(f"[full-audit] audited {start + len(batch)}/{len(audit_samples)}", flush=True)

    summary = _summarize_audit_counts(aggregate, class_names=class_names)
    payload = {
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "max_images": int(args.max_images),
        "match_iou": float(args.match_iou),
        "active_plan": {
            "budget_k": active_plan.get("budget_k"),
            "target_total": active_plan.get("target_total"),
            "selected_count": len(selected_ids),
        },
        "scorer": scorer.summary(),
        "summary": summary,
        "samples": sample_rows,
    }
    output = args.output
    if output is None:
        output = resolve_ddt_daod_run_dir(cfg) / f"query_recovery_full_audit_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[full-audit] selected={summary['selected']} precision={summary['precision']:.4f} "
        f"missed_gt={summary['missed_gt']} recovered_missed_gt={summary['recovered_missed_gt']} "
        f"recovery_recall={summary['recovery_recall']:.4f}",
        flush=True,
    )
    print(f"Saved: {output}", flush=True)


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    return json.loads(summary_path.read_text())


def _run_label(run_dir: Path) -> str:
    if len(run_dir.parents) >= 3:
        return run_dir.parents[1].name
    return run_dir.name


def _metric_record(name: str, summary: dict[str, Any], *, baseline_ap50: float | None) -> dict[str, Any]:
    bbox = summary.get("final_target_val_metrics", {}).get("bbox", {})
    ap50 = float(bbox.get("AP50"))
    record = {
        "name": str(name),
        "final_model": summary.get("final_model"),
        "AP": float(bbox.get("AP")),
        "AP50": ap50,
        "global_step": int(summary.get("global_step", 0)),
    }
    record["delta_AP50"] = None if baseline_ap50 is None else float(ap50 - baseline_ap50)
    return record


def _per_class_delta_ap(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    current = summary.get("final_target_val_metrics", {}).get("bbox", {})
    base = baseline.get("final_target_val_metrics", {}).get("bbox", {})
    result = {}
    for key, value in current.items():
        if not key.startswith("AP-") or key not in base:
            continue
        result[key[3:]] = float(value - base[key])
    return result


def _resolve_checkpoint(cfg: Any, checkpoint: str) -> Path:
    checkpoint = str(checkpoint)
    if checkpoint == "source":
        return resolve_ddt_daod_source_ckpt_path(
            cfg,
            which=str(getattr(getattr(cfg, "detector", object()), "source_ckpt", "best")),
        )
    run_dir = resolve_ddt_daod_run_dir(cfg)
    if checkpoint == "teacher_last":
        return run_dir / "teacher_last.pth"
    if checkpoint == "student_last":
        return run_dir / "student_last.pth"
    return Path(checkpoint)


def _audit_one_sample(
    *,
    sample: dict[str, Any],
    pseudo_rows: list[dict[str, Any]],
    recovered_rows: list[dict[str, Any]],
    num_classes: int,
    match_iou: float,
) -> dict[str, Any]:
    gt_rows = [
        {"bbox": [float(v) for v in ann["bbox"]], "category_id": int(ann["category_id"])}
        for ann in sample.get("annotations", [])
        if 0 <= int(ann.get("category_id", -1)) < int(num_classes)
    ]
    standard_covered = _covered_gt_indices(pseudo_rows, gt_rows, match_iou=match_iou)
    missed_gt = set(range(len(gt_rows))) - standard_covered
    matched_gt: set[int] = set()
    recovered_missed_gt: set[int] = set()
    selected_by_class = [0] * int(num_classes)
    true_by_class = [0] * int(num_classes)
    false_by_class = [0] * int(num_classes)
    recovered_missed_by_class = [0] * int(num_classes)

    for row in sorted(recovered_rows, key=lambda item: float(item.get("_query_recovery_score", item.get("score", 0.0))), reverse=True):
        class_id = int(row.get("category_id", -1))
        if class_id < 0 or class_id >= int(num_classes):
            continue
        selected_by_class[class_id] += 1
        best_idx, best_iou = _best_gt(row, gt_rows, class_id=class_id)
        if best_idx is not None and best_iou >= float(match_iou) and best_idx not in matched_gt:
            matched_gt.add(best_idx)
            true_by_class[class_id] += 1
            if best_idx in missed_gt and best_idx not in recovered_missed_gt:
                recovered_missed_gt.add(best_idx)
                recovered_missed_by_class[class_id] += 1
        else:
            false_by_class[class_id] += 1

    gt_by_class = [0] * int(num_classes)
    missed_by_class = [0] * int(num_classes)
    for gt_index, gt in enumerate(gt_rows):
        class_id = int(gt["category_id"])
        gt_by_class[class_id] += 1
        if gt_index in missed_gt:
            missed_by_class[class_id] += 1

    return {
        "sample_id": str(sample["sample_id"]),
        "gt": len(gt_rows),
        "standard_pseudo": len(pseudo_rows),
        "standard_covered_gt": len(standard_covered),
        "missed_gt": len(missed_gt),
        "selected": len(recovered_rows),
        "true": int(sum(true_by_class)),
        "false": int(sum(false_by_class)),
        "recovered_missed_gt": len(recovered_missed_gt),
        "gt_by_class": gt_by_class,
        "missed_by_class": missed_by_class,
        "selected_by_class": selected_by_class,
        "true_by_class": true_by_class,
        "false_by_class": false_by_class,
        "recovered_missed_by_class": recovered_missed_by_class,
    }


def _covered_gt_indices(
    rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    *,
    match_iou: float,
) -> set[int]:
    covered = set()
    for gt_index, gt in enumerate(gt_rows):
        class_id = int(gt["category_id"])
        for row in rows:
            if int(row.get("category_id", -1)) != class_id:
                continue
            if xyxy_iou([float(v) for v in row["bbox"]], gt["bbox"]) >= float(match_iou):
                covered.add(int(gt_index))
                break
    return covered


def _best_gt(
    row: dict[str, Any],
    gt_rows: list[dict[str, Any]],
    *,
    class_id: int,
) -> tuple[int | None, float]:
    best_idx = None
    best_iou = 0.0
    for gt_index, gt in enumerate(gt_rows):
        if int(gt["category_id"]) != int(class_id):
            continue
        iou = xyxy_iou([float(v) for v in row["bbox"]], gt["bbox"])
        if iou > best_iou:
            best_iou = float(iou)
            best_idx = int(gt_index)
    return best_idx, best_iou


def _new_audit_counts(num_classes: int) -> dict[str, Any]:
    return {
        "images": 0,
        "gt": 0,
        "standard_pseudo": 0,
        "standard_covered_gt": 0,
        "missed_gt": 0,
        "candidate_count": 0,
        "selected": 0,
        "true": 0,
        "false": 0,
        "recovered_missed_gt": 0,
        "gt_by_class": [0] * int(num_classes),
        "missed_by_class": [0] * int(num_classes),
        "selected_by_class": [0] * int(num_classes),
        "true_by_class": [0] * int(num_classes),
        "false_by_class": [0] * int(num_classes),
        "recovered_missed_by_class": [0] * int(num_classes),
    }


def _accumulate_counts(total: dict[str, Any], sample_payload: dict[str, Any], *, num_classes: int) -> None:
    total["images"] += 1
    for key in [
        "gt",
        "standard_pseudo",
        "standard_covered_gt",
        "missed_gt",
        "candidate_count",
        "selected",
        "true",
        "false",
        "recovered_missed_gt",
    ]:
        total[key] += int(sample_payload.get(key, 0))
    for key in [
        "gt_by_class",
        "missed_by_class",
        "selected_by_class",
        "true_by_class",
        "false_by_class",
        "recovered_missed_by_class",
    ]:
        values = sample_payload.get(key, [0] * int(num_classes))
        for idx in range(int(num_classes)):
            total[key][idx] += int(values[idx])


def _summarize_audit_counts(counts: dict[str, Any], *, class_names: list[str]) -> dict[str, Any]:
    selected = int(counts["selected"])
    missed_gt = int(counts["missed_gt"])
    per_class = {}
    for class_id, class_name in enumerate(class_names):
        cls_selected = int(counts["selected_by_class"][class_id])
        cls_missed = int(counts["missed_by_class"][class_id])
        cls_true = int(counts["true_by_class"][class_id])
        cls_recovered = int(counts["recovered_missed_by_class"][class_id])
        per_class[class_name] = {
            "gt": int(counts["gt_by_class"][class_id]),
            "missed_gt": cls_missed,
            "selected": cls_selected,
            "true": cls_true,
            "false": int(counts["false_by_class"][class_id]),
            "recovered_missed_gt": cls_recovered,
            "precision": float(cls_true / max(cls_selected, 1)),
            "recovery_recall": float(cls_recovered / max(cls_missed, 1)),
        }
    return {
        **{key: int(value) for key, value in counts.items() if not isinstance(value, list)},
        "precision": float(int(counts["true"]) / max(selected, 1)),
        "false_rate": float(int(counts["false"]) / max(selected, 1)),
        "recovery_recall": float(int(counts["recovered_missed_gt"]) / max(missed_gt, 1)),
        "selection_rate": float(int(counts["selected"]) / max(int(counts["candidate_count"]), 1)),
        "per_class": per_class,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
