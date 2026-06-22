#!/usr/bin/env python3
"""Evaluate paper-facing Foggy-Cityscapes DDT DINO checkpoints for per-class AP50.

Detectron2's default JSON reports per-class AP averaged over IoU thresholds
(`AP-car`, etc.). Several DAOD papers instead report per-class AP at IoU=0.5.
This script re-runs target-val evaluation and extracts the AP50 slice directly
from COCOeval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.data.daod import get_daod_thing_classes  # noqa: E402
from src.data.daod.detectron2 import (  # noqa: E402
    build_daod_detection_test_loader,
    export_daod_coco_json,
    materialize_daod_dicts,
    register_daod_eval_dataset,
)
from src.models import build_daod_model  # noqa: E402


RUN_ROOT = REPO_ROOT / "runs"
DDT_RUN_ROOT = RUN_ROOT / "baselines" / "ddt_daod"
FOGGY_DINO_REL = Path("cityscapes__to__foggy_cityscapes") / "dino_r50_4scale_12ep"
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class EvalTarget:
    method: str
    family: str
    seed: int | None
    run_dir: Path
    config_path: Path
    checkpoint_path: Path

    @property
    def key(self) -> str:
        seed_part = "anchor" if self.seed is None else f"seed{self.seed}"
        return f"{self.method}_{seed_part}".replace("/", "__")


DDT_METHODS: tuple[tuple[str, str, str], ...] = (
    ("pure_sfda", "ddt_pure", "ddt_daod_cityscapes_to_foggy_cityscapes_dino_seed{seed}"),
    (
        "random_supervised",
        "ddt_random_5pct",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_budget005_seed{seed}",
    ),
    (
        "selection",
        "selection_threshold_calibration",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_selection_threshold_calibration_budget005_seed{seed}",
    ),
    (
        "selection",
        "selection_threshold_mapping",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_selection_threshold_mapping_budget005_seed{seed}",
    ),
    (
        "selection",
        "selection_pseudo_score_reweight",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_selection_pseudo_score_reweight_budget005_seed{seed}",
    ),
    (
        "completion",
        "completion_query_recovery_scorer",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_query_recovery_scorer_budget005_seed{seed}",
    ),
    (
        "completion",
        "completion_query_recovery_multiview",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_query_recovery_multiview_budget005_seed{seed}",
    ),
    (
        "completion",
        "completion_query_revival_scorer",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_query_revival_scorer_foreground_budget005_seed{seed}",
    ),
    (
        "completion",
        "completion_query_revival_multiview",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_query_revival_multiview_foreground_budget005_seed{seed}",
    ),
    (
        "optimization_control",
        "control_pcgrad_pseudo_only",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_control_target_anchored_pcgrad_pseudo_only_budget005_seed{seed}",
    ),
    (
        "optimization_control",
        "control_sparse_loss_balance",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_control_sparse_loss_balance_budget005_seed{seed}",
    ),
    (
        "optimization_control",
        "control_label_guided_aema",
        "ddt_daod_cityscapes_to_foggy_cityscapes_dino_random_control_label_guided_aema_budget005_seed{seed}",
    ),
)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _existing_config(run_dir: Path) -> Path:
    for name in ("resolved_config.yaml", "config.yaml"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No config.yaml/resolved_config.yaml found under {run_dir}")


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_targets() -> list[EvalTarget]:
    targets: list[EvalTarget] = [
        EvalTarget(
            method="source_only",
            family="detector_anchor",
            seed=None,
            run_dir=RUN_ROOT / "daod_source" / FOGGY_DINO_REL,
            config_path=_require_file(RUN_ROOT / "daod_source" / FOGGY_DINO_REL / "resolved_config.yaml"),
            checkpoint_path=_require_file(RUN_ROOT / "daod_source" / FOGGY_DINO_REL / "model_final.pth"),
        ),
        EvalTarget(
            method="oracle",
            family="detector_anchor",
            seed=None,
            run_dir=RUN_ROOT / "daod_oracle" / FOGGY_DINO_REL,
            config_path=_require_file(RUN_ROOT / "daod_oracle" / FOGGY_DINO_REL / "resolved_config.yaml"),
            checkpoint_path=_require_file(RUN_ROOT / "daod_oracle" / FOGGY_DINO_REL / "model_final.pth"),
        ),
    ]

    for family, method, template in DDT_METHODS:
        for seed in SEEDS:
            exp_name = template.format(seed=seed)
            run_dir = DDT_RUN_ROOT / exp_name / FOGGY_DINO_REL
            targets.append(
                EvalTarget(
                    method=method,
                    family=family,
                    seed=seed,
                    run_dir=_require_file(run_dir),
                    config_path=_existing_config(run_dir),
                    checkpoint_path=_require_file(run_dir / "student_last.pth"),
                )
            )
    return targets


def _set_eval_workers(cfg: Any, num_workers: int) -> None:
    for section in (getattr(cfg, "eval", None), getattr(getattr(cfg, "method", object()), "eval", None)):
        if section is not None:
            section.num_workers = int(num_workers)


def _valid_cityscapes_root(path: Path) -> bool:
    return (path / "leftImg8bit_foggy" / "val").is_dir() and (path / "gtFine" / "val").is_dir()


def _normalize_data_root(cfg: Any, data_root: str | None) -> str:
    current = Path(str(cfg.data.root)).expanduser()
    candidates: list[Path] = []
    if data_root:
        candidates.append(Path(data_root).expanduser())
    candidates.extend(
        [
            current,
            Path.home() / "data" / "ins-seg" / "cityscapes",
            Path("/home/ljzhang/data/ins-seg/cityscapes"),
            Path("/local/home/ljzhang/data/ins-seg/cityscapes"),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _valid_cityscapes_root(candidate):
            cfg.data.root = str(candidate)
            return str(candidate)

    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a local Cityscapes/Foggy-Cityscapes root for evaluation. "
        "Pass --data-root explicitly. Checked:\n  - " + checked
    )


def _limit_dataset(dataset_dicts: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return dataset_dicts
    return dataset_dicts[:limit]


def _load_model(cfg: Any, checkpoint_path: Path, device: str) -> torch.nn.Module:
    adapter = build_daod_model(cfg, load_weights=False, device=device)
    model = adapter.model
    DetectionCheckpointer(model).load(str(checkpoint_path))
    model.eval()
    return model


def _extract_per_class_ap50(
    gt_json_path: Path,
    pred_json_path: Path,
    class_names: tuple[str, ...],
) -> dict[str, float]:
    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    precisions = coco_eval.eval["precision"]
    # precision has shape [IoU threshold, recall threshold, class, area, max detections].
    iou_index = int(np.flatnonzero(np.isclose(coco_eval.params.iouThrs, 0.5))[0])
    area_index = list(coco_eval.params.areaRngLbl).index("all")
    max_det_index = len(coco_eval.params.maxDets) - 1

    per_class: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        precision = precisions[iou_index, :, class_index, area_index, max_det_index]
        precision = precision[precision > -1]
        per_class[class_name] = float(np.mean(precision) * 100.0) if precision.size else float("nan")
    return per_class


def evaluate_target(target: EvalTarget, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(target.config_path)
    data_root = _normalize_data_root(cfg, args.data_root)
    _set_eval_workers(cfg, args.num_workers)

    dataset_dicts = materialize_daod_dicts(cfg, "target_val")
    dataset_dicts = _limit_dataset(dataset_dicts, args.target_val_limit)
    class_names = tuple(get_daod_thing_classes(cfg))
    model = _load_model(cfg, target.checkpoint_path, args.device)

    eval_name = f"per_class_ap50_{target.key}_{time.time_ns()}"
    with tempfile.TemporaryDirectory(prefix=f"{target.key}_eval_") as tmp:
        tmp_dir = Path(tmp)
        json_path = export_daod_coco_json(cfg, dataset_dicts, tmp_dir / "target_val.json")
        register_daod_eval_dataset(eval_name, cfg, dataset_dicts, json_path)
        loader = build_daod_detection_test_loader(cfg, dataset_dicts)
        evaluator = COCOEvaluator(eval_name, output_dir=str(tmp_dir / "eval"))
        metrics = inference_on_dataset(model, loader, evaluator)
        pred_json_path = tmp_dir / "eval" / "coco_instances_results.json"
        per_class_ap50 = _extract_per_class_ap50(json_path, pred_json_path, class_names)

    class_values = [value for value in per_class_ap50.values() if math.isfinite(value)]
    mean_class_ap50 = statistics.mean(class_values) if class_values else float("nan")
    overall_ap50 = float(metrics.get("bbox", {}).get("AP50", float("nan")))
    return {
        "method": target.method,
        "family": target.family,
        "seed": target.seed,
        "run_dir": str(target.run_dir.relative_to(REPO_ROOT)),
        "config_path": str(target.config_path.relative_to(REPO_ROOT)),
        "checkpoint_path": str(target.checkpoint_path.relative_to(REPO_ROOT)),
        "data_root": data_root,
        "target_val_limit": args.target_val_limit,
        "overall_ap50": overall_ap50,
        "mean_class_ap50": mean_class_ap50,
        "mean_class_minus_overall_ap50": mean_class_ap50 - overall_ap50,
        "per_class_ap50": per_class_ap50,
        "bbox_metrics": metrics.get("bbox", {}),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if not math.isfinite(parsed) else f"{parsed:.3f}"


def _write_run_csv(rows: list[dict[str, Any]], class_names: tuple[str, ...], path: Path) -> None:
    columns = [
        "family",
        "method",
        "seed",
        "overall_ap50",
        "mean_class_ap50",
        "mean_class_minus_overall_ap50",
        *[f"AP50-{name}" for name in class_names],
        "checkpoint_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            per_class = row["per_class_ap50"]
            payload = {
                "family": row["family"],
                "method": row["method"],
                "seed": "" if row["seed"] is None else row["seed"],
                "overall_ap50": _fmt(row["overall_ap50"]),
                "mean_class_ap50": _fmt(row["mean_class_ap50"]),
                "mean_class_minus_overall_ap50": _fmt(row["mean_class_minus_overall_ap50"]),
                "checkpoint_path": row["checkpoint_path"],
            }
            payload.update({f"AP50-{name}": _fmt(per_class.get(name)) for name in class_names})
            writer.writerow(payload)


def _aggregate(rows: list[dict[str, Any]], class_names: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for method, group in grouped.items():
        group = sorted(group, key=lambda r: (-1 if r["seed"] is None else int(r["seed"])))
        summary: dict[str, Any] = {
            "family": group[0]["family"],
            "method": method,
            "n": len(group),
            "seeds": ",".join(str(r["seed"]) for r in group if r["seed"] is not None),
        }
        for key in ("overall_ap50", "mean_class_ap50"):
            values = [float(r[key]) for r in group if math.isfinite(float(r[key]))]
            summary[f"{key}_mean"] = statistics.mean(values) if values else float("nan")
            summary[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        for class_name in class_names:
            values = [
                float(r["per_class_ap50"].get(class_name, float("nan")))
                for r in group
                if math.isfinite(float(r["per_class_ap50"].get(class_name, float("nan"))))
            ]
            summary[f"AP50-{class_name}_mean"] = statistics.mean(values) if values else float("nan")
            summary[f"AP50-{class_name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def _write_summary_csv(rows: list[dict[str, Any]], class_names: tuple[str, ...], path: Path) -> None:
    columns = [
        "family",
        "method",
        "n",
        "seeds",
        "overall_ap50_mean",
        "overall_ap50_std",
        "mean_class_ap50_mean",
        "mean_class_ap50_std",
    ]
    for class_name in class_names:
        columns.extend([f"AP50-{class_name}_mean", f"AP50-{class_name}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {
                "family": row["family"],
                "method": row["method"],
                "n": row["n"],
                "seeds": row["seeds"],
            }
            payload.update(
                {
                    column: _fmt(row.get(column))
                    for column in columns
                    if column not in payload
                }
            )
            writer.writerow(payload)


def _write_markdown(rows: list[dict[str, Any]], class_names: tuple[str, ...], path: Path) -> None:
    columns = ["method", "n", "AP50", *class_names]
    lines = [
        "# Foggy-DINO DDT Per-Class AP50",
        "",
        "Values are AP at IoU=0.50. DDT rows are mean over seeds 42/43/44; source/oracle are single checkpoints.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = [
            str(row["method"]),
            str(row["n"]),
            f"{_fmt(row['overall_ap50_mean'])}±{_fmt(row['overall_ap50_std'])}",
        ]
        for class_name in class_names:
            cells.append(
                f"{_fmt(row[f'AP50-{class_name}_mean'])}±{_fmt(row[f'AP50-{class_name}_std'])}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "negative_results_summary" / "per_class_ap50" / "foggy_dino_ddt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Local Cityscapes root. If omitted, SageMaker paths fall back to ~/data/ins-seg/cityscapes.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-val-limit", type=int, default=0, help="Debug only; 0 means full target val.")
    parser.add_argument("--limit-runs", type=int, default=0, help="Debug only; 0 means all runs.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate runs even when cached JSON exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print target list and check paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = build_targets()
    if args.limit_runs > 0:
        targets = targets[: args.limit_runs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"Found {len(targets)} eval targets.")
        for target in targets:
            print(f"{target.key}: {target.checkpoint_path.relative_to(REPO_ROOT)}")
        return

    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        cache_path = args.output_dir / "per_run_json" / f"{target.key}.json"
        if cache_path.exists() and not args.force:
            row = _load_json(cache_path)
            print(f"[{index:02d}/{len(targets)}] skip cached {target.key}: AP50={_fmt(row.get('overall_ap50'))}")
        else:
            print(f"[{index:02d}/{len(targets)}] evaluating {target.key}")
            row = evaluate_target(target, args)
            _json_dump(cache_path, row)
            print(f"[{index:02d}/{len(targets)}] done {target.key}: AP50={_fmt(row.get('overall_ap50'))}")
        rows.append(row)

    # Use the first row config to lock class order. All targets share the same pair.
    first_cfg = load_config(targets[0].config_path)
    class_names = tuple(get_daod_thing_classes(first_cfg))
    rows = sorted(rows, key=lambda r: (r["family"], r["method"], -1 if r["seed"] is None else int(r["seed"])))
    summaries = _aggregate(rows, class_names)
    method_order = {target.method: idx for idx, target in enumerate(targets)}
    summaries = sorted(summaries, key=lambda r: method_order.get(str(r["method"]), 999))

    _json_dump(args.output_dir / "foggy_dino_ddt_per_class_ap50_runs.json", {"rows": rows})
    _write_run_csv(rows, class_names, args.output_dir / "foggy_dino_ddt_per_class_ap50_runs.csv")
    _write_summary_csv(summaries, class_names, args.output_dir / "foggy_dino_ddt_per_class_ap50_summary.csv")
    _write_markdown(summaries, class_names, args.output_dir / "foggy_dino_ddt_per_class_ap50_summary.md")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
