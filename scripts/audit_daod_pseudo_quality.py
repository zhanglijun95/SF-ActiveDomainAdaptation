#!/usr/bin/env python3
"""Audit source-teacher pseudo-label quality against hidden DAOD target GT.

This is a diagnostic script only. It intentionally uses target annotations to
measure pseudo-label precision/recall, so its output must not be used during
training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
import torch

from baselines.ddt_daod.config import resolve_ddt_daod_source_ckpt_path
from baselines.ddt_daod.pseudo import filter_pseudo_rows
from baselines.ddt_daod.trainer import _teacher_outputs_for_unlabeled
from src.config import load_config
from src.data.daod import build_daod_dataset, get_daod_thing_classes, match_predictions_to_gt
from src.models import build_daod_model


def _sample_indices(total: int, *, limit: int, seed: int, sequential: bool) -> list[int]:
    if limit <= 0 or limit >= total:
        return list(range(total))
    if sequential:
        return list(range(limit))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), limit))


def _new_class_stats(num_classes: int) -> list[dict[str, int]]:
    return [
        {
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for _ in range(num_classes)
    ]


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _summarize_class_stats(class_stats: list[dict[str, int]], class_names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, stats in enumerate(class_stats):
        tp = int(stats["tp"])
        fp = int(stats["fp"])
        fn = int(stats["fn"])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                **stats,
                "precision": _rate(tp, tp + fp),
                "recall": _rate(tp, tp + fn),
            }
        )
    return rows


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[pseudo-audit] config={args.config}", flush=True)
    print(f"[pseudo-audit] split={args.split} device={device}", flush=True)

    dataset = build_daod_dataset(cfg, args.split, transform=None)
    total = len(dataset)
    indices = _sample_indices(total, limit=int(args.limit), seed=int(args.seed), sequential=bool(args.sequential))
    print(f"[pseudo-audit] dataset_total={total} sampled={len(indices)}", flush=True)

    adapter = build_daod_model(cfg, load_weights=False, device=device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else resolve_ddt_daod_source_ckpt_path(
        cfg,
        which=str(getattr(getattr(cfg, "detector", object()), "source_ckpt", "final")),
    )
    DetectionCheckpointer(adapter.model).load(str(checkpoint))
    adapter.model.eval()
    print(f"[pseudo-audit] checkpoint={checkpoint}", flush=True)

    num_classes = int(cfg.data.num_classes)
    class_names = tuple(get_daod_thing_classes(cfg))
    thresholds = [float(args.threshold)] * num_classes
    weak_rng = random.Random(int(args.seed) + 9173)
    class_stats = _new_class_stats(num_classes)
    tp_total = fp_total = fn_total = gt_total = pred_total = images_with_pseudo = 0
    pseudo_per_image: list[int] = []

    with torch.no_grad():
        for offset, index in enumerate(indices, start=1):
            sample = dict(dataset[index])
            sample["image_id"] = offset
            teacher_item = _teacher_outputs_for_unlabeled(
                adapter,
                [sample],
                weak_view_rng=weak_rng,
            )[0]
            rows = filter_pseudo_rows(
                teacher_item["query_rows"],
                thresholds=thresholds,
                dedup_iou_thresh=float(args.dedup_iou),
            )
            matches, fps, fns = match_predictions_to_gt(
                sample.get("annotations", []),
                rows,
                iou_thresh=float(args.match_iou),
            )

            pred_total += len(rows)
            gt_total += len(sample.get("annotations", []))
            tp_total += len(matches)
            fp_total += len(fps)
            fn_total += len(fns)
            images_with_pseudo += int(bool(rows))
            pseudo_per_image.append(len(rows))

            for ann in sample.get("annotations", []):
                class_id = int(ann["category_id"])
                if 0 <= class_id < num_classes:
                    class_stats[class_id]["gt"] += 1
            for row in rows:
                class_id = int(row["category_id"])
                if 0 <= class_id < num_classes:
                    class_stats[class_id]["pred"] += 1
            for match in matches:
                class_id = int(match["gt"]["category_id"])
                if 0 <= class_id < num_classes:
                    class_stats[class_id]["tp"] += 1
            for row in fps:
                class_id = int(row["category_id"])
                if 0 <= class_id < num_classes:
                    class_stats[class_id]["fp"] += 1
            for row in fns:
                class_id = int(row["category_id"])
                if 0 <= class_id < num_classes:
                    class_stats[class_id]["fn"] += 1

            if args.progress > 0 and (offset % int(args.progress) == 0 or offset == len(indices)):
                print(
                    "[pseudo-audit] "
                    f"processed={offset}/{len(indices)} "
                    f"precision={_rate(tp_total, tp_total + fp_total):.3f} "
                    f"recall={_rate(tp_total, tp_total + fn_total):.3f} "
                    f"pseudo_per_image={_rate(pred_total, offset):.2f}",
                    flush=True,
                )

    summary = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "split": str(args.split),
        "dataset_total": int(total),
        "sampled": int(len(indices)),
        "threshold": float(args.threshold),
        "dedup_iou": float(args.dedup_iou),
        "match_iou": float(args.match_iou),
        "gt_total": int(gt_total),
        "pred_total": int(pred_total),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "precision": _rate(tp_total, tp_total + fp_total),
        "recall": _rate(tp_total, tp_total + fn_total),
        "images_with_pseudo": int(images_with_pseudo),
        "mean_pseudo_per_image": _rate(pred_total, len(indices)),
        "per_class": _summarize_class_stats(class_stats, class_names),
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[pseudo-audit] wrote={output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="target_val", choices=["target_train", "target_val"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--dedup-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress", type=int, default=25)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = run_audit(args)
    print(
        "[pseudo-audit][summary] "
        f"precision={summary['precision']:.3f} "
        f"recall={summary['recall']:.3f} "
        f"gt={summary['gt_total']} pred={summary['pred_total']} "
        f"mean_pseudo_per_image={summary['mean_pseudo_per_image']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
