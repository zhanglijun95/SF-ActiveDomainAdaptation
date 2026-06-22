#!/usr/bin/env python
"""Run local DINO query-attention diagnostics for sparse target labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DETREX_ROOT = REPO_ROOT / "external" / "detrex"
if str(DETREX_ROOT) not in sys.path:
    sys.path.insert(0, str(DETREX_ROOT))

from detectron2.checkpoint import DetectionCheckpointer

from baselines.ddt_daod.config import resolve_ddt_daod_run_dir, resolve_ddt_daod_source_ckpt_path
from baselines.ddt_daod.pseudo import filter_pseudo_rows
from src.config import load_config
from src.data.daod.analysis import raw_output_to_query_rows
from src.data.daod.detectron2 import materialize_daod_dicts
from src.data.daod.pairs import get_daod_thing_classes
from src.engine.daod_query_attention_diagnostic import (
    DinoDecoderAttentionRecorder,
    best_gt_for_prediction,
    best_query_for_gt,
    resolve_sparse_target_split,
    summarize_query_attention,
    summarize_rows,
    write_csv,
    write_json,
    write_jsonl,
)
from src.models import build_daod_model, run_daod_raw_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether DINO decoder deformable-attention sampling points "
            "localize inside sparse target GT boxes and pseudo boxes."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/baselines/ddt_daod/cityscapes_to_foggy_cityscapes_dino_random_budget005_seed42.yaml",
        help="DAOD config used to build the DINO model and sparse random split.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint to analyze. If omitted, the script tries teacher_last.pth, "
            "student_last.pth, then the configured source checkpoint."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional override for cfg.data.root, e.g. ~/data/ins-seg/cityscapes.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-labeled-images", type=int, default=25)
    parser.add_argument("--max-unlabeled-images", type=int, default=25)
    parser.add_argument("--pseudo-threshold", type=float, default=None)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--pseudo-correct-iou", type=float, default=0.5)
    parser.add_argument("--topk-points", type=int, default=8)
    parser.add_argument(
        "--allow-any-class-gt-match",
        action="store_true",
        help="For labeled GT objects, match by IoU only instead of requiring predicted class == GT class.",
    )
    parser.add_argument(
        "--no-full-gt-pseudo-check",
        action="store_true",
        help="Do not use hidden GT annotations to split pseudo objects into correct/wrong diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _maybe_override_data_root(cfg, args.data_root)

    seed = int(getattr(cfg, "seed", 42))
    active_cfg = getattr(getattr(cfg, "method", object()), "active", object())
    budget_total = getattr(active_cfg, "budget_total", 0.05)
    checkpoint = _resolve_checkpoint(cfg, args.checkpoint)
    device = torch.device(args.device)
    output_dir = _resolve_output_dir(cfg, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[query-attn] config={args.config}")
    print(f"[query-attn] checkpoint={checkpoint}")
    print(f"[query-attn] data_root={cfg.data.root}")
    print(f"[query-attn] device={device} output_dir={output_dir}")

    adapter = build_daod_model(cfg, load_weights=False, device=device)
    DetectionCheckpointer(adapter.model).load(str(checkpoint))
    adapter.model.eval()

    target_train = materialize_daod_dicts(cfg, "target_train")
    labeled_items, unlabeled_items, selected_ids, split_summary = resolve_sparse_target_split(
        target_train,
        budget_total=budget_total,
        seed=seed,
    )
    labeled_items = labeled_items[: max(0, int(args.max_labeled_images))]
    unlabeled_items = unlabeled_items[: max(0, int(args.max_unlabeled_images))]
    thing_classes = list(get_daod_thing_classes(cfg))
    pseudo_cfg = getattr(getattr(cfg, "method", object()), "pseudo", object())
    threshold = (
        float(args.pseudo_threshold)
        if args.pseudo_threshold is not None
        else float(getattr(pseudo_cfg, "threshold", 0.4))
    )
    thresholds = [threshold for _ in range(int(cfg.data.num_classes))]

    rows: list[dict[str, Any]] = []
    counters: dict[str, Any] = {
        "labeled_images_scanned": len(labeled_items),
        "unlabeled_images_scanned": len(unlabeled_items),
        "labeled_gt_objects": 0,
        "labeled_gt_matched": 0,
        "pseudo_objects": 0,
        "pseudo_correct": 0,
        "pseudo_wrong": 0,
    }

    with DinoDecoderAttentionRecorder(adapter.model) as recorder:
        if recorder.num_patched_modules <= 0:
            raise RuntimeError("No DINO decoder deformable cross-attention modules were found to diagnose.")
        print(f"[query-attn] patched_decoder_cross_attention_modules={recorder.num_patched_modules}")

        for idx, sample in enumerate(labeled_items, start=1):
            print(f"[query-attn][labeled] {idx}/{len(labeled_items)} {sample['sample_id']}")
            recorder.clear()
            raw_output = run_daod_raw_outputs(adapter, sample, with_grad=False)[0]
            query_rows = raw_output_to_query_rows(
                raw_output,
                image_size=(int(sample["height"]), int(sample["width"])),
            )
            rows.extend(
                _rows_for_labeled_sample(
                    sample,
                    query_rows,
                    recorder.records,
                    thing_classes=thing_classes,
                    match_iou=float(args.match_iou),
                    require_class_match=not bool(args.allow_any_class_gt_match),
                    topk_points=int(args.topk_points),
                    counters=counters,
                )
            )

        for idx, sample in enumerate(unlabeled_items, start=1):
            print(f"[query-attn][pseudo] {idx}/{len(unlabeled_items)} {sample['sample_id']}")
            recorder.clear()
            raw_output = run_daod_raw_outputs(adapter, sample, with_grad=False)[0]
            query_rows = raw_output_to_query_rows(
                raw_output,
                image_size=(int(sample["height"]), int(sample["width"])),
            )
            pseudo_rows = filter_pseudo_rows(
                query_rows,
                thresholds=thresholds,
                dedup_iou_thresh=float(getattr(pseudo_cfg, "dedup_iou_thresh", 0.7)),
            )
            rows.extend(
                _rows_for_pseudo_sample(
                    sample,
                    pseudo_rows,
                    recorder.records,
                    thing_classes=thing_classes,
                    pseudo_correct_iou=float(args.pseudo_correct_iou),
                    use_full_gt=not bool(args.no_full_gt_pseudo_check),
                    topk_points=int(args.topk_points),
                    counters=counters,
                )
            )

    summary = {
        "config": str(Path(args.config)),
        "checkpoint": str(checkpoint),
        "data_root": str(cfg.data.root),
        "seed": seed,
        "budget_total": budget_total,
        "split": split_summary,
        "selected_id_count": len(selected_ids),
        "pseudo_threshold": threshold,
        "match_iou": float(args.match_iou),
        "pseudo_correct_iou": float(args.pseudo_correct_iou),
        "topk_points": int(args.topk_points),
        "counters": counters,
        "attention_summary": summarize_rows(rows),
        "output_dir": str(output_dir),
    }

    write_jsonl(output_dir / "query_attention_rows.jsonl", rows)
    write_csv(output_dir / "query_attention_rows.csv", rows)
    write_json(output_dir / "summary.json", summary)
    _print_summary(summary)


def _rows_for_labeled_sample(
    sample: dict[str, Any],
    query_rows: list[dict[str, Any]],
    snapshots,
    *,
    thing_classes: list[str],
    match_iou: float,
    require_class_match: bool,
    topk_points: int,
    counters: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for ann_idx, ann in enumerate(sample.get("annotations", [])):
        counters["labeled_gt_objects"] += 1
        matched_row, matched_iou = best_query_for_gt(
            query_rows,
            ann,
            require_class_match=require_class_match,
            min_iou=match_iou,
        )
        if matched_row is None:
            continue
        counters["labeled_gt_matched"] += 1
        category_id = int(ann["category_id"])
        attention = summarize_query_attention(
            snapshots,
            query_index=int(matched_row["query_index"]),
            reference_box=[float(v) for v in ann["bbox"]],
            image_height=int(sample["height"]),
            image_width=int(sample["width"]),
            topk_points=topk_points,
        )
        rows.append(
            {
                "group": "labeled_gt_match",
                "sample_id": sample["sample_id"],
                "file_name": sample["file_name"],
                "object_index": ann_idx,
                "query_index": int(matched_row["query_index"]),
                "category_id": category_id,
                "class_name": _class_name(thing_classes, category_id),
                "pred_category_id": int(matched_row["category_id"]),
                "pred_class_name": _class_name(thing_classes, int(matched_row["category_id"])),
                "score": float(matched_row.get("score", 0.0)),
                "match_iou": float(matched_iou),
                "bbox": [float(v) for v in ann["bbox"]],
                **attention,
            }
        )
    return rows


def _rows_for_pseudo_sample(
    sample: dict[str, Any],
    pseudo_rows: list[dict[str, Any]],
    snapshots,
    *,
    thing_classes: list[str],
    pseudo_correct_iou: float,
    use_full_gt: bool,
    topk_points: int,
    counters: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for pseudo_idx, pseudo_row in enumerate(pseudo_rows):
        counters["pseudo_objects"] += 1
        category_id = int(pseudo_row["category_id"])
        pred_box = [float(v) for v in pseudo_row["bbox"]]
        gt_match = (
            best_gt_for_prediction(sample, pred_box, category_id)
            if use_full_gt
            else {"best_gt_iou": None, "best_gt_class": None, "best_same_class_iou": None}
        )
        is_correct = (
            use_full_gt
            and gt_match["best_same_class_iou"] is not None
            and float(gt_match["best_same_class_iou"]) >= float(pseudo_correct_iou)
        )
        group = "pseudo_correct" if is_correct else ("pseudo_wrong" if use_full_gt else "pseudo")
        if group == "pseudo_correct":
            counters["pseudo_correct"] += 1
        elif group == "pseudo_wrong":
            counters["pseudo_wrong"] += 1

        attention = summarize_query_attention(
            snapshots,
            query_index=int(pseudo_row["query_index"]),
            reference_box=pred_box,
            image_height=int(sample["height"]),
            image_width=int(sample["width"]),
            topk_points=topk_points,
        )
        rows.append(
            {
                "group": group,
                "sample_id": sample["sample_id"],
                "file_name": sample["file_name"],
                "object_index": pseudo_idx,
                "query_index": int(pseudo_row["query_index"]),
                "category_id": category_id,
                "class_name": _class_name(thing_classes, category_id),
                "score": float(pseudo_row.get("score", 0.0)),
                "bbox": pred_box,
                "best_gt_iou": gt_match["best_gt_iou"],
                "best_gt_class": gt_match["best_gt_class"],
                "best_same_class_iou": gt_match["best_same_class_iou"],
                **attention,
            }
        )
    return rows


def _resolve_checkpoint(cfg: Any, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        path = Path(checkpoint_arg).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = []
    try:
        run_dir = resolve_ddt_daod_run_dir(cfg)
        candidates.extend([run_dir / "teacher_last.pth", run_dir / "student_last.pth"])
    except Exception:
        pass
    try:
        candidates.append(resolve_ddt_daod_source_ckpt_path(cfg, which=str(getattr(cfg.detector, "source_ckpt", "final"))))
    except Exception:
        pass
    try:
        candidates.append(resolve_ddt_daod_source_ckpt_path(cfg, which="final"))
        candidates.append(resolve_ddt_daod_source_ckpt_path(cfg, which="best"))
    except Exception:
        pass

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not auto-resolve a checkpoint. Pass --checkpoint explicitly. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def _resolve_output_dir(cfg: Any, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).expanduser()
    exp_name = str(getattr(getattr(cfg, "method", object()), "exp_name", "dino_query_attention"))
    return Path("runs/diagnostics/dino_query_attention") / exp_name


def _maybe_override_data_root(cfg: Any, data_root: str | None) -> None:
    if data_root:
        cfg.data.root = str(Path(data_root).expanduser())
        return
    root = Path(str(cfg.data.root)).expanduser()
    if root.exists():
        cfg.data.root = str(root)
        return
    local_candidate = Path("/local") / str(root).lstrip("/")
    if local_candidate.exists():
        cfg.data.root = str(local_candidate)


def _class_name(thing_classes: list[str], category_id: int) -> str:
    if 0 <= int(category_id) < len(thing_classes):
        return thing_classes[int(category_id)]
    return str(category_id)


def _print_summary(summary: dict[str, Any]) -> None:
    counters = summary["counters"]
    print(
        "[query-attn] done "
        f"rows={summary['attention_summary']['num_rows']} "
        f"gt_matched={counters['labeled_gt_matched']}/{counters['labeled_gt_objects']} "
        f"pseudo_correct={counters['pseudo_correct']} pseudo_wrong={counters['pseudo_wrong']}"
    )
    groups = summary["attention_summary"]["groups"]
    for group_name in sorted(groups):
        group = groups[group_name]
        inside = group.get("final_inside_box_mass_mean")
        center = group.get("final_center_distance_mean")
        entropy = group.get("final_attention_entropy_mean")
        inside_s = f"{inside:.3f}" if isinstance(inside, (int, float)) else "n/a"
        center_s = f"{center:.3f}" if isinstance(center, (int, float)) else "n/a"
        entropy_s = f"{entropy:.3f}" if isinstance(entropy, (int, float)) else "n/a"
        print(
            f"[query-attn][group] {group_name}: count={group['count']} "
            f"final_inside={inside_s} center_dist={center_s} entropy={entropy_s}"
        )
    print(f"[query-attn] wrote diagnostics under {summary['output_dir']}")


if __name__ == "__main__":
    main()
