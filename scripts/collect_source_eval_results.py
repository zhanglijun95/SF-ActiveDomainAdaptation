#!/usr/bin/env python
"""Aggregate source-eval metrics into CSV summaries and source->target matrices."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_metric_files(runs_root: Path) -> list[Path]:
    return sorted((runs_root / "source").glob("*/*/metrics_eval_source.json"))


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x * 100.0:.3f}"


def _canon_domain(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def collect_rows(files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in files:
        # Expected layout: runs/source/<dataset>/<source>/metrics_eval_source.json
        if len(p.parts) < 5:
            continue
        dataset = p.parts[-3]
        source = p.parts[-2]
        payload = _read_json(p)
        metrics = payload.get("metrics", {})
        for split_name, split_metrics in metrics.items():
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "eval_split": split_name,
                    "acc_top1": _safe_float(split_metrics.get("acc_top1", None)),
                    "mean_acc": _safe_float(split_metrics.get("mean_acc", None)),
                    "ckpt_path": payload.get("ckpt_path", ""),
                    "metrics_file": str(p),
                }
            )
    return rows


def write_long_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "source", "eval_split", "acc_top1", "mean_acc", "ckpt_path", "metrics_file"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "acc_top1": _fmt_pct(r["acc_top1"]),
                    "mean_acc": _fmt_pct(r["mean_acc"]),
                }
            )


def write_target_matrices(rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # dataset -> canon_source -> canon_target -> acc_top1
    cube: dict[str, dict[str, dict[str, float | None]]] = defaultdict(lambda: defaultdict(dict))
    source_val_map: dict[str, dict[str, float | None]] = defaultdict(dict)
    sources_by_ds: dict[str, set[str]] = defaultdict(set)
    targets_by_ds: dict[str, set[str]] = defaultdict(set)
    # canonical domain -> display name
    display_by_ds: dict[str, dict[str, str]] = defaultdict(dict)

    for r in rows:
        split = str(r["eval_split"])
        dataset = str(r["dataset"])
        source_raw = str(r["source"])
        source = _canon_domain(source_raw)
        display_by_ds[dataset].setdefault(source, source_raw)
        if split == "source_val":
            source_val_map[dataset][source] = r["acc_top1"]
            sources_by_ds[dataset].add(source)
            continue
        if not split.startswith("target_test:"):
            continue
        target_raw = split.split("target_test:", 1)[1]
        target = _canon_domain(target_raw)
        display_by_ds[dataset].setdefault(target, target_raw)
        cube[dataset][source][target] = r["acc_top1"]
        sources_by_ds[dataset].add(source)
        targets_by_ds[dataset].add(target)

    written: list[Path] = []
    for dataset in sorted(cube.keys()):
        sources = sorted(sources_by_ds[dataset])
        # Matrix columns include all known target domains plus source domains
        # so source_val can be placed on diagonal (S -> S).
        targets = sorted(targets_by_ds[dataset] | sources_by_ds[dataset])
        labels = display_by_ds[dataset]
        out_csv = out_dir / f"{dataset}_source_to_target_top1.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_domain", *[labels.get(t, t) for t in targets]])
            for src in sources:
                vals = []
                for tgt in targets:
                    # Put source_val on diagonal source->source.
                    if tgt == src:
                        v = source_val_map.get(dataset, {}).get(src)
                    else:
                        v = cube[dataset].get(src, {}).get(tgt)
                    vals.append(_fmt_pct(v))
                w.writerow([labels.get(src, src), *vals])
        written.append(out_csv)
    return written


def write_source_val_summary(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    source_rows = [r for r in rows if str(r["eval_split"]) == "source_val"]
    source_rows.sort(key=lambda r: (r["dataset"], r["source"]))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "source_domain", "source_val_acc_top1", "source_val_mean_acc"])
        for r in source_rows:
            a1 = _fmt_pct(r["acc_top1"])
            ma = _fmt_pct(r["mean_acc"])
            w.writerow([r["dataset"], r["source"], a1, ma])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs", help="Root directory that contains runs/source")
    parser.add_argument("--out-dir", default="runs/source/summary", help="Directory to write CSV summaries")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    files = _find_metric_files(runs_root)
    if not files:
        raise SystemExit(f"No metrics_eval_source.json files found under: {runs_root / 'source'}")

    rows = collect_rows(files)
    write_long_csv(rows, out_dir / "source_eval_results_long.csv")
    write_source_val_summary(rows, out_dir / "source_eval_source_val.csv")
    matrix_files = write_target_matrices(rows, out_dir)

    print(f"Loaded {len(files)} metrics files.")
    print(f"Wrote: {out_dir / 'source_eval_results_long.csv'}")
    print(f"Wrote: {out_dir / 'source_eval_source_val.csv'}")
    for p in matrix_files:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
