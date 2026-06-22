#!/usr/bin/env python3
"""Summarize Wave-5 LPLD/LPU/PETS plugin results.

The Wave-5 jobs may exist in two layouts:

1. The older shared SageMaker sync layout:
   runs/baselines/<baseline>/<exp_name>/...
2. The safer job-scoped layout:
   runs/<staging>/<sagemaker-job-name>/baselines/<baseline>/<exp_name>/...

This script scans one or more roots recursively, deduplicates by exp_name, and
reports only the registered Wave-5 plugin runs plus each baseline's pure/random
anchors. It is intentionally reporting-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - caller env issue.
    raise SystemExit("PyYAML is required. Use /home/ljzhang/conda/envs/sfada/bin/python.") from exc


BASELINES = ("lpld_daod", "lpu_daod", "pets_daod")
SEEDS = (42, 43, 44)

METHOD_ORDER = (
    "pure_sfod",
    "random_5pct",
    "selection/threshold_calibration",
    "selection/threshold_mapping",
    "selection/pseudo_score_reweight",
    "completion/query_recovery_scorer",
    "completion/query_recovery_multiview",
    "completion/query_revival_scorer",
    "completion/query_revival_multiview",
    "control/pcgrad_pseudo_only",
    "control/sparse_loss_balance",
    "control/label_guided_aema",
)

METHOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pure_sfod", re.compile(r"_dino_seed\d+$")),
    ("random_5pct", re.compile(r"_dino_random_budget005_seed\d+$")),
    ("selection/threshold_calibration", re.compile(r"_random_selection_threshold_calibration_budget005_seed\d+$")),
    ("selection/threshold_mapping", re.compile(r"_random_selection_threshold_mapping_budget005_seed\d+$")),
    ("selection/pseudo_score_reweight", re.compile(r"_random_selection_pseudo_score_reweight_budget005_seed\d+$")),
    ("completion/query_recovery_scorer", re.compile(r"_random_query_recovery_scorer_budget005_seed\d+$")),
    ("completion/query_recovery_multiview", re.compile(r"_random_query_recovery_multiview_budget005_seed\d+$")),
    ("completion/query_revival_scorer", re.compile(r"_random_query_revival_scorer_foreground_budget005_seed\d+$")),
    ("completion/query_revival_multiview", re.compile(r"_random_query_revival_multiview_foreground_budget005_seed\d+$")),
    ("control/pcgrad_pseudo_only", re.compile(r"_random_control_target_anchored_pcgrad_pseudo_only_budget005_seed\d+$")),
    ("control/sparse_loss_balance", re.compile(r"_random_control_sparse_loss_balance_budget005_seed\d+$")),
    ("control/label_guided_aema", re.compile(r"_random_control_label_guided_aema_budget005_seed\d+$")),
)


@dataclass(frozen=True)
class RunRow:
    exp_name: str
    sfod_baseline: str
    method: str
    seed: int | None
    ap50: float | None
    ap: float | None
    global_step: int | None
    epochs: int | None
    summary_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_from_path(summary_path: Path) -> tuple[str | None, str | None]:
    parts = list(summary_path.parts)
    if "baselines" not in parts:
        return None, None
    idx = parts.index("baselines")
    baseline = parts[idx + 1] if idx + 1 < len(parts) else None
    exp_name = parts[idx + 2] if idx + 2 < len(parts) else None
    return baseline, exp_name


def _method_from_exp(exp_name: str) -> str | None:
    for name, pattern in METHOD_PATTERNS:
        if pattern.search(exp_name):
            return name
    return None


def _seed_from_exp(exp_name: str) -> int | None:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", exp_name)
    return int(match.group(1)) if match else None


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "final_target_val_metrics",
        "teacher_target_val_metrics",
        "student_target_val_metrics",
        "target_val_metrics",
    ):
        value = summary.get(key)
        if isinstance(value, dict) and isinstance(value.get("bbox"), dict):
            return value["bbox"]
    return {}


def _row_from_summary(summary_path: Path) -> RunRow | None:
    run_dir = summary_path.parent
    summary = _load_json(summary_path)
    config = {}
    for name in ("resolved_config.yaml", "config.yaml"):
        path = run_dir / name
        if path.exists():
            config = _load_yaml(path)
            if config:
                break

    method_cfg = config.get("method", {}) if isinstance(config.get("method"), dict) else {}
    baseline_from_path, exp_from_path = _infer_from_path(summary_path)
    exp_name = str(method_cfg.get("exp_name") or exp_from_path or run_dir.name)
    sfod_baseline = str(baseline_from_path or summary.get("algorithm") or "")
    if sfod_baseline not in BASELINES:
        return None
    method = _method_from_exp(exp_name)
    if method is None:
        return None
    metrics = _metrics(summary)
    return RunRow(
        exp_name=exp_name,
        sfod_baseline=sfod_baseline,
        method=method,
        seed=_safe_int(config.get("seed")) or _seed_from_exp(exp_name),
        ap50=_safe_float(metrics.get("AP50")),
        ap=_safe_float(metrics.get("AP")),
        global_step=_safe_int(summary.get("global_step")),
        epochs=_safe_int(summary.get("epochs")),
        summary_path=summary_path,
    )


def _better(existing: RunRow, candidate: RunRow) -> RunRow:
    """Prefer complete, longer, then later path lexicographically for stability."""

    def key(row: RunRow) -> tuple[int, int, int, str]:
        return (
            1 if row.ap50 is not None else 0,
            row.global_step or -1,
            row.epochs or -1,
            str(row.summary_path),
        )

    return candidate if key(candidate) > key(existing) else existing


def _scan_roots(roots: list[Path]) -> tuple[dict[str, RunRow], dict[str, list[RunRow]]]:
    by_exp: dict[str, RunRow] = {}
    duplicates: dict[str, list[RunRow]] = defaultdict(list)
    for root in roots:
        for summary_path in sorted(root.rglob("summary.json")):
            row = _row_from_summary(summary_path)
            if row is None:
                continue
            duplicates[row.exp_name].append(row)
            if row.exp_name in by_exp:
                by_exp[row.exp_name] = _better(by_exp[row.exp_name], row)
            else:
                by_exp[row.exp_name] = row
    duplicates = {key: value for key, value in duplicates.items() if len(value) > 1}
    return by_exp, duplicates


def _wave5_expected(registry_path: Path) -> list[dict[str, Any]]:
    payload = _load_yaml(registry_path)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    prefixes = tuple(f"{baseline.removesuffix('_daod')}_b005_" for baseline in BASELINES)
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("id", "")).startswith(prefixes)
    ]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _summary_rows(by_exp: dict[str, RunRow]) -> list[dict[str, Any]]:
    anchors = {
        (row.sfod_baseline, row.seed): row.ap50
        for row in by_exp.values()
        if row.method == "random_5pct" and row.seed is not None and row.ap50 is not None
    }
    rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        for method in METHOD_ORDER:
            values = [
                row
                for row in by_exp.values()
                if row.sfod_baseline == baseline and row.method == method and row.ap50 is not None
            ]
            if not values:
                continue
            ap50s = [row.ap50 for row in values if row.ap50 is not None]
            aps = [row.ap for row in values if row.ap is not None]
            deltas: list[float] = []
            if method == "random_5pct":
                deltas = [0.0 for _ in values]
            elif method != "pure_sfod":
                for row in values:
                    anchor = anchors.get((row.sfod_baseline, row.seed))
                    if anchor is not None and row.ap50 is not None:
                        deltas.append(row.ap50 - anchor)
            rows.append(
                {
                    "sfod_baseline": baseline,
                    "method": method,
                    "n": len(values),
                    "AP50_mean": _mean(ap50s),
                    "AP50_std": _std(ap50s),
                    "delta_AP50_vs_random_mean": _mean(deltas),
                    "AP_mean": _mean(aps),
                }
            )
    return rows


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    duplicates: dict[str, list[RunRow]],
    roots: list[Path],
) -> None:
    present = len(expected) - len(missing)
    lines = [
        "# Wave 5 LPLD/LPU/PETS 5% Plugin Summary",
        "",
        f"- Scanned roots: {', '.join(str(root) for root in roots)}",
        f"- Expected registered plugin runs: {len(expected)}",
        f"- Present registered plugin runs: {present}",
        f"- Missing registered plugin runs: {len(missing)}",
        f"- Duplicate exp_names observed while scanning: {len(duplicates)}",
        "",
        "## Result Table",
        "",
        "| SFOD baseline | Method | n | AP50 mean | AP50 std | Delta AP50 vs 5% random | AP mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {sfod_baseline} | {method} | {n} | {ap50} | {ap50_std} | {delta} | {ap} |".format(
                sfod_baseline=row["sfod_baseline"],
                method=row["method"],
                n=row["n"],
                ap50=_fmt(row["AP50_mean"]),
                ap50_std=_fmt(row["AP50_std"]),
                delta=_fmt(row["delta_AP50_vs_random_mean"]),
                ap=_fmt(row["AP_mean"]),
            )
        )

    lines.extend(["", "## Missing Registered Plugin Runs", ""])
    if not missing:
        lines.append("None.")
    else:
        counts = Counter(str(entry.get("sfod_baseline")) for entry in missing)
        lines.append(", ".join(f"{key}: {value}" for key, value in sorted(counts.items())))
        lines.append("")
        for entry in missing:
            lines.append(f"- {entry.get('id')}: `{entry.get('exp_name')}`")

    lines.extend(["", "## Duplicate Notes", ""])
    if not duplicates:
        lines.append("No duplicate exp_names were found in the scanned roots.")
    else:
        lines.append("The script deduplicated by preferring complete summaries with larger `global_step`.")
        for exp_name, rows in sorted(duplicates.items()):
            lines.append(f"- `{exp_name}`: {len(rows)} summaries")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sfod_baseline",
        "method",
        "n",
        "AP50_mean",
        "AP50_std",
        "delta_AP50_vs_random_mean",
        "AP_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        action="append",
        default=[],
        help="Root to scan recursively. Can be passed multiple times.",
    )
    parser.add_argument(
        "--registry",
        default="configs/negative_results/registry.yaml",
        help="Negative-results registry YAML.",
    )
    parser.add_argument(
        "--out-md",
        default="runs/negative_results_summary/wave5_lpld_lpu_pets_5pct_summary.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--out-csv",
        default="runs/negative_results_summary/wave5_lpld_lpu_pets_5pct_summary.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    roots = [Path(root) for root in (args.runs_root or ["runs/baselines"])]
    by_exp, duplicates = _scan_roots(roots)
    expected = _wave5_expected(Path(args.registry))
    missing = [entry for entry in expected if str(entry.get("exp_name")) not in by_exp]
    rows = _summary_rows(by_exp)
    _write_markdown(Path(args.out_md), rows, expected, missing, duplicates, roots)
    _write_csv(Path(args.out_csv), rows)

    print(f"[wave5-summary] scanned_roots={len(roots)} unique_exp_names={len(by_exp)}")
    print(f"[wave5-summary] expected_plugins={len(expected)} present_plugins={len(expected)-len(missing)} missing_plugins={len(missing)}")
    if missing:
        print(f"[wave5-summary] missing_by_baseline={dict(Counter(str(entry.get('sfod_baseline')) for entry in missing))}")
    print(f"[wave5-summary] markdown={args.out_md}")
    print(f"[wave5-summary] csv={args.out_csv}")


if __name__ == "__main__":
    main()
