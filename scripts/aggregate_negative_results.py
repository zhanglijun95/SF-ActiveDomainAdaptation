#!/usr/bin/env python3
"""Aggregate DAOD negative-results summaries into CSV and JSON tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on the caller env.
    yaml = None


PER_CLASS_PREFIX = "AP-"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not available in this Python environment")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _maybe_load_config(run_dir: Path) -> dict[str, Any]:
    for name in ("resolved_config.yaml", "config.yaml"):
        path = run_dir / name
        if path.exists():
            try:
                return _load_yaml(path)
            except Exception:
                return {}
    return {}


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_context_from_path(summary_path: Path) -> dict[str, str | None]:
    parts = list(summary_path.parts)
    context: dict[str, str | None] = {
        "run_family": None,
        "sfod_baseline_from_path": None,
        "exp_name_from_path": None,
        "dataset_from_path": None,
        "detector_from_path": None,
    }
    if "baselines" in parts:
        idx = parts.index("baselines")
        context["run_family"] = "baselines"
        if idx + 1 < len(parts):
            context["sfod_baseline_from_path"] = parts[idx + 1]
        if idx + 2 < len(parts):
            context["exp_name_from_path"] = parts[idx + 2]
        if idx + 3 < len(parts):
            context["dataset_from_path"] = parts[idx + 3]
        if idx + 4 < len(parts):
            context["detector_from_path"] = parts[idx + 4]
    elif "daod_method" in parts:
        idx = parts.index("daod_method")
        context["run_family"] = "daod_method"
        if idx + 1 < len(parts):
            context["exp_name_from_path"] = parts[idx + 1]
        if idx + 2 < len(parts):
            context["dataset_from_path"] = parts[idx + 2]
        if idx + 3 < len(parts):
            context["detector_from_path"] = parts[idx + 3]
    elif "diagnostics" in parts:
        idx = parts.index("diagnostics")
        context["run_family"] = "diagnostics"
        if idx + 1 < len(parts):
            context["sfod_baseline_from_path"] = parts[idx + 1]
        if idx + 2 < len(parts):
            context["exp_name_from_path"] = parts[idx + 2]
    return context


def _split_dataset_label(label: Any) -> tuple[str | None, str | None]:
    if not isinstance(label, str) or not label:
        return None, None
    if "__to__" in label:
        source, target = label.split("__to__", 1)
        return source or None, target or None
    if "_to_" in label:
        source, target = label.split("_to_", 1)
        return source or None, target or None
    return None, None


def _registry_entries(registry_path: Path | None) -> dict[str, dict[str, Any]]:
    if registry_path is None or not registry_path.exists():
        return {}
    if yaml is None:
        print("[aggregate-negative-results] PyYAML unavailable; registry metadata will be skipped.")
        return {}
    registry = _load_yaml(registry_path)
    entries = registry.get("entries", [])
    if not isinstance(entries, list):
        return {}
    by_exp_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("exp_name", "id"):
            value = str(entry.get(key, "")).strip()
            if value:
                by_exp_name[value] = entry
    return by_exp_name


def _heuristic_category(exp_name: str, active_enabled: bool) -> tuple[str, str, str]:
    lower = exp_name.lower()
    if re.search(r"_random_budget\d+_seed\d+$", lower):
        return "baseline", "random_supervised_anchor", "random_supervised"
    if re.search(r"_dino_seed\d+$", lower):
        return "baseline", "pure_sfod", "none"
    if "oracle" in lower:
        return "oracle_diagnostic", "diagnostic", "oracle"
    if "query_recovery" in lower:
        return "completion", "realistic_method", "query_recovery"
    if "query_revival" in lower or "soft_query" in lower or "latent_query" in lower:
        return "completion", "realistic_method", "query_activation_or_revival"
    if "score_calibration" in lower or "label_rarity" in lower or "recal" in lower:
        return "selection", "realistic_method", "pseudo_label_selection"
    if (
        "pcgrad" in lower
        or "cagrad" in lower
        or "l2rw" in lower
        or "ldc" in lower
        or "selective" in lower
        or "label_guided_aema" in lower
        or "target_safe" in lower
        or "pseudo_bias" in lower
    ):
        return "optimization_control", "realistic_method", "optimization_control"
    if "replay" in lower or "crop" in lower or "aux_append" in lower:
        return "completion", "historical_method", "object_replay"
    if "random" in lower and active_enabled:
        return "baseline", "historical_method", "uncategorized_random_method"
    return "baseline", "pure_sfod", "none"


def _infer_seed(exp_name: str) -> int | None:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", exp_name)
    return int(match.group(1)) if match else None


def _infer_label_ratio(exp_name: str, active_enabled: bool) -> float | None:
    match = re.search(r"(?:^|_)budget(\d+)(?:_|$)", exp_name)
    if match:
        raw = match.group(1)
        return float(int(raw)) / float(10 ** len(raw))
    if active_enabled or "random" in exp_name.lower():
        return None
    return 0.0


def _metrics_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = _nested(summary, "final_target_val_metrics", "bbox", default=None)
    if metrics is None:
        metrics = _nested(summary, "teacher_target_val_metrics", "bbox", default=None)
    if metrics is None:
        metrics = _nested(summary, "student_target_val_metrics", "bbox", default=None)
    if metrics is None:
        metrics = _nested(summary, "target_val_metrics", "bbox", default=None)
    return metrics if isinstance(metrics, dict) else {}


def _extract_row(summary_path: Path, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_dir = summary_path.parent
    summary = _load_json(summary_path)
    config = _maybe_load_config(run_dir)
    context = _run_context_from_path(summary_path)

    method_cfg = config.get("method", {}) if isinstance(config.get("method", {}), dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    detector_cfg = config.get("detector", {}) if isinstance(config.get("detector", {}), dict) else {}
    active_cfg = method_cfg.get("active", {}) if isinstance(method_cfg.get("active", {}), dict) else {}

    exp_name = str(method_cfg.get("exp_name") or context.get("exp_name_from_path") or run_dir.name)
    registry_entry = registry.get(exp_name, {})
    active_enabled = bool(active_cfg.get("enabled", _nested(summary, "active_plan", "enabled", default=False)))
    category, role, enhancement = _heuristic_category(exp_name, active_enabled=active_enabled)
    inferred_source, inferred_target = _split_dataset_label(context.get("dataset_from_path"))

    metrics = _metrics_from_summary(summary)
    inferred_seed = _infer_seed(exp_name)
    inferred_label_ratio = _infer_label_ratio(exp_name, active_enabled=active_enabled)
    row: dict[str, Any] = {
        "registry_id": registry_entry.get("id"),
        "registry_status": registry_entry.get("status"),
        "wave": registry_entry.get("wave"),
        "category": registry_entry.get("category", category),
        "role": registry_entry.get("role", role),
        "enhancement": registry_entry.get("enhancement", enhancement),
        "sfod_baseline": registry_entry.get(
            "sfod_baseline",
            context.get("sfod_baseline_from_path") or summary.get("algorithm"),
        ),
        "exp_name": exp_name,
        "run_family": context.get("run_family"),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "source_domain": registry_entry.get("source_domain", data_cfg.get("source_domain", inferred_source)),
        "target_domain": registry_entry.get("target_domain", data_cfg.get("target_domain", inferred_target)),
        "dataset": registry_entry.get("dataset", context.get("dataset_from_path")),
        "detector": registry_entry.get(
            "detector",
            detector_cfg.get("model_name") or context.get("detector_from_path"),
        ),
        "seed": _safe_int(registry_entry.get("seed", config.get("seed", inferred_seed))),
        "label_ratio": _safe_float(
            registry_entry.get(
                "label_ratio",
                active_cfg.get(
                    "budget_total",
                    _nested(summary, "active_plan", "budget_total", default=inferred_label_ratio),
                ),
            )
        ),
        "label_strategy": active_cfg.get("strategy", _nested(summary, "active_plan", "strategy", default=None)),
        "selected_count": _safe_int(_nested(summary, "active_plan", "budget_k", default=None)),
        "target_total": _safe_int(_nested(summary, "active_plan", "target_total", default=None)),
        "epochs": _safe_int(summary.get("epochs")),
        "global_step": _safe_int(summary.get("global_step")),
        "final_model": summary.get("final_model"),
        "final_checkpoint": summary.get("final_checkpoint"),
        "AP": _safe_float(metrics.get("AP")),
        "AP50": _safe_float(metrics.get("AP50")),
        "AP75": _safe_float(metrics.get("AP75")),
        "APs": _safe_float(metrics.get("APs")),
        "APm": _safe_float(metrics.get("APm")),
        "APl": _safe_float(metrics.get("APl")),
    }

    for key, value in metrics.items():
        if isinstance(key, str) and key.startswith(PER_CLASS_PREFIX):
            row[key] = _safe_float(value)

    last_history = summary.get("history", [])[-1] if isinstance(summary.get("history"), list) and summary.get("history") else {}
    if isinstance(last_history, dict):
        row["pseudo_boxes_last_epoch"] = _safe_int(last_history.get("pseudo_boxes"))
        row["loss_total_last_epoch"] = _safe_float(last_history.get("loss_total"))
        row["loss_pseudo_last_epoch"] = _safe_float(last_history.get("loss_pseudo"))
        row["loss_supervised_last_epoch"] = _safe_float(last_history.get("loss_supervised"))
        oracle_stats = _nested(last_history, "oracle_pseudo", "stats", default={})
        if isinstance(oracle_stats, dict):
            row["oracle_input_pseudo"] = _safe_int(oracle_stats.get("input_pseudo"))
            row["oracle_kept"] = _safe_int(oracle_stats.get("kept"))
            row["oracle_dropped"] = _safe_int(oracle_stats.get("dropped"))
            row["oracle_recovered"] = _safe_int(oracle_stats.get("recovered"))
            row["oracle_output_pseudo"] = _safe_int(oracle_stats.get("output_pseudo"))

    label_guided = summary.get("label_guided_components", {})
    if isinstance(label_guided, dict):
        row["label_guided_enabled_components"] = ",".join(
            str(value) for value in label_guided.get("enabled_component_names", [])
        )
        row["label_guided_enabled_categories"] = ",".join(
            str(value) for value in label_guided.get("enabled_categories", [])
        )
        row["label_guided_has_legacy_live_prototype"] = bool(
            label_guided.get("has_legacy_live_prototype", False)
        )
        row["label_guided_legacy_components"] = ",".join(
            str(value) for value in label_guided.get("legacy_live_component_names", [])
        )

    return row


def _comparison_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sfod_baseline"),
        row.get("source_domain"),
        row.get("target_domain"),
        row.get("detector"),
        row.get("seed"),
        row.get("label_ratio"),
    )


def _attach_anchor_deltas(rows: list[dict[str, Any]]) -> None:
    anchors: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("role") == "random_supervised_anchor" and row.get("AP50") is not None:
            anchors[_comparison_key(row)] = row

    for row in rows:
        anchor = anchors.get(_comparison_key(row))
        if anchor is None:
            row["anchor_exp_name"] = None
            row["delta_AP50_vs_random"] = None
            row["delta_AP_vs_random"] = None
            continue
        row["anchor_exp_name"] = anchor.get("exp_name")
        row["delta_AP50_vs_random"] = (
            None if row.get("AP50") is None else float(row["AP50"]) - float(anchor["AP50"])
        )
        row["delta_AP_vs_random"] = (
            None if row.get("AP") is None or anchor.get("AP") is None else float(row["AP"]) - float(anchor["AP"])
        )


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for key in (
        "registry_id",
        "registry_status",
        "wave",
        "category",
        "role",
        "enhancement",
        "sfod_baseline",
        "exp_name",
        "source_domain",
        "target_domain",
        "dataset",
        "detector",
        "seed",
        "label_ratio",
        "AP50",
        "delta_AP50_vs_random",
        "AP",
        "delta_AP_vs_random",
        "AP75",
        "APs",
        "APm",
        "APl",
        "final_model",
        "global_step",
        "selected_count",
        "target_total",
        "anchor_exp_name",
        "run_dir",
    ):
        fieldnames.append(key)
    dynamic_keys = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(dynamic_keys)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(runs_root: Path, registry_path: Path | None, out_dir: Path) -> tuple[Path, Path, int]:
    registry = _registry_entries(registry_path)
    summary_paths = sorted(runs_root.rglob("summary.json"))
    rows = [_extract_row(path, registry) for path in summary_paths]
    _attach_anchor_deltas(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "negative_results_runs.json"
    csv_path = out_dir / "negative_results_runs.csv"
    _write_json(json_path, rows)
    _write_csv(csv_path, rows)
    return json_path, csv_path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/baselines", help="Root to scan for summary.json files.")
    parser.add_argument(
        "--registry",
        default="configs/negative_results/registry.yaml",
        help="Optional negative-results registry YAML.",
    )
    parser.add_argument("--out-dir", default="runs/negative_results_summary", help="Output directory.")
    args = parser.parse_args()

    registry_path = Path(args.registry) if args.registry else None
    json_path, csv_path, count = aggregate(Path(args.runs_root), registry_path, Path(args.out_dir))
    print(f"[aggregate-negative-results] rows={count}")
    print(f"[aggregate-negative-results] json={json_path}")
    print(f"[aggregate-negative-results] csv={csv_path}")


if __name__ == "__main__":
    main()
