#!/usr/bin/env python3
"""Summarize registry gaps in the negative-results aggregate table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on caller env.
    yaml = None


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def _load_expected_registry_entries(registry_path: Path, statuses: set[str] | None) -> list[dict[str, Any]]:
    if yaml is None or not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    expected = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("config") or not entry.get("id"):
            continue
        if statuses is not None and str(entry.get("status", "")).strip() not in statuses:
            continue
        expected.append(entry)
    return expected


def _group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("sfod_baseline", ""),
        row.get("category", ""),
        row.get("enhancement", ""),
    )


def _summarize_group(rows: list[dict[str, str]]) -> dict[str, Any]:
    ap50_values = [_safe_float(row.get("AP50")) for row in rows]
    delta_values = [_safe_float(row.get("delta_AP50_vs_random")) for row in rows]
    ap50_values = [value for value in ap50_values if value is not None]
    delta_values = [value for value in delta_values if value is not None]
    best_row = max(rows, key=lambda row: _safe_float(row.get("AP50")) or float("-inf"))
    return {
        "count": len(rows),
        "best_exp_name": best_row.get("exp_name"),
        "best_AP50": _safe_float(best_row.get("AP50")),
        "best_delta_AP50_vs_random": _safe_float(best_row.get("delta_AP50_vs_random")),
        "min_AP50": min(ap50_values) if ap50_values else None,
        "max_AP50": max(ap50_values) if ap50_values else None,
        "min_delta_AP50_vs_random": min(delta_values) if delta_values else None,
        "max_delta_AP50_vs_random": max(delta_values) if delta_values else None,
    }


def audit(
    table: Path,
    out_dir: Path,
    baseline: str | None,
    registry: Path,
    expected_statuses: set[str] | None,
) -> tuple[Path, Path, dict[str, Any]]:
    rows = _read_rows(table)
    if baseline:
        rows = [row for row in rows if row.get("sfod_baseline") == baseline]

    registered = [row for row in rows if row.get("registry_id")]
    unregistered = [row for row in rows if not row.get("registry_id")]
    expected_entries = _load_expected_registry_entries(registry, expected_statuses)
    if baseline:
        expected_entries = [entry for entry in expected_entries if entry.get("sfod_baseline") == baseline]
    seen_ids = {row.get("registry_id") for row in registered if row.get("registry_id")}
    missing_registered = [entry for entry in expected_entries if entry.get("id") not in seen_ids]
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in unregistered:
        groups.setdefault(_group_key(row), []).append(row)

    group_summaries = []
    for (sfod_baseline, category, enhancement), group_rows in sorted(groups.items()):
        group_summaries.append(
            {
                "sfod_baseline": sfod_baseline,
                "category": category,
                "enhancement": enhancement,
                **_summarize_group(group_rows),
            }
        )

    payload = {
        "table": str(table),
        "baseline_filter": baseline,
        "total_rows": len(rows),
        "registered_rows": len(registered),
        "unregistered_rows": len(unregistered),
        "expected_registered_rows": len(expected_entries),
        "missing_registered_rows": len(missing_registered),
        "missing_registered": [
            {
                "id": entry.get("id"),
                "status": entry.get("status"),
                "wave": entry.get("wave"),
                "category": entry.get("category"),
                "sfod_baseline": entry.get("sfod_baseline"),
                "enhancement": entry.get("enhancement"),
                "seed": entry.get("seed"),
                "exp_name": entry.get("exp_name"),
                "config": entry.get("config"),
            }
            for entry in missing_registered
        ],
        "groups": group_summaries,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "registry_gap_audit.json"
    md_path = out_dir / "registry_gap_audit.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Registry Gap Audit\n\n")
        handle.write(f"- Table: `{table}`\n")
        handle.write(f"- Baseline filter: `{baseline or 'all'}`\n")
        handle.write(f"- Total rows: `{len(rows)}`\n")
        handle.write(f"- Registered rows: `{len(registered)}`\n")
        handle.write(f"- Unregistered rows: `{len(unregistered)}`\n\n")
        handle.write(f"- Expected registered rows: `{len(expected_entries)}`\n")
        handle.write(f"- Missing registered rows: `{len(missing_registered)}`\n\n")
        if missing_registered:
            handle.write("## Missing Registered Jobs\n\n")
            handle.write("| ID | Status | Wave | Baseline | Category | Enhancement | Seed | Config |\n")
            handle.write("| --- | --- | ---: | --- | --- | --- | ---: | --- |\n")
            for entry in missing_registered:
                handle.write(
                    "| "
                    f"{entry.get('id')} | "
                    f"{entry.get('status')} | "
                    f"{entry.get('wave')} | "
                    f"{entry.get('sfod_baseline')} | "
                    f"{entry.get('category')} | "
                    f"{entry.get('enhancement')} | "
                    f"{entry.get('seed')} | "
                    f"`{entry.get('config')}` |\n"
                )
            handle.write("\n")
        handle.write("## Unregistered Historical Runs\n\n")
        handle.write("| Baseline | Category | Enhancement | Count | Best AP50 | Best Delta | Best Run |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for group in group_summaries:
            handle.write(
                "| "
                f"{group['sfod_baseline']} | "
                f"{group['category']} | "
                f"{group['enhancement']} | "
                f"{group['count']} | "
                f"{_fmt(group['best_AP50'])} | "
                f"{_fmt(group['best_delta_AP50_vs_random'])} | "
                f"`{group['best_exp_name']}` |\n"
            )
    return json_path, md_path, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default="runs/negative_results_summary/negative_results_runs.csv",
        help="CSV produced by scripts/aggregate_negative_results.py.",
    )
    parser.add_argument("--registry", default="configs/negative_results/registry.yaml")
    parser.add_argument(
        "--expected-status",
        action="append",
        default=["ready", "diagnostic"],
        help="Registry status to count as expected; may be repeated. Use empty string to disable filtering.",
    )
    parser.add_argument("--out-dir", default="runs/negative_results_summary")
    parser.add_argument("--baseline", default="ddt_daod", help="Optional sfod_baseline filter; use empty string for all.")
    args = parser.parse_args()

    baseline = args.baseline.strip() or None
    expected_statuses = _as_set(args.expected_status)
    json_path, md_path, payload = audit(
        Path(args.table),
        Path(args.out_dir),
        baseline,
        Path(args.registry),
        expected_statuses,
    )
    print(f"[audit-negative-results] total={payload['total_rows']}")
    print(f"[audit-negative-results] expected_registered={payload['expected_registered_rows']}")
    print(f"[audit-negative-results] registered={payload['registered_rows']}")
    print(f"[audit-negative-results] missing_registered={payload['missing_registered_rows']}")
    print(f"[audit-negative-results] unregistered={payload['unregistered_rows']}")
    print(f"[audit-negative-results] json={json_path}")
    print(f"[audit-negative-results] markdown={md_path}")


if __name__ == "__main__":
    main()
