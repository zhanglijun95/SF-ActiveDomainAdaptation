#!/usr/bin/env python3
"""Audit SageMaker jobs for duplicate negative-results registry entries.

The batch launcher embeds both a short registry-token and config-path hash in
each SageMaker job name. This script matches on that suffix instead of using
NameContains, because SageMaker's 63-character limit can truncate the readable
prefix (for example, "sfada" can become "s" or "sf").
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import boto3
import yaml


def _load_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _as_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def _normalized_ratio(value: Any) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _job_part(value: str) -> str:
    part = re.sub(r"[^a-z0-9-]+", "-", str(value).lower().replace("_", "-")).strip("-")
    return part or "job"


def _job_name_needle(entry: dict[str, Any]) -> str:
    token = _job_part(str(entry.get("id", "")))[:24]
    config_hash = hashlib.sha1(str(entry["config"]).encode("utf-8")).hexdigest()[:8]
    return f"{token}-{config_hash}"


def _matches(
    entry: dict[str, Any],
    *,
    waves: set[str] | None,
    statuses: set[str] | None,
    categories: set[str] | None,
    baselines: set[str] | None,
    label_ratios: set[str] | None,
) -> bool:
    if not entry.get("id") or not entry.get("config"):
        return False
    if waves is not None and str(entry.get("wave", "")).strip() not in waves:
        return False
    if statuses is not None and str(entry.get("status", "")).strip() not in statuses:
        return False
    if categories is not None and str(entry.get("category", "")).strip() not in categories:
        return False
    if baselines is not None and str(entry.get("sfod_baseline", "")).strip() not in baselines:
        return False
    if label_ratios is not None and _normalized_ratio(entry.get("label_ratio")) not in label_ratios:
        return False
    return True


def _list_jobs(client: Any, *, max_jobs: int) -> list[dict[str, Any]]:
    paginator = client.get_paginator("list_training_jobs")
    jobs: list[dict[str, Any]] = []
    for page in paginator.paginate(
        SortBy="CreationTime",
        SortOrder="Descending",
        PaginationConfig={"PageSize": 100},
    ):
        jobs.extend(page.get("TrainingJobSummaries", []))
        if len(jobs) >= int(max_jobs):
            return jobs[: int(max_jobs)]
    return jobs


def _created_value(job: dict[str, Any]) -> str:
    created = job.get("CreationTime", "")
    return created.isoformat() if hasattr(created, "isoformat") else str(created)


def _format_row(job: dict[str, Any]) -> str:
    return (
        f"{_created_value(job)} | "
        f"{job.get('TrainingJobStatus', ''):10s} | "
        f"{job.get('TrainingJobName', '')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/negative_results/registry.yaml")
    parser.add_argument("--wave", action="append", help="Filter by registry wave; may be repeated.")
    parser.add_argument("--status", action="append", default=None, help="Registry status; default: ready.")
    parser.add_argument("--category", action="append", help="Filter by category; may be repeated.")
    parser.add_argument("--baseline", action="append", help="Filter by sfod_baseline; may be repeated.")
    parser.add_argument("--label-ratio", action="append", help="Filter by label_ratio; may be repeated.")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=500,
        help="Number of recent SageMaker jobs to scan without a NameContains filter.",
    )
    parser.add_argument(
        "--stop-commands-out",
        default="",
        help="Optional path for generated aws stop-training-job commands for duplicate running jobs.",
    )
    args = parser.parse_args()

    registry = _load_registry(Path(args.registry))
    entries = registry.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Registry field 'entries' must be a list")

    waves = _as_set(args.wave)
    statuses = _as_set(args.status) or {"ready"}
    categories = _as_set(args.category)
    baselines = _as_set(args.baseline)
    label_ratios = _as_set(args.label_ratio)
    if label_ratios is not None:
        label_ratios = {_normalized_ratio(value) for value in label_ratios}

    candidates = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and _matches(
            entry,
            waves=waves,
            statuses=statuses,
            categories=categories,
            baselines=baselines,
            label_ratios=label_ratios,
        )
    ]
    needles = {_job_name_needle(entry): entry for entry in candidates}

    client = boto3.client("sagemaker", region_name=args.region)
    jobs = _list_jobs(client, max_jobs=max(1, int(args.max_jobs)))

    matched: dict[str, list[dict[str, Any]]] = {str(entry["id"]): [] for entry in candidates}
    for job in jobs:
        name = str(job.get("TrainingJobName", ""))
        for needle, entry in needles.items():
            if needle in name:
                matched[str(entry["id"])].append(job)
                break

    missing_ids = [entry_id for entry_id, items in matched.items() if not items]
    duplicate_items = {
        entry_id: sorted(items, key=lambda item: item.get("CreationTime"))
        for entry_id, items in matched.items()
        if len(items) > 1
    }

    print(
        "[audit-sagemaker-wave] "
        f"registry_entries={len(candidates)} scanned_jobs={len(jobs)} "
        f"matched_entries={sum(1 for items in matched.values() if items)} "
        f"missing_entries={len(missing_ids)} duplicate_entries={len(duplicate_items)}"
    )

    if missing_ids:
        print("\nMissing registry entries:")
        for entry_id in missing_ids:
            print(f"  {entry_id}")

    stop_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by {Path(__file__).name}; stops newer duplicate running SageMaker jobs only.",
    ]
    stop_count = 0
    if duplicate_items:
        print("\nDuplicate registry entries:")
        for entry_id, items in duplicate_items.items():
            print(f"\n{entry_id}")
            print("  keep:")
            print(f"    {_format_row(items[0])}")
            for job in items[1:]:
                print("  duplicate:")
                print(f"    {_format_row(job)}")
                if str(job.get("TrainingJobStatus", "")) == "InProgress":
                    stop_lines.append(
                        "aws sagemaker stop-training-job "
                        f"--region {args.region} "
                        f"--training-job-name {job.get('TrainingJobName')}"
                    )
                    stop_count += 1

    if args.stop_commands_out:
        out_path = Path(args.stop_commands_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if stop_count == 0:
            stop_lines.append("# No duplicate InProgress jobs found.")
        out_path.write_text("\n".join(stop_lines) + "\n", encoding="utf-8")
        print(f"\n[audit-sagemaker-wave] wrote stop commands: {out_path} ({stop_count} jobs)")


if __name__ == "__main__":
    main()
