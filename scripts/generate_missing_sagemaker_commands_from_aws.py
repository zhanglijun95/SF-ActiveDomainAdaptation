#!/usr/bin/env python3
"""Generate launch commands for registry jobs missing from SageMaker.

The launcher embeds a normalized registry ID and a config-path hash in each
SageMaker training job name. This script queries SageMaker for those name
fragments and writes a command file for entries that have no matching job.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import yaml

from generate_negative_results_commands import (
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_PYTHON,
    DEFAULT_ROLE,
    DEFAULT_S3_DATA,
    DEFAULT_S3_SOURCE_CKPT,
    _format_command_file,
    _launch_command,
)


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
    if not entry.get("config") or not entry.get("id"):
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


def _is_throttle(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    message = str(error.get("Message", ""))
    return code in {
        "ThrottlingException",
        "TooManyRequestsException",
        "RequestLimitExceeded",
        "ThrottledException",
    } or "Rate exceeded" in message


def _list_training_jobs_with_retry(
    client: Any,
    *,
    name_contains: str,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
) -> list[dict[str, Any]]:
    paginator = client.get_paginator("list_training_jobs")
    paginate_kwargs: dict[str, Any] = {
        "PaginationConfig": {
            "PageSize": 100,
        }
    }
    if name_contains:
        paginate_kwargs["NameContains"] = name_contains

    for attempt in range(1, int(max_attempts) + 1):
        try:
            jobs: list[dict[str, Any]] = []
            for page in paginator.paginate(**paginate_kwargs):
                jobs.extend(page.get("TrainingJobSummaries", []))
            return jobs
        except ClientError as exc:
            if attempt >= int(max_attempts) or not _is_throttle(exc):
                raise
            delay = min(float(max_delay), float(base_delay) * (2 ** (attempt - 1)))
            delay = random.uniform(delay * 0.5, delay * 1.5)
            print(
                "[missing-sagemaker-commands][retry] "
                f"ListTrainingJobs throttled; attempt={attempt}/{max_attempts}, "
                f"sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    return []


def _infer_list_name_contains(*, baselines: set[str] | None) -> str:
    if baselines and len(baselines) == 1:
        baseline = next(iter(baselines))
        return _job_part(baseline.split("_", 1)[0])
    # Multi-baseline jobs can have their human-readable prefix truncated by
    # SageMaker's 63-character job-name limit, so a broad filter such as
    # "sfada" may miss valid jobs. Use no NameContains filter by default.
    return ""


def _job_counts_by_status(jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        status = str(job.get("TrainingJobStatus", "Unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/negative_results/registry.yaml")
    parser.add_argument("--wave", action="append", help="Filter by wave; may be repeated.")
    parser.add_argument("--status", action="append", default=None, help="Filter by status; default: ready.")
    parser.add_argument("--category", action="append", help="Filter by category; may be repeated.")
    parser.add_argument("--baseline", action="append", help="Filter by sfod_baseline; may be repeated.")
    parser.add_argument("--label-ratio", action="append", help="Filter by label_ratio; may be repeated.")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--list-name-contains",
        default="",
        help=(
            "Broad SageMaker job-name filter used for one bulk ListTrainingJobs call. "
            "Default infers a single baseline token for one baseline, otherwise scans "
            "without NameContains to avoid missing truncated job names."
        ),
    )
    parser.add_argument("--list-retries", type=int, default=12)
    parser.add_argument("--list-retry-base-delay", type=float, default=20.0)
    parser.add_argument("--list-retry-max-delay", type=float, default=300.0)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--s3-data", default=DEFAULT_S3_DATA)
    parser.add_argument("--s3-source-ckpt", default=DEFAULT_S3_SOURCE_CKPT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--launch-stagger-seconds", type=float, default=10.0)
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Use rolling queue behavior in the generated missing-job command file.",
    )
    parser.add_argument("--out", required=True)
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

    name_contains = str(args.list_name_contains).strip() or _infer_list_name_contains(baselines=baselines)
    client = boto3.client(
        "sagemaker",
        region_name=args.region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    known_jobs = _list_training_jobs_with_retry(
        client,
        name_contains=name_contains,
        max_attempts=max(1, int(args.list_retries)),
        base_delay=max(1.0, float(args.list_retry_base_delay)),
        max_delay=max(1.0, float(args.list_retry_max_delay)),
    )
    valid_existing_statuses = {"Completed", "InProgress", "Stopping"}
    known_job_names = [
        str(job.get("TrainingJobName", ""))
        for job in known_jobs
        if str(job.get("TrainingJobName", ""))
        and str(job.get("TrainingJobStatus", "")) in valid_existing_statuses
    ]

    missing: list[dict[str, Any]] = []
    present: list[tuple[dict[str, Any], list[str]]] = []
    for entry in candidates:
        needle = _job_name_needle(entry)
        matching_names = [name for name in known_job_names if needle in name]
        if matching_names:
            present.append((entry, matching_names))
        else:
            missing.append(entry)

    commands = [
        "# "
        f"{entry.get('id', 'unknown')} | "
        f"wave={entry.get('wave')} | "
        f"status={entry.get('status')} | "
        f"category={entry.get('category')} | missing_from_sagemaker\n"
        + _launch_command(
            str(entry["config"]),
            job_name_token=str(entry.get("id", Path(str(entry["config"])).stem)),
            python_bin=args.python,
            role=args.role,
            instance_type=args.instance_type,
            s3_data=args.s3_data,
            s3_source_ckpt=args.s3_source_ckpt,
            skip_build=True,
        )
        for entry in missing
    ]
    text = _format_command_file(
        commands,
        max_parallel=max(0, int(args.max_parallel)),
        launch_stagger_seconds=max(0.0, float(args.launch_stagger_seconds)),
        rolling=bool(args.rolling),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(
        "[missing-sagemaker-commands] "
        f"name_contains={name_contains!r} listed_jobs={len(known_jobs)} "
        f"valid_existing={len(known_job_names)} status_counts={_job_counts_by_status(known_jobs)} "
        f"candidates={len(candidates)} present={len(present)} missing={len(missing)} out={out_path}"
    )
    for entry in missing:
        print(f"  missing {entry['id']} {entry['config']}")


if __name__ == "__main__":
    main()
