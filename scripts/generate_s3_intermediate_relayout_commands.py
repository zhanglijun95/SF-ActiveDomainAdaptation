#!/usr/bin/env python3
"""Generate safe S3 commands to normalize job-scoped intermediate outputs.

Some SageMaker jobs may have written intermediate artifacts as:

    s3://bucket/.../intermediate/<job_name>/baselines/<baseline>/<exp>/...

This helper finds valid job-scoped ``summary.json`` files and generates
``aws s3 sync`` commands that copy each experiment folder back to the historical
layout:

    s3://bucket/.../intermediate/baselines/<baseline>/<exp>/...

Safety defaults:
- only copy sources whose summary has a valid AP50;
- do not overwrite destinations that already have a valid AP50 summary;
- write a dry-run shell script by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import boto3


METRIC_CONTAINERS = (
    "final_target_val_metrics",
    "teacher_target_val_metrics",
    "student_target_val_metrics",
    "target_val_metrics",
)


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    prefix: str


@dataclass(frozen=True)
class SummaryInfo:
    job_name: str
    baseline: str
    exp_name: str
    dataset: str
    detector: str
    source_summary_key: str
    source_root_key: str
    dest_summary_key: str
    dest_root_key: str
    ap50: float | None
    ap: float | None
    epochs: int | None
    global_step: int | None
    last_modified: str


def _parse_s3_uri(uri: str) -> S3Uri:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return S3Uri(bucket=parsed.netloc, prefix=prefix)


def _s3_join(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/")) + "/"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_summary(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        payload = json.loads(body)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    for name in METRIC_CONTAINERS:
        value = summary.get(name)
        if isinstance(value, dict) and isinstance(value.get("bbox"), dict):
            return value["bbox"]
    return {}


def _summary_numbers(summary: dict[str, Any]) -> tuple[float | None, float | None, int | None, int | None]:
    metrics = _metrics(summary)
    return (
        _safe_float(metrics.get("AP50")),
        _safe_float(metrics.get("AP")),
        _safe_int(summary.get("epochs")),
        _safe_int(summary.get("global_step")),
    )


def _relative_parts(key: str, intermediate_prefix: str) -> tuple[str, ...]:
    relative = key[len(intermediate_prefix) :] if key.startswith(intermediate_prefix) else key
    return PurePosixPath(relative).parts


def _parse_job_scoped_summary(
    *,
    key: str,
    last_modified: str,
    intermediate_prefix: str,
    s3: Any,
    bucket: str,
) -> SummaryInfo | None:
    parts = _relative_parts(key, intermediate_prefix)
    if "baselines" not in parts or not key.endswith("/summary.json"):
        return None
    baselines_idx = parts.index("baselines")
    if baselines_idx == 0:
        return None
    if baselines_idx + 5 >= len(parts):
        return None

    job_name = parts[0]
    baseline = parts[baselines_idx + 1]
    exp_name = parts[baselines_idx + 2]
    dataset = parts[baselines_idx + 3]
    detector = parts[baselines_idx + 4]
    source_root_key = key[: -len("summary.json")]
    dest_root_key = _s3_join(
        intermediate_prefix,
        "baselines",
        baseline,
        exp_name,
        dataset,
        detector,
    )
    dest_summary_key = f"{dest_root_key}summary.json"
    summary = _load_summary(s3, bucket, key)
    ap50, ap, epochs, global_step = _summary_numbers(summary)
    return SummaryInfo(
        job_name=job_name,
        baseline=baseline,
        exp_name=exp_name,
        dataset=dataset,
        detector=detector,
        source_summary_key=key,
        source_root_key=source_root_key,
        dest_summary_key=dest_summary_key,
        dest_root_key=dest_root_key,
        ap50=ap50,
        ap=ap,
        epochs=epochs,
        global_step=global_step,
        last_modified=last_modified,
    )


def _is_better(candidate: SummaryInfo, incumbent: SummaryInfo) -> bool:
    candidate_key = (
        1 if candidate.ap50 is not None else 0,
        candidate.global_step or -1,
        candidate.epochs or -1,
        candidate.last_modified,
        candidate.source_summary_key,
    )
    incumbent_key = (
        1 if incumbent.ap50 is not None else 0,
        incumbent.global_step or -1,
        incumbent.epochs or -1,
        incumbent.last_modified,
        incumbent.source_summary_key,
    )
    return candidate_key > incumbent_key


def _list_summary_objects(s3: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if key.endswith("/summary.json"):
                objects.append(item)
    return objects


def _dest_has_valid_summary(s3: Any, bucket: str, key: str) -> bool:
    summary = _load_summary(s3, bucket, key)
    ap50, _, _, _ = _summary_numbers(summary)
    return ap50 is not None


def _collect_candidates(
    *,
    s3: Any,
    intermediate: S3Uri,
    require_ap50: bool,
) -> dict[str, SummaryInfo]:
    selected: dict[str, SummaryInfo] = {}
    for item in _list_summary_objects(s3, intermediate.bucket, intermediate.prefix):
        key = str(item.get("Key", ""))
        info = _parse_job_scoped_summary(
            key=key,
            last_modified=str(item.get("LastModified", "")),
            intermediate_prefix=intermediate.prefix,
            s3=s3,
            bucket=intermediate.bucket,
        )
        if info is None:
            continue
        if require_ap50 and info.ap50 is None:
            continue
        if info.dest_summary_key not in selected or _is_better(info, selected[info.dest_summary_key]):
            selected[info.dest_summary_key] = info
    return selected


def _write_report(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "action",
        "job_name",
        "baseline",
        "exp_name",
        "dataset",
        "detector",
        "AP50",
        "AP",
        "epochs",
        "global_step",
        "source_uri",
        "dest_uri",
        "reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_command_file(path: str, commands: list[tuple[str, str]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Default is dry-run. Use DRY_RUN=0 bash this_file.sh to execute copies.",
        'DRY_RUN="${DRY_RUN:-1}"',
        'DRY_RUN_ARG=""',
        'if [ "$DRY_RUN" = "1" ]; then',
        '  DRY_RUN_ARG="--dryrun"',
        "fi",
        "",
    ]
    for source_uri, dest_uri in commands:
        lines.extend(
            [
                f"aws s3 sync {source_uri} {dest_uri} --only-show-errors $DRY_RUN_ARG",
                "",
            ]
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s3-intermediate",
        default="s3://lijun-domainadaptation-sagemaker/sagemaker-output/intermediate/",
        help="S3 intermediate root containing both shared and job-scoped folders.",
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Generate copy commands even when the shared destination already has a valid AP50 summary.",
    )
    parser.add_argument(
        "--allow-no-ap50",
        action="store_true",
        help="Allow copying job-scoped runs whose summary does not contain AP50.",
    )
    parser.add_argument(
        "--out",
        default="runs/negative_results_summary/relayout_job_scoped_intermediate_to_baselines.commands.sh",
        help="Output shell command file.",
    )
    parser.add_argument(
        "--report",
        default="runs/negative_results_summary/relayout_job_scoped_intermediate_to_baselines.tsv",
        help="Output TSV report.",
    )
    args = parser.parse_args()

    intermediate = _parse_s3_uri(args.s3_intermediate)
    s3 = boto3.client("s3", region_name=args.region)
    selected = _collect_candidates(
        s3=s3,
        intermediate=intermediate,
        require_ap50=not args.allow_no_ap50,
    )

    report_rows: list[dict[str, Any]] = []
    commands: list[tuple[str, str]] = []
    for info in sorted(selected.values(), key=lambda item: item.dest_summary_key):
        source_uri = f"s3://{intermediate.bucket}/{info.source_root_key}"
        dest_uri = f"s3://{intermediate.bucket}/{info.dest_root_key}"
        dest_valid = _dest_has_valid_summary(s3, intermediate.bucket, info.dest_summary_key)
        if dest_valid and not args.allow_overwrite:
            action = "skip"
            reason = "destination already has valid AP50 summary"
        else:
            action = "copy"
            reason = "destination missing valid AP50 summary" if not dest_valid else "overwrite allowed"
            commands.append((source_uri, dest_uri))
        report_rows.append(
            {
                "action": action,
                "job_name": info.job_name,
                "baseline": info.baseline,
                "exp_name": info.exp_name,
                "dataset": info.dataset,
                "detector": info.detector,
                "AP50": "" if info.ap50 is None else f"{info.ap50:.6f}",
                "AP": "" if info.ap is None else f"{info.ap:.6f}",
                "epochs": "" if info.epochs is None else str(info.epochs),
                "global_step": "" if info.global_step is None else str(info.global_step),
                "source_uri": source_uri,
                "dest_uri": dest_uri,
                "reason": reason,
            }
        )

    _write_report(args.report, report_rows)
    _write_command_file(args.out, commands)
    print(
        "[relayout-job-scoped] "
        f"job_scoped_valid={len(selected)} copy_commands={len(commands)} "
        f"skipped={len(report_rows) - len(commands)}"
    )
    print(f"[relayout-job-scoped] report={args.report}")
    print(f"[relayout-job-scoped] commands={args.out}")
    print("[relayout-job-scoped] generated script defaults to dry-run; execute with DRY_RUN=0 after review")


if __name__ == "__main__":
    main()
