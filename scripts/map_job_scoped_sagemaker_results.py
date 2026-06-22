#!/usr/bin/env python3
"""Map job-scoped SageMaker intermediate folders back to experiment names.

This is a recovery helper for intermediate layouts like:

    <local-root>/<job_name>/baselines/<baseline>/<exp_name>/<dataset>/<detector>/summary.json

The job name itself is not meant to be human-readable. The reliable identity is
inside the synced path under ``baselines/<baseline>/<exp_name>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "final_target_val_metrics",
        "teacher_target_val_metrics",
        "student_target_val_metrics",
        "target_val_metrics",
    ):
        value = _nested(summary, key, "bbox")
        if isinstance(value, dict):
            return value
    return {}


def _parse_summary_path(local_root: Path, summary_path: Path) -> dict[str, str] | None:
    rel_parts = summary_path.relative_to(local_root).parts
    if "baselines" not in rel_parts:
        return None
    idx = rel_parts.index("baselines")
    if idx + 4 >= len(rel_parts):
        return None
    job_name = "<shared-intermediate>" if idx == 0 else rel_parts[0]
    baseline = rel_parts[idx + 1]
    exp_name = rel_parts[idx + 2]
    dataset = rel_parts[idx + 3]
    detector = rel_parts[idx + 4]
    return {
        "job_name": job_name,
        "baseline": baseline,
        "exp_name": exp_name,
        "dataset": dataset,
        "detector": detector,
    }


def scan(local_root: Path, s3_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(local_root.rglob("summary.json")):
        parsed = _parse_summary_path(local_root, summary_path)
        if parsed is None:
            continue
        summary = _load_json(summary_path)
        metrics = _metrics(summary)
        s3_baselines_uri = ""
        if parsed["job_name"] != "<shared-intermediate>":
            s3_baselines_uri = f"{s3_root.rstrip('/')}/{parsed['job_name']}/baselines/"
        rows.append(
            {
                **parsed,
                "AP50": _safe_float(metrics.get("AP50")),
                "AP": _safe_float(metrics.get("AP")),
                "epochs": _safe_int(summary.get("epochs")),
                "global_step": _safe_int(summary.get("global_step")),
                "summary_path": str(summary_path),
                "s3_baselines_uri": s3_baselines_uri,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_name",
        "baseline",
        "exp_name",
        "dataset",
        "detector",
        "AP50",
        "AP",
        "epochs",
        "global_step",
        "s3_baselines_uri",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_sync_script(path: Path, rows: list[dict[str, Any]], local_dest: str) -> None:
    """Write a review-first flattening script.

    The script is intentionally not executed by this helper. It syncs each
    unique job's ``baselines/`` subtree back into the historical local layout.
    Review the mapping TSV before running it.
    """

    uris = sorted({row["s3_baselines_uri"] for row in rows if row.get("s3_baselines_uri")})
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Review the TSV mapping before running this file.",
        "# It copies job-scoped intermediate baselines back into the old local layout.",
        "",
        f"mkdir -p {local_dest}",
        "",
    ]
    for uri in uris:
        lines.extend(
            [
                f"aws s3 sync {uri} {local_dest}/",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, help="Downloaded intermediate staging root.")
    parser.add_argument(
        "--s3-root",
        default="s3://lijun-domainadaptation-sagemaker/sagemaker-output/intermediate/",
        help="S3 intermediate root used to reconstruct sync commands.",
    )
    parser.add_argument(
        "--out-tsv",
        default="runs/negative_results_summary/job_scoped_intermediate_mapping.tsv",
        help="Output TSV mapping path.",
    )
    parser.add_argument(
        "--out-sync-script",
        default="runs/negative_results_summary/sync_job_scoped_baselines_to_runs.commands.sh",
        help="Optional review-first script for flattening full baselines into runs/baselines.",
    )
    parser.add_argument(
        "--local-dest",
        default="runs/baselines",
        help="Destination used in the generated flattening sync script.",
    )
    args = parser.parse_args()

    rows = scan(Path(args.local_root), args.s3_root)
    write_tsv(Path(args.out_tsv), rows)
    write_sync_script(Path(args.out_sync_script), rows, args.local_dest)

    unique_exp = {row["exp_name"] for row in rows}
    unique_job = {row["job_name"] for row in rows if row["job_name"] != "<shared-intermediate>"}
    print(f"[job-scoped-map] summaries={len(rows)} unique_exp={len(unique_exp)} unique_job={len(unique_job)}")
    print(f"[job-scoped-map] tsv={args.out_tsv}")
    print(f"[job-scoped-map] review_sync_script={args.out_sync_script}")


if __name__ == "__main__":
    main()
