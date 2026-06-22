#!/usr/bin/env python3
"""Generate SageMaker launch commands from the negative-results registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROLE = "arn:aws:iam::339713066702:role/VulcanStowSageMakerPipelineRole"
DEFAULT_INSTANCE_TYPE = "ml.g4dn.12xlarge"
DEFAULT_S3_DATA = "s3://lijun-domainadaptation-sagemaker/data/cityscapes/"
DEFAULT_S3_SOURCE_CKPT = "s3://lijun-domainadaptation-sagemaker/source_ckpt/cityscapes_to_foggy/"
DEFAULT_PYTHON = "/home/ljzhang/conda/envs/sfada/bin/python"


def _load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected registry object in {path}")
    return payload


def _as_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def _matches(
    entry: dict[str, Any],
    *,
    entry_ids: set[str] | None,
    waves: set[str] | None,
    statuses: set[str] | None,
    categories: set[str] | None,
    baselines: set[str] | None,
    label_ratios: set[str] | None,
) -> bool:
    if entry_ids is not None and str(entry.get("id", "")).strip() not in entry_ids:
        return False
    if waves is not None and str(entry.get("wave", "")).strip() not in waves:
        return False
    if statuses is not None and str(entry.get("status", "")).strip() not in statuses:
        return False
    if categories is not None and str(entry.get("category", "")).strip() not in categories:
        return False
    if baselines is not None and str(entry.get("sfod_baseline", "")).strip() not in baselines:
        return False
    if label_ratios is not None:
        value = entry.get("label_ratio", None)
        if value is None:
            return False
        normalized = f"{float(value):.6f}".rstrip("0").rstrip(".")
        if normalized not in label_ratios:
            return False
    if not entry.get("config"):
        return False
    return True


def _launch_command(
    config: str,
    *,
    job_name_token: str,
    python_bin: str,
    role: str,
    instance_type: str,
    s3_data: str,
    s3_source_ckpt: str,
    skip_build: bool,
) -> str:
    lines = [
        f"{python_bin} sagemaker/launch_sagemaker.py \\",
        f"    --role {role} \\",
        f"    --instance-type {instance_type} \\",
        f"    --config {config} \\",
        f"    --job-name-token {job_name_token} \\",
        f"    --s3-data {s3_data} \\",
        f"    --s3-source-ckpt {s3_source_ckpt} \\",
    ]
    if skip_build:
        lines.append("    --skip-build &")
    else:
        lines[-1] = lines[-1].rstrip(" \\")
        lines.append("&")
    return "\n".join(lines)


def _format_command_file(
    commands: list[str],
    *,
    max_parallel: int,
    launch_stagger_seconds: float,
    rolling: bool,
) -> str:
    """Format launch commands, optionally batching local submissions."""

    if not commands:
        return ""
    if int(max_parallel) <= 0:
        return "\n\n".join(commands) + "\n"

    if rolling:
        poll_seconds = 30
        blocks = [
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "",
                    f"# Rolling queue: keeps at most {int(max_parallel)} SageMaker launch processes alive.",
                    "# launch_sagemaker.py uses blocking estimator.fit(), so each process stays alive",
                    "# until its SageMaker training job finishes; when one process finishes,",
                    "# the next command is submitted.",
                    "# This tracks launcher PIDs explicitly instead of using 'wait -n',",
                    "# so it works on Bash 4.2 and in non-interactive screen sessions.",
                    f"# Starts are staggered by {float(launch_stagger_seconds):.1f}s to avoid CreateTrainingJob API throttling.",
                    f"MAX_PARALLEL={int(max_parallel)}",
                    f"POLL_SECONDS={poll_seconds}",
                    "PIDS=\"\"",
                    "",
                    "prune_finished() {",
                    "    local live=\"\"",
                    "    local pid",
                    "    for pid in $PIDS; do",
                    "        if kill -0 \"$pid\" 2>/dev/null; then",
                    "            live=\"$live $pid\"",
                    "        else",
                    "            wait \"$pid\" 2>/dev/null || true",
                    "        fi",
                    "    done",
                    "    PIDS=\"$live\"",
                    "}",
                    "",
                    "pid_count() {",
                    "    set -- $PIDS",
                    "    echo \"$#\"",
                    "}",
                    "",
                    "wait_for_slot() {",
                    "    while true; do",
                    "        prune_finished",
                    "        if [ \"$(pid_count)\" -lt \"$MAX_PARALLEL\" ]; then",
                    "            break",
                    "        fi",
                    "        sleep \"$POLL_SECONDS\"",
                    "    done",
                    "}",
                ]
            )
        ]
        for idx, command in enumerate(commands, start=1):
            blocks.append("wait_for_slot")
            blocks.append(command)
            blocks.append("PIDS=\"$PIDS $!\"")
            if float(launch_stagger_seconds) > 0.0 and idx < len(commands):
                blocks.append(f"sleep {float(launch_stagger_seconds):.1f}")
        blocks.append("# Waiting for remaining launch processes.")
        blocks.append(
            "\n".join(
                [
                    "for pid in $PIDS; do",
                    "    wait \"$pid\" || true",
                    "done",
                ]
            )
        )
        return "\n\n".join(blocks) + "\n"

    blocks = [
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f"# Submits at most {int(max_parallel)} SageMaker launch processes at once.",
                "# launch_sagemaker.py uses blocking estimator.fit(), so each process stays alive",
                "# until its SageMaker training job finishes; wait barriers therefore also limit",
                "# concurrent SageMaker training jobs.",
                f"# Starts are staggered by {float(launch_stagger_seconds):.1f}s to avoid CreateTrainingJob API throttling.",
            ]
        )
    ]
    for idx, command in enumerate(commands, start=1):
        blocks.append(command)
        if float(launch_stagger_seconds) > 0.0 and idx < len(commands):
            blocks.append(f"sleep {float(launch_stagger_seconds):.1f}")
        if idx % int(max_parallel) == 0 and idx < len(commands):
            blocks.append(f"# Waiting after {idx} submitted launch processes.")
            blocks.append("wait")
    blocks.append("# Waiting for remaining launch processes.")
    blocks.append("wait")
    return "\n\n".join(blocks) + "\n"


def generate_commands(args: argparse.Namespace) -> list[str]:
    registry = _load_registry(Path(args.registry))
    entries = registry.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Registry field 'entries' must be a list")

    entry_ids = _as_set(args.entry_id)
    waves = _as_set(args.wave)
    statuses = _as_set(args.status)
    if statuses is None:
        statuses = {"ready"}
    categories = _as_set(args.category)
    baselines = _as_set(args.baseline)
    label_ratios = _as_set(args.label_ratio)
    if label_ratios is not None:
        label_ratios = {f"{float(value):.6f}".rstrip("0").rstrip(".") for value in label_ratios}
    commands = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _matches(
            entry,
            entry_ids=entry_ids,
            waves=waves,
            statuses=statuses,
            categories=categories,
            baselines=baselines,
            label_ratios=label_ratios,
        ):
            continue
        commands.append(
            "# "
            f"{entry.get('id', 'unknown')} | "
            f"wave={entry.get('wave')} | "
            f"status={entry.get('status')} | "
            f"category={entry.get('category')}\n"
            + _launch_command(
                str(entry["config"]),
                job_name_token=str(entry.get("id", Path(str(entry["config"])).stem)),
                python_bin=args.python,
                role=args.role,
                instance_type=args.instance_type,
                s3_data=args.s3_data,
                s3_source_ckpt=args.s3_source_ckpt,
                skip_build=not args.no_skip_build,
            )
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/negative_results/registry.yaml")
    parser.add_argument("--entry-id", action="append", help="Filter by exact registry id; may be repeated.")
    parser.add_argument("--wave", action="append", help="Filter by wave number; may be repeated.")
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Filter by registry status; default: ready. May be repeated.",
    )
    parser.add_argument("--category", action="append", help="Filter by category; may be repeated.")
    parser.add_argument("--baseline", action="append", help="Filter by sfod_baseline; may be repeated.")
    parser.add_argument("--label-ratio", action="append", help="Filter by exact label_ratio; may be repeated.")
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--s3-data", default=DEFAULT_S3_DATA)
    parser.add_argument("--s3-source-ckpt", default=DEFAULT_S3_SOURCE_CKPT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--no-skip-build", action="store_true", help="Do not add --skip-build to launch commands.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Insert wait barriers after this many backgrounded launcher processes. 0 means no barriers.",
    )
    parser.add_argument(
        "--launch-stagger-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds between starting launcher processes.",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Use a rolling queue: submit one new command whenever one background launcher finishes.",
    )
    parser.add_argument("--out", default="", help="Optional file path to write commands.")
    args = parser.parse_args()

    commands = generate_commands(args)
    text = _format_command_file(
        commands,
        max_parallel=max(0, int(args.max_parallel)),
        launch_stagger_seconds=max(0.0, float(args.launch_stagger_seconds)),
        rolling=bool(args.rolling),
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"[negative-results-commands] wrote {len(commands)} commands to {out_path} "
            f"(max_parallel={max(0, int(args.max_parallel))}, rolling={bool(args.rolling)})"
        )
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
