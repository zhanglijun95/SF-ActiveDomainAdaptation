#!/usr/bin/env python3
"""Generate a small BDD100K pure-LPU PST/LSCL weight search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SEED = 42
BASE_CONFIG = Path("configs/baselines/lpu_daod/cityscapes_to_bdd100k_dino_bddfix_seed42.yaml")
OUT_SCRIPT = Path("runs/negative_results_summary/wave6m_bdd100k_lpu_weight_search_sagemaker.commands.sh")
OUT_MANIFEST = Path("runs/negative_results_summary/wave6m_bdd100k_lpu_weight_search_manifest.md")

WEIGHT_GRID = [
    ("pst0_lscl0", 0.0, 0.0),
    ("pst5_lscl0p5", 5.0, 0.5),
    ("pst10_lscl1", 10.0, 1.0),
    ("pst10_lscl2p5", 10.0, 2.5),
    ("pst15_lscl1p5", 15.0, 1.5),
    ("pst25_lscl0p5", 25.0, 0.5),
    ("pst25_lscl1", 25.0, 1.0),
    ("pst25_lscl5", 25.0, 5.0),
    ("pst50_lscl1", 50.0, 1.0),
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _config_path(slug: str) -> Path:
    return Path(f"configs/baselines/lpu_daod/cityscapes_to_bdd100k_dino_bddfix_{slug}_seed{SEED}.yaml")


def _exp_name(slug: str) -> str:
    return f"lpu_daod_cityscapes_to_bdd100k_dino_bddfix_{slug}_seed{SEED}"


def _token(slug: str) -> str:
    return f"bdd-lpu-{slug.replace('_', '-')}-s{SEED}"


def _payload(slug: str, pst_weight: float, lscl_weight: float) -> dict[str, Any]:
    payload = _load_yaml(BASE_CONFIG)
    payload["seed"] = SEED
    method = payload.setdefault("method", {})
    if not isinstance(method, dict):
        raise ValueError(f"Expected method mapping in {BASE_CONFIG}")
    method["exp_name"] = _exp_name(slug)
    active = method.setdefault("active", {})
    if not isinstance(active, dict):
        raise ValueError(f"Expected method.active mapping in {BASE_CONFIG}")
    active["enabled"] = False
    lpu = method.setdefault("lpu", {})
    if not isinstance(lpu, dict):
        raise ValueError(f"Expected method.lpu mapping in {BASE_CONFIG}")
    lpu["pst_weight"] = float(pst_weight)
    lpu["lscl_weight"] = float(lscl_weight)
    return payload


def _write_launcher(jobs: list[tuple[Path, str]]) -> None:
    OUT_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "cd /home/ljzhang/code/SFADA",
        "",
        "IMAGE_URI=339713066702.dkr.ecr.us-west-2.amazonaws.com/sfada-daod:latest",
        "ROLE=arn:aws:iam::339713066702:role/VulcanStowSageMakerPipelineRole",
        "INSTANCE_TYPE=ml.g4dn.12xlarge",
        "S3_CITYSCAPES=s3://lijun-domainadaptation-sagemaker/data/cityscapes/",
        "S3_BDD100K=s3://lijun-domainadaptation-sagemaker/data/bdd100k/",
        "S3_SOURCE_CKPT=s3://lijun-domainadaptation-sagemaker/source_ckpt/cityscapes_to_foggy/",
        f"MAX_PARALLEL={len(jobs)}",
        "LAUNCH_STAGGER_SECONDS=20",
        'PIDS=""',
        "",
        'docker build -t "$IMAGE_URI" -f sagemaker/Dockerfile .',
        "aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 339713066702.dkr.ecr.us-west-2.amazonaws.com",
        'docker push "$IMAGE_URI"',
        "",
        "prune_finished() {",
        '    local live=""',
        "    local pid",
        "    for pid in $PIDS; do",
        '        if kill -0 "$pid" 2>/dev/null; then',
        '            live="$live $pid"',
        "        else",
        '            wait "$pid" 2>/dev/null || true',
        "        fi",
        "    done",
        '    PIDS="$live"',
        "}",
        "",
        "pid_count() {",
        "    set -- $PIDS",
        '    echo "$#"',
        "}",
        "",
        "wait_for_slot() {",
        "    while true; do",
        "        prune_finished",
        '        if [ "$(pid_count)" -lt "$MAX_PARALLEL" ]; then',
        "            break",
        "        fi",
        "        sleep 30",
        "    done",
        "}",
        "",
        "submit_job() {",
        '    local config="$1"',
        '    local token="$2"',
        "",
        "    wait_for_slot",
        "    /home/ljzhang/conda/envs/sfada/bin/python sagemaker/launch_sagemaker.py \\",
        '        --role "$ROLE" \\',
        '        --instance-type "$INSTANCE_TYPE" \\',
        '        --config "$config" \\',
        '        --job-name-token "$token" \\',
        '        --s3-data "$S3_CITYSCAPES" \\',
        '        --s3-target-data "$S3_BDD100K" \\',
        '        --s3-source-ckpt "$S3_SOURCE_CKPT" \\',
        "        --skip-build &",
        '    PIDS="$PIDS $!"',
        '    sleep "$LAUNCH_STAGGER_SECONDS"',
        "}",
        "",
        "# Pure LPU BDD PST/LSCL weight search, seed 42 only.",
    ]
    for config, token in jobs:
        lines.append(f"submit_job {config} {token}")
    lines.extend(
        [
            "",
            'while [ "$(pid_count)" -gt 0 ]; do',
            "    prune_finished",
            "    sleep 30",
            "done",
            "",
            'echo "Submitted all BDD100K pure-LPU PST/LSCL weight-search jobs."',
            "",
        ]
    )
    OUT_SCRIPT.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(jobs: list[tuple[Path, str]]) -> None:
    rows = ["| # | PST | LSCL | config | token |", "|---:|---:|---:|---|---|"]
    for idx, ((slug, pst, lscl), (config, token)) in enumerate(zip(WEIGHT_GRID, jobs), start=1):
        rows.append(f"| {idx} | `{pst:.3g}` | `{lscl:.3g}` | `{config}` | `{token}` |")
    OUT_MANIFEST.write_text(
        "\n".join(
            [
                "# Wave 6M: BDD100K Pure-LPU PST/LSCL Weight Search",
                "",
                "Seed-42-only search around the best current pure-LPU pilot.",
                "Everything except `lpu.pst_weight` and `lpu.lscl_weight` inherits the BDD-fix config.",
                "",
                "Launch:",
                "",
                "```bash",
                f"bash {OUT_SCRIPT}",
                "```",
                "",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    jobs: list[tuple[Path, str]] = []
    for slug, pst_weight, lscl_weight in WEIGHT_GRID:
        config = _config_path(slug)
        _write_yaml(config, _payload(slug, pst_weight, lscl_weight))
        jobs.append((config, _token(slug)))
    _write_launcher(jobs)
    _write_manifest(jobs)
    print(f"[bdd-lpu-weight-search] wrote_configs={len(jobs)} launcher={OUT_SCRIPT}")


if __name__ == "__main__":
    main()
