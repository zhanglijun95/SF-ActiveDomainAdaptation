#!/usr/bin/env python3
"""Generate BDD100K LPLD +5% random/plugin configs and SageMaker launcher."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from generate_cross_baseline_negative_plugins import METHODS


SEEDS = (42, 43, 44)
BASELINE = "lpld_daod"
DATASET = "cityscapes_to_bdd100k"
DETECTOR = "dino"
LABEL_RATIO = "005"
OUT_SCRIPT = Path(
    "runs/negative_results_summary/"
    "wave6l_bdd100k_lpld_bddfix_random_plugins_33jobs_sagemaker.commands.sh"
)
OUT_MANIFEST = Path(
    "runs/negative_results_summary/"
    "wave6l_bdd100k_lpld_bddfix_random_plugins_manifest.md"
)


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


def _base_path(seed: int) -> Path:
    return Path(f"configs/baselines/{BASELINE}/{DATASET}_{DETECTOR}_bddfix_seed{seed}.yaml")


def _random_path(seed: int) -> Path:
    return Path(
        f"configs/baselines/{BASELINE}/{DATASET}_{DETECTOR}_random_bddfix_"
        f"budget{LABEL_RATIO}_seed{seed}.yaml"
    )


def _plugin_path(suffix: str, seed: int) -> Path:
    return Path(
        f"configs/baselines/{BASELINE}/{DATASET}_{DETECTOR}_random_bddfix_{suffix}_"
        f"budget{LABEL_RATIO}_seed{seed}.yaml"
    )


def _exp_name(seed: int, suffix: str | None = None) -> str:
    middle = "random_bddfix" if suffix is None else f"random_bddfix_{suffix}"
    return f"{BASELINE}_{DATASET}_{DETECTOR}_{middle}_budget{LABEL_RATIO}_seed{seed}"


def _config_from_base(seed: int, exp_name: str) -> dict[str, Any]:
    payload = _load_yaml(_base_path(seed))
    payload["seed"] = int(seed)
    method = payload.setdefault("method", {})
    if not isinstance(method, dict):
        raise ValueError(f"Expected method mapping in {_base_path(seed)}")
    method["exp_name"] = exp_name
    active = method.setdefault("active", {})
    if not isinstance(active, dict):
        raise ValueError(f"Expected method.active mapping in {_base_path(seed)}")
    active["enabled"] = True
    active["strategy"] = "random"
    active["budget_total"] = 0.05
    active["supervised_weight"] = 1.0
    return payload


def _token(method: dict[str, Any] | None, seed: int) -> str:
    if method is None:
        return f"bdd-lpld-rand005-fix-s{seed}"
    suffix = str(method["config_suffix"])
    replacements = {
        "selection_threshold_calibration": "sel-thrcal",
        "selection_threshold_mapping": "sel-thrmap",
        "selection_pseudo_score_reweight": "sel-reweight",
        "query_recovery_scorer": "qrec-scorer",
        "query_recovery_multiview": "qrec-mv",
        "query_revival_scorer_foreground": "qrev-scorer",
        "query_revival_multiview_foreground": "qrev-mv",
        "control_target_anchored_pcgrad_pseudo_only": "ctrl-pcgrad",
        "control_sparse_loss_balance": "ctrl-balance",
        "control_label_guided_aema": "ctrl-aema",
    }
    return f"bdd-lpld-{replacements[suffix]}-s{seed}"


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
        "MAX_PARALLEL=33",
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
        "# Corrected LPLD +5% random anchor: BDD-fix hyperparameters + student early stop.",
    ]
    for config, token in jobs[:3]:
        lines.append(f"submit_job {config} {token}")
    lines.append("")
    lines.append("# LPLD plugin sweeps under the same BDD-fix +5% random anchor.")
    for config, token in jobs[3:]:
        lines.append(f"submit_job {config} {token}")
    lines.extend(
        [
            "",
            'while [ "$(pid_count)" -gt 0 ]; do',
            "    prune_finished",
            "    sleep 30",
            "done",
            "",
            'echo "Submitted all 33 BDD100K LPLD bddfix + random/plugin jobs."',
            "",
        ]
    )
    OUT_SCRIPT.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(jobs: list[tuple[Path, str]]) -> None:
    rows = ["| # | config | token |", "|---:|---|---|"]
    for idx, (config, token) in enumerate(jobs, start=1):
        rows.append(f"| {idx} | `{config}` | `{token}` |")
    OUT_MANIFEST.write_text(
        "\n".join(
            [
                "# Wave 6L: BDD100K LPLD BDD-Fix + Random/Plugin Jobs",
                "",
                "This batch contains 33 jobs: 3 corrected `+5% random` anchors and 30 plugin jobs.",
                "All configs inherit the BDD-fix LPLD hyperparameters and student intermediate early stopping.",
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
    for seed in SEEDS:
        config = _random_path(seed)
        _write_yaml(config, _config_from_base(seed, _exp_name(seed)))
        jobs.append((config, _token(None, seed)))

    for method in METHODS:
        suffix = str(method["config_suffix"])
        for seed in SEEDS:
            config = _plugin_path(suffix, seed)
            payload = _config_from_base(seed, _exp_name(seed, suffix))
            method_cfg = payload["method"]
            for key, value in deepcopy(method["block"]).items():
                method_cfg[key] = value
            _write_yaml(config, payload)
            jobs.append((config, _token(method, seed)))

    _write_launcher(jobs)
    _write_manifest(jobs)
    print(f"[bdd-lpld-generator] wrote_configs={len(jobs)} launcher={OUT_SCRIPT}")


if __name__ == "__main__":
    main()
