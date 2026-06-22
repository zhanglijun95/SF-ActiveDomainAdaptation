#!/usr/bin/env python3
"""Generate BDD100K LPU tuned pure/+random/plugin configs and launcher."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from generate_cross_baseline_negative_plugins import METHODS


SEEDS = (42, 43, 44)
PURE_EXTRA_SEEDS = (43, 44)
BASELINE = "lpu_daod"
DATASET = "cityscapes_to_bdd100k"
DETECTOR = "dino"
LABEL_RATIO = "005"
PST_WEIGHT = 10.0
LSCL_WEIGHT = 2.5
TUNED_SLUG = "pst10_lscl2p5"
OUT_SCRIPT = Path(
    "runs/negative_results_summary/"
    "wave6n_bdd100k_lpu_tuned_random_plugins_35jobs_sagemaker.commands.sh"
)
OUT_MANIFEST = Path(
    "runs/negative_results_summary/"
    "wave6n_bdd100k_lpu_tuned_random_plugins_manifest.md"
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


def _pure_path(seed: int) -> Path:
    return Path(f"configs/baselines/{BASELINE}/{DATASET}_{DETECTOR}_bddfix_{TUNED_SLUG}_seed{seed}.yaml")


def _random_path(seed: int) -> Path:
    return Path(
        f"configs/baselines/{BASELINE}/{DATASET}_{DETECTOR}_random_bddfix_{TUNED_SLUG}_"
        f"budget{LABEL_RATIO}_seed{seed}.yaml"
    )


def _plugin_path(suffix: str, seed: int) -> Path:
    return Path(
        f"configs/baselines/{BASELINE}/{DATASET}_{DETECTOR}_random_bddfix_{TUNED_SLUG}_{suffix}_"
        f"budget{LABEL_RATIO}_seed{seed}.yaml"
    )


def _pure_exp_name(seed: int) -> str:
    return f"{BASELINE}_{DATASET}_{DETECTOR}_bddfix_{TUNED_SLUG}_seed{seed}"


def _random_exp_name(seed: int, suffix: str | None = None) -> str:
    middle = f"random_bddfix_{TUNED_SLUG}" if suffix is None else f"random_bddfix_{TUNED_SLUG}_{suffix}"
    return f"{BASELINE}_{DATASET}_{DETECTOR}_{middle}_budget{LABEL_RATIO}_seed{seed}"


def _config_from_base(seed: int, exp_name: str, *, active_enabled: bool) -> dict[str, Any]:
    payload = _load_yaml(_base_path(seed))
    payload["seed"] = int(seed)
    method = payload.setdefault("method", {})
    if not isinstance(method, dict):
        raise ValueError(f"Expected method mapping in {_base_path(seed)}")
    method["exp_name"] = exp_name
    active = method.setdefault("active", {})
    if not isinstance(active, dict):
        raise ValueError(f"Expected method.active mapping in {_base_path(seed)}")
    active["enabled"] = bool(active_enabled)
    active["strategy"] = "random"
    active["budget_total"] = 0.05
    active["supervised_weight"] = 1.0
    lpu = method.setdefault("lpu", {})
    if not isinstance(lpu, dict):
        raise ValueError(f"Expected method.lpu mapping in {_base_path(seed)}")
    lpu["pst_weight"] = PST_WEIGHT
    lpu["lscl_weight"] = LSCL_WEIGHT
    return payload


def _token(method: dict[str, Any] | str | None, seed: int) -> str:
    if method == "pure":
        return f"bdd-lpu-{TUNED_SLUG.replace('_', '-')}-pure-s{seed}"
    if method is None:
        return f"bdd-lpu-{TUNED_SLUG.replace('_', '-')}-rand005-s{seed}"
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
    return f"bdd-lpu-{TUNED_SLUG.replace('_', '-')}-{replacements[suffix]}-s{seed}"


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
        "MAX_PARALLEL=35",
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
        f"# LPU tuned setting: pst_weight={PST_WEIGHT}, lscl_weight={LSCL_WEIGHT}.",
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
            'echo "Submitted BDD100K LPU tuned pure/random/plugin jobs."',
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
                "# Wave 6N: BDD100K LPU Tuned Pure/Random/Plugin Jobs",
                "",
                f"Tuned LPU setting: `pst_weight={PST_WEIGHT}`, `lscl_weight={LSCL_WEIGHT}`.",
                "This keeps the LPU auxiliary losses nonzero while using the best nonzero seed-42 pilot.",
                "",
                "Contains 35 jobs: 2 extra pure seeds, 3 `+5% random` anchors, and 30 plugin jobs.",
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
    for seed in PURE_EXTRA_SEEDS:
        config = _pure_path(seed)
        _write_yaml(config, _config_from_base(seed, _pure_exp_name(seed), active_enabled=False))
        jobs.append((config, _token("pure", seed)))

    for seed in SEEDS:
        config = _random_path(seed)
        _write_yaml(config, _config_from_base(seed, _random_exp_name(seed), active_enabled=True))
        jobs.append((config, _token(None, seed)))

    for method in METHODS:
        suffix = str(method["config_suffix"])
        for seed in SEEDS:
            config = _plugin_path(suffix, seed)
            payload = _config_from_base(seed, _random_exp_name(seed, suffix), active_enabled=True)
            method_cfg = payload["method"]
            for key, value in deepcopy(method["block"]).items():
                method_cfg[key] = value
            _write_yaml(config, payload)
            jobs.append((config, _token(method, seed)))

    _write_launcher(jobs)
    _write_manifest(jobs)
    print(f"[bdd-lpu-tuned-generator] wrote_configs={len(jobs)} launcher={OUT_SCRIPT}")


if __name__ == "__main__":
    main()
