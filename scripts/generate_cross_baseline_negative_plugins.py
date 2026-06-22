#!/usr/bin/env python3
"""Generate 5% negative-result plugin configs for LPLD/LPU/PETS."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


BASELINES = {
    "lpld_daod": "lpld",
    "lpu_daod": "lpu",
    "pets_daod": "pets",
}
SEEDS = [42, 43, 44]
WAVE = 5
LABEL_RATIO = 0.05


def _label_guided(method: str, nested: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_guided": {
            "enabled": True,
            "method": method,
            "fit_max_images": 0,
            method: nested,
        }
    }


def _query_recovery_block(*, train_as: str, multiview: bool) -> dict[str, Any]:
    if train_as == "hard_pseudo" and not multiview:
        return {
            "query_recovery": {
                "enabled": True,
                "fit_max_images": 0,
                "min_score": 0.01,
                "below_threshold_only": True,
                "positive_iou": 0.5,
                "negative_iou": 0.3,
                "miss_iou": 0.5,
                "precision_floor": 0.55,
                "f_beta": 2.0,
                "min_class_positives": 3,
                "min_class_candidates": 20,
                "max_per_image": 8,
                "per_class_max": 4,
                "train_steps": 300,
                "lr": 0.05,
                "l2": 0.001,
                "max_pos_weight": 10.0,
                "max_negative_records": 30000,
                "multi_view": {
                    "enabled": False,
                    "views": 1,
                    "support_iou": 0.5,
                },
            }
        }
    if train_as == "hard_pseudo" and multiview:
        return {
            "query_recovery": {
                "enabled": True,
                "fit_max_images": 0,
                "min_score": 0.01,
                "below_threshold_only": False,
                "positive_iou": 0.5,
                "negative_iou": 0.3,
                "miss_iou": 0.5,
                "precision_floor": 0.50,
                "f_beta": 2.0,
                "min_class_positives": 3,
                "min_class_candidates": 20,
                "max_per_image": 10,
                "per_class_max": 5,
                "train_steps": 300,
                "lr": 0.05,
                "l2": 0.001,
                "max_pos_weight": 10.0,
                "max_negative_records": 30000,
                "multi_view": {
                    "enabled": True,
                    "views": 2,
                    "support_iou": 0.5,
                },
            }
        }

    if train_as != "revival_loss":
        raise ValueError(f"Unsupported query recovery train_as: {train_as}")

    if multiview:
        precision_floor = 0.50
        below_threshold_only = False
        min_recall = 0.02
        min_selected = 5
        budget_scale = 0.20
        views = 2
    else:
        precision_floor = 0.55
        below_threshold_only = True
        min_recall = 0.001
        min_selected = 1
        budget_scale = 0.25
        views = 1

    return {
        "query_recovery": {
            "enabled": True,
            "train_as": "revival_loss",
            "fit_max_images": 0,
            "min_score": 0.01,
            "below_threshold_only": below_threshold_only,
            "positive_iou": 0.5,
            "negative_iou": 0.3,
            "miss_iou": 0.5,
            "precision_floor": precision_floor,
            "f_beta": 2.0,
            "min_class_positives": 3,
            "min_class_candidates": 20,
            "max_per_image": 4,
            "per_class_max": 2,
            "train_steps": 300,
            "lr": 0.05,
            "l2": 0.001,
            "max_pos_weight": 10.0,
            "max_negative_records": 30000,
            "risk_gate": {
                "enabled": True,
                "min_precision": precision_floor,
                "min_recall": min_recall,
                "min_total_positive": 5,
                "min_selected": min_selected,
                "precision_power": 1.0,
                "recall_power": 0.5,
                "normalize": True,
                "gate_floor": 0.0,
                "gate_max": 1.0,
                "budget": {
                    "enabled": True,
                    "scale": budget_scale,
                    "min_budget": 0.02,
                    "max_budget": 0.75,
                },
            },
            "revival_loss": {
                "loss_weight": 0.02,
                "match_mode": "box_iou",
                "min_match_iou": 0.4,
                "match_class_aware": False,
                "positive_target": 0.7,
                "foreground_pool": "mean_logsumexp",
                "foreground_temperature": 1.0,
                "recovery_weight_power": 1.0,
                "min_candidate_weight": 0.1,
            },
            "multi_view": {
                "enabled": multiview,
                "views": views,
                "support_iou": 0.5,
            },
        }
    }


METHODS: list[dict[str, Any]] = [
    {
        "category": "selection",
        "role": "selection_score_threshold_calibration",
        "enhancement": "score_threshold_calibration",
        "id_slug": "selection_threshold_calibration",
        "config_suffix": "selection_threshold_calibration",
        "block": _label_guided(
            "score_threshold_calibration",
            {
                "match_iou": 0.5,
                "min_score": 0.01,
                "target_precision": 0.75,
                "min_selected": 2,
                "min_positives": 1,
                "min_threshold": 0.25,
                "max_threshold": 0.55,
                "max_delta_down": 0.10,
                "max_delta_up": 0.15,
            },
        ),
    },
    {
        "category": "selection",
        "role": "selection_threshold_mapping",
        "enhancement": "threshold_mapping",
        "id_slug": "selection_threshold_mapping",
        "config_suffix": "selection_threshold_mapping",
        "block": _label_guided(
            "threshold_mapping",
            {
                "min_score": 0.01,
                "smoothing": 1.0,
                "ratio_temperature": 0.75,
                "min_threshold": 0.25,
                "max_threshold": 0.55,
                "max_delta_down": 0.10,
                "max_delta_up": 0.10,
            },
        ),
    },
    {
        "category": "selection",
        "role": "selection_pseudo_score_reweight",
        "enhancement": "pseudo_score_reweight",
        "id_slug": "selection_pseudo_score_reweight",
        "config_suffix": "selection_pseudo_score_reweight",
        "block": _label_guided(
            "pseudo_score_reweight",
            {
                "match_iou": 0.5,
                "min_score": 0.01,
                "target_precision": 0.75,
                "min_candidates": 3,
                "min_positives": 1,
                "min_weight": 0.50,
                "max_weight": 1.00,
                "power": 1.0,
            },
        ),
    },
    {
        "category": "completion",
        "role": "completion_query_recovery_scorer",
        "enhancement": "query_recovery_scorer",
        "id_slug": "completion_query_recovery_scorer",
        "config_suffix": "query_recovery_scorer",
        "block": _query_recovery_block(train_as="hard_pseudo", multiview=False),
    },
    {
        "category": "completion",
        "role": "completion_query_recovery_multiview",
        "enhancement": "query_recovery_multiview",
        "id_slug": "completion_query_recovery_multiview",
        "config_suffix": "query_recovery_multiview",
        "block": _query_recovery_block(train_as="hard_pseudo", multiview=True),
    },
    {
        "category": "completion",
        "role": "completion_query_revival_scorer",
        "enhancement": "query_revival_scorer",
        "id_slug": "completion_query_revival_scorer",
        "config_suffix": "query_revival_scorer_foreground",
        "block": _query_recovery_block(train_as="revival_loss", multiview=False),
    },
    {
        "category": "completion",
        "role": "completion_query_revival_multiview",
        "enhancement": "query_revival_multiview",
        "id_slug": "completion_query_revival_multiview",
        "config_suffix": "query_revival_multiview_foreground",
        "block": _query_recovery_block(train_as="revival_loss", multiview=True),
    },
    {
        "category": "optimization_control",
        "role": "optimization_target_anchored_pcgrad",
        "enhancement": "target_anchored_pcgrad_pseudo_only",
        "id_slug": "control_target_anchored_pcgrad_pseudo_only",
        "config_suffix": "control_target_anchored_pcgrad_pseudo_only",
        "block": {
            "gradient_surgery": {
                "enabled": True,
                "method": "target_anchored_pcgrad",
                "apply_to_pseudo": True,
                "apply_to_masked": False,
                "eps": 1.0e-12,
            }
        },
    },
    {
        "category": "optimization_control",
        "role": "optimization_sparse_loss_balance",
        "enhancement": "sparse_loss_balance",
        "id_slug": "control_sparse_loss_balance",
        "config_suffix": "control_sparse_loss_balance",
        "block": _label_guided(
            "sparse_loss_balance",
            {
                "warmup_steps": 100,
                "ema_momentum": 0.95,
                "alpha": 0.5,
                "target_ratio": 1.0,
                "min_pseudo_scale": 0.5,
                "max_pseudo_scale": 1.5,
                "apply_to_masked": True,
            },
        ),
    },
    {
        "category": "optimization_control",
        "role": "optimization_label_guided_aema",
        "enhancement": "label_guided_aema",
        "id_slug": "control_label_guided_aema",
        "config_suffix": "control_label_guided_aema",
        "block": {
            "label_guided_aema": {
                "enabled": True,
                "merge": "max",
                "guidance_weight": 1.0,
                "normalize": True,
                "loss": "full",
                "supervised_loss_weight": 1.0,
                "top_fraction": 0.1,
                "adaptive_momentum": 0.997,
            }
        },
    },
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _base_config_path(baseline: str, seed: int) -> Path:
    return Path(
        f"configs/baselines/{baseline}/"
        f"cityscapes_to_foggy_cityscapes_dino_random_budget005_seed{seed}.yaml"
    )


def _new_config_path(baseline: str, method: dict[str, Any], seed: int) -> Path:
    return Path(
        f"configs/baselines/{baseline}/"
        f"cityscapes_to_foggy_cityscapes_dino_random_{method['config_suffix']}_budget005_seed{seed}.yaml"
    )


def _exp_name(baseline: str, method: dict[str, Any], seed: int) -> str:
    return (
        f"{baseline}_cityscapes_to_foggy_cityscapes_dino_random_"
        f"{method['config_suffix']}_budget005_seed{seed}"
    )


def _registry_id(baseline: str, method: dict[str, Any], seed: int) -> str:
    prefix = BASELINES[baseline]
    return f"{prefix}_b005_{method['id_slug']}_seed{seed}"


def _registry_entry(baseline: str, method: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "id": _registry_id(baseline, method, seed),
        "status": "ready",
        "wave": WAVE,
        "category": method["category"],
        "role": method["role"],
        "sfod_baseline": baseline,
        "enhancement": method["enhancement"],
        "seed": int(seed),
        "label_ratio": LABEL_RATIO,
        "exp_name": _exp_name(baseline, method, seed),
        "config": str(_new_config_path(baseline, method, seed)),
    }


def generate(*, write: bool, overwrite: bool) -> dict[str, Any]:
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    registry_path = Path("configs/negative_results/registry.yaml")
    registry = _load_yaml(registry_path)
    entries = registry.setdefault("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Registry field 'entries' must be a list")
    existing_ids = {str(entry.get("id", "")) for entry in entries if isinstance(entry, dict)}
    existing_configs = {str(entry.get("config", "")) for entry in entries if isinstance(entry, dict)}

    for baseline in BASELINES:
        for method in METHODS:
            for seed in SEEDS:
                base_path = _base_config_path(baseline, seed)
                config_path = _new_config_path(baseline, method, seed)
                entry = _registry_entry(baseline, method, seed)
                if not base_path.exists():
                    skipped.append({"id": entry["id"], "reason": "missing_base_config", "base_config": str(base_path)})
                    continue
                if entry["id"] in existing_ids:
                    skipped.append({"id": entry["id"], "reason": "registry_entry_exists"})
                    continue
                if str(config_path) in existing_configs:
                    skipped.append({"id": entry["id"], "reason": "registry_config_exists", "config": str(config_path)})
                    continue
                if config_path.exists() and not overwrite:
                    skipped.append({"id": entry["id"], "reason": "config_exists", "config": str(config_path)})
                    continue

                payload = _load_yaml(base_path)
                payload["seed"] = int(seed)
                method_cfg = payload.setdefault("method", {})
                if not isinstance(method_cfg, dict):
                    raise ValueError(f"Expected method mapping in {base_path}")
                method_cfg["exp_name"] = entry["exp_name"]
                for key, value in deepcopy(method["block"]).items():
                    method_cfg[key] = value

                if write:
                    _write_yaml(config_path, payload)
                    entries.append(entry)
                    existing_ids.add(entry["id"])
                    existing_configs.add(str(config_path))
                planned.append(entry)

    if write:
        _write_yaml(registry_path, registry)
    return {"planned": planned, "skipped": skipped, "write": bool(write)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write configs and registry entries.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing config files.")
    args = parser.parse_args()
    result = generate(write=bool(args.write), overwrite=bool(args.overwrite))
    mode = "write" if result["write"] else "dry-run"
    print(
        "[generate-cross-baseline-negative-plugins] "
        f"mode={mode} planned={len(result['planned'])} skipped={len(result['skipped'])}"
    )
    for entry in result["planned"][:12]:
        print(f"  plan {entry['id']} -> {entry['config']}")
    if len(result["planned"]) > 12:
        print(f"  ... {len(result['planned']) - 12} more")
    for item in result["skipped"][:12]:
        print(f"  skip {item}")
    if len(result["skipped"]) > 12:
        print(f"  ... {len(result['skipped']) - 12} more skipped")


if __name__ == "__main__":
    main()
