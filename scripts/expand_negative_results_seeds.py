#!/usr/bin/env python3
"""Clone negative-results registry/config entries to additional random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = "configs/negative_results/registry.yaml"
DEFAULT_TARGET_SEEDS = [43, 44]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _as_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def _replace_seed_token(value: Any, *, source_seed: int, target_seed: int) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace(f"seed{int(source_seed)}", f"seed{int(target_seed)}")


def _matches(
    entry: dict[str, Any],
    *,
    source_seed: int,
    waves: set[str] | None,
    statuses: set[str] | None,
    categories: set[str] | None,
    baselines: set[str] | None,
) -> bool:
    if int(entry.get("seed", -1)) != int(source_seed):
        return False
    if not entry.get("config") or not entry.get("exp_name"):
        return False
    if waves is not None and str(entry.get("wave", "")).strip() not in waves:
        return False
    if statuses is not None and str(entry.get("status", "")).strip() not in statuses:
        return False
    if categories is not None and str(entry.get("category", "")).strip() not in categories:
        return False
    if baselines is not None and str(entry.get("sfod_baseline", "")).strip() not in baselines:
        return False
    return True


def _clone_entry(entry: dict[str, Any], *, source_seed: int, target_seed: int) -> dict[str, Any]:
    cloned = dict(entry)
    cloned["id"] = _replace_seed_token(entry.get("id"), source_seed=source_seed, target_seed=target_seed)
    cloned["seed"] = int(target_seed)
    cloned["exp_name"] = _replace_seed_token(
        entry.get("exp_name"),
        source_seed=source_seed,
        target_seed=target_seed,
    )
    cloned["config"] = _replace_seed_token(
        entry.get("config"),
        source_seed=source_seed,
        target_seed=target_seed,
    )
    cloned["source_entry_id"] = str(entry.get("id", ""))
    return cloned


def _clone_config(
    source_path: Path,
    target_path: Path,
    *,
    seed: int,
    exp_name: str,
    overwrite: bool,
    write: bool,
) -> str:
    existed = target_path.exists()
    if existed and not overwrite:
        return "reused_existing"
    payload = _load_yaml(source_path)
    payload["seed"] = int(seed)
    method_cfg = payload.setdefault("method", {})
    if not isinstance(method_cfg, dict):
        raise ValueError(f"Expected method mapping in {source_path}")
    method_cfg["exp_name"] = str(exp_name)
    if write:
        _write_yaml(target_path, payload)
    return "overwritten" if existed else "created"


def expand(args: argparse.Namespace) -> dict[str, Any]:
    registry_path = Path(args.registry)
    registry = _load_yaml(registry_path)
    entries = registry.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Registry field 'entries' must be a list")

    waves = _as_set(args.wave)
    statuses = _as_set(args.status)
    if statuses is None:
        statuses = {"ready", "diagnostic"}
    categories = _as_set(args.category)
    baselines = _as_set(args.baseline)
    target_seeds = [int(seed) for seed in (args.seed or DEFAULT_TARGET_SEEDS)]
    source_seed = int(args.source_seed)

    existing_ids = {str(entry.get("id", "")) for entry in entries if isinstance(entry, dict)}
    existing_exp_names = {str(entry.get("exp_name", "")) for entry in entries if isinstance(entry, dict)}
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    appended: list[dict[str, Any]] = []

    source_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and _matches(
            entry,
            source_seed=source_seed,
            waves=waves,
            statuses=statuses,
            categories=categories,
            baselines=baselines,
        )
    ]

    for entry in source_entries:
        source_config = Path(str(entry["config"]))
        if not source_config.exists():
            skipped.append(
                {
                    "source_id": entry.get("id"),
                    "reason": "missing_source_config",
                    "config": str(source_config),
                }
            )
            continue
        for target_seed in target_seeds:
            if target_seed == source_seed:
                continue
            cloned = _clone_entry(entry, source_seed=source_seed, target_seed=target_seed)
            clone_id = str(cloned.get("id", ""))
            clone_exp_name = str(cloned.get("exp_name", ""))
            if clone_id in existing_ids or clone_exp_name in existing_exp_names:
                skipped.append(
                    {
                        "source_id": entry.get("id"),
                        "target_seed": int(target_seed),
                        "reason": "registry_entry_exists",
                        "id": clone_id,
                        "exp_name": clone_exp_name,
                    }
                )
                continue
            config_status = _clone_config(
                source_config,
                Path(str(cloned["config"])),
                seed=target_seed,
                exp_name=clone_exp_name,
                overwrite=bool(args.overwrite),
                write=bool(args.write),
            )
            plan = {
                "source_id": entry.get("id"),
                "target_id": clone_id,
                "target_seed": int(target_seed),
                "category": cloned.get("category"),
                "sfod_baseline": cloned.get("sfod_baseline"),
                "config": cloned.get("config"),
                "config_status": config_status,
            }
            planned.append(plan)
            appended.append(cloned)
            existing_ids.add(clone_id)
            existing_exp_names.add(clone_exp_name)

    if args.write and appended:
        entries.extend(appended)
        out_registry = Path(args.out_registry) if args.out_registry else registry_path
        _write_yaml(out_registry, registry)

    return {
        "registry": str(registry_path),
        "write": bool(args.write),
        "source_seed": int(source_seed),
        "target_seeds": target_seeds,
        "source_entries": len(source_entries),
        "planned_clones": planned,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--out-registry", default="", help="Optional registry output path; default overwrites input with --write.")
    parser.add_argument("--source-seed", type=int, default=42)
    parser.add_argument("--seed", action="append", type=int, default=None, help="Target seed; may be repeated. Default: 43 and 44.")
    parser.add_argument("--wave", action="append", help="Filter by wave; may be repeated.")
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Filter by status; default: ready and diagnostic.",
    )
    parser.add_argument("--category", action="append", help="Filter by category; may be repeated.")
    parser.add_argument("--baseline", action="append", help="Filter by sfod_baseline; may be repeated.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cloned config files.")
    parser.add_argument("--write", action="store_true", help="Actually write configs and registry entries. Omit for dry run.")
    args = parser.parse_args()

    payload = expand(args)
    mode = "write" if payload["write"] else "dry-run"
    print(
        "[expand-negative-results-seeds] "
        f"mode={mode} source_entries={payload['source_entries']} "
        f"planned={len(payload['planned_clones'])} skipped={len(payload['skipped'])}"
    )
    for item in payload["planned_clones"]:
        print(
            "  plan "
            f"{item['source_id']} -> {item['target_id']} "
            f"seed={item['target_seed']} config_status={item['config_status']}"
        )
    for item in payload["skipped"]:
        print(f"  skip {item}")


if __name__ == "__main__":
    main()
