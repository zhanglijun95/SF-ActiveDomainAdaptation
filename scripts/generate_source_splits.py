#!/usr/bin/env python3
"""Generate deterministic source train/val split artifacts for SFADA datasets.

Outputs are saved under each dataset folder:
  <dataset_root>/splits/<source_name>/seed_<seed>/

Files:
  - source_train.txt        # rel_path label sample_id
  - source_val.txt          # rel_path label sample_id
  - source_train_ids.json
  - source_val_ids.json
  - meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Record:
    rel_path: str
    label: int
    sample_id: str


def _source_seed(base_seed: int, source_name: str) -> int:
    digest = hashlib.md5(source_name.encode("utf-8")).hexdigest()[:8]
    return base_seed + int(digest, 16)


def _parse_labeled_list(list_path: Path, sample_prefix: str) -> list[Record]:
    records: list[Record] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Expected '<rel_path> <label>' format: {list_path} :: {line}")
            rel_path = parts[0]
            label = int(parts[-1])
            sample_id = f"{sample_prefix}:{rel_path}"
            records.append(Record(rel_path=rel_path, label=label, sample_id=sample_id))
    if not records:
        raise ValueError(f"No records parsed from {list_path}")
    return records


def _split_records(records: list[Record], seed: int, val_ratio: float) -> tuple[list[Record], list[Record]]:
    """Stratified source train/val split by class label."""
    by_label: dict[int, list[Record]] = {}
    for rec in records:
        by_label.setdefault(rec.label, []).append(rec)

    rng = random.Random(seed)
    train_records: list[Record] = []
    val_records: list[Record] = []

    for label in sorted(by_label):
        bucket = list(by_label[label])
        rng.shuffle(bucket)

        # Keep a class presence in train; add at least one val sample when class has >1.
        n = len(bucket)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(n * val_ratio)
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)

        val_records.extend(bucket[:n_val])
        train_records.extend(bucket[n_val:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_split_list(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r.rel_path} {r.label} {r.sample_id}\n")


def _gen_for_source(
    dataset_name: str,
    source_name: str,
    list_path: Path,
    out_dir: Path,
    base_seed: int,
    val_ratio: float,
) -> dict:
    sample_prefix = f"{dataset_name}:{source_name}"
    records = _parse_labeled_list(list_path, sample_prefix=sample_prefix)

    seed = _source_seed(base_seed, f"{dataset_name}:{source_name}")
    train_records, val_records = _split_records(records, seed=seed, val_ratio=val_ratio)

    _write_split_list(out_dir / "source_train.txt", train_records)
    _write_split_list(out_dir / "source_val.txt", val_records)
    _write_json(out_dir / "source_train_ids.json", [r.sample_id for r in train_records])
    _write_json(out_dir / "source_val_ids.json", [r.sample_id for r in val_records])

    labels = sorted({r.label for r in records})
    per_class = {}
    for c in labels:
        total = sum(1 for r in records if r.label == c)
        n_train = sum(1 for r in train_records if r.label == c)
        n_val = sum(1 for r in val_records if r.label == c)
        per_class[str(c)] = {"total": total, "train": n_train, "val": n_val}

    meta = {
        "dataset": dataset_name,
        "source_name": source_name,
        "list_path": str(list_path),
        "seed_base": base_seed,
        "seed_effective": seed,
        "split_strategy": "stratified_by_label",
        "val_ratio": val_ratio,
        "num_total": len(records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_classes": len(labels),
        "per_class_counts": per_class,
        "files": {
            "source_train": str(out_dir / "source_train.txt"),
            "source_val": str(out_dir / "source_val.txt"),
            "source_train_ids": str(out_dir / "source_train_ids.json"),
            "source_val_ids": str(out_dir / "source_val_ids.json"),
        },
    }
    _write_json(out_dir / "meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/ljzhang/data/sfada")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    root = Path(args.data_root)
    manifests: list[dict] = []

    # office-31: each domain can be source
    off31 = root / "office-31"
    for domain in ["amazon", "dslr", "webcam"]:
        manifests.append(
            _gen_for_source(
                dataset_name="office-31",
                source_name=domain,
                list_path=off31 / domain / f"{domain}_list.txt",
                out_dir=off31 / "splits" / domain / f"seed_{args.seed}",
                base_seed=args.seed,
                val_ratio=args.val_ratio,
            )
        )

    # office-home: each domain can be source
    offhome = root / "office-home"
    for domain in ["Art", "Clipart", "Product", "Real World"]:
        manifests.append(
            _gen_for_source(
                dataset_name="office-home",
                source_name=domain,
                list_path=offhome / domain / f"{domain}_list.txt",
                out_dir=offhome / "splits" / domain / f"seed_{args.seed}",
                base_seed=args.seed,
                val_ratio=args.val_ratio,
            )
        )

    # visda-c: source is train
    visda = root / "visda-c"
    manifests.append(
        _gen_for_source(
            dataset_name="visda-c",
            source_name="train",
            list_path=visda / "train" / "sim_list.txt",
            out_dir=visda / "splits" / "train" / f"seed_{args.seed}",
            base_seed=args.seed,
            val_ratio=args.val_ratio,
        )
    )

    manifest_path = root / f"split_manifest_seed_{args.seed}.json"
    _write_json(manifest_path, manifests)
    print(f"Wrote split manifest: {manifest_path}")
    print(f"Generated {len(manifests)} source splits.")


if __name__ == "__main__":
    main()
