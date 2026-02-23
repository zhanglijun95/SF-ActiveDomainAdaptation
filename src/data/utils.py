"""Split utilities and dataloader builders."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .datasets import build_dataset
from .transforms import build_eval_transform, build_strong_transform, build_weak_transform
from .wrappers import IdFilteredDataset, LabelRouterDataset, TwoViewDataset


def make_source_split(
    source_full_ids: list[str], seed: int, val_ratio: float = 0.1
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    ids = list(source_full_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def save_ids(path: str | Path, ids: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)


def load_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(json.load(f))


def _loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_pretrain_loaders(
    cfg: Any,
    source_train_ids: list[str],
    source_val_ids: list[str],
) -> dict[str, DataLoader]:
    weak_tf = build_weak_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    source_domain = cfg.data.source_domain
    dataset_name = cfg.data.dataset_name
    base = build_dataset(
        cfg,
        dataset_name=dataset_name,
        split="source_train",
        domain=source_domain,
        transform=weak_tf,
        return_id=True,
    )

    train_ds = IdFilteredDataset(
        base,
        mode="labeled",
        queried_ids=set(source_train_ids),
        pseudo_ids=set(),
    )
    val_base = build_dataset(
        cfg,
        dataset_name=dataset_name,
        split="source_val",
        domain=source_domain,
        transform=eval_tf,
        return_id=True,
    )
    val_ds = IdFilteredDataset(
        val_base,
        mode="labeled",
        queried_ids=set(source_val_ids),
        pseudo_ids=set(),
    )

    batch_size = int(cfg.train.batch_size)
    workers = int(getattr(cfg.train, "num_workers", 4))
    return {
        "source_train": _loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
        "source_val": _loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
    }


def build_adapt_loaders(cfg: Any, round_state: Any) -> dict[str, DataLoader]:
    weak_tf = build_weak_transform(cfg)
    strong_tf = build_strong_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    dataset_name = cfg.data.dataset_name
    target_domain = cfg.data.target_domain
    source_domain = cfg.data.source_domain

    target_adapt_gt = build_dataset(
        cfg,
        dataset_name=dataset_name,
        split="target_adapt",
        domain=target_domain,
        transform=eval_tf,
        return_id=True,
    )
    target_test = build_dataset(
        cfg,
        dataset_name=dataset_name,
        split="target_test",
        domain=target_domain,
        transform=eval_tf,
        return_id=True,
    )

    routed = LabelRouterDataset(
        base_ds=target_adapt_gt,
        queried_ids=set(round_state.queried_ids),
        pseudo_store=dict(round_state.pseudo_store),
        unlabeled_label=-1,
    )
    pseudo_ids = set(round_state.pseudo_store.keys())

    labeled = IdFilteredDataset(routed, mode="labeled", queried_ids=set(round_state.queried_ids), pseudo_ids=pseudo_ids)
    pool = IdFilteredDataset(routed, mode="pool", queried_ids=set(round_state.queried_ids), pseudo_ids=pseudo_ids)

    loaders: dict[str, DataLoader] = {
        "target_adapt_labeled": _loader(
            TwoViewDataset(labeled, weak_tf, strong_tf),
            batch_size=int(cfg.train.batch_size),
            shuffle=True,
            num_workers=int(getattr(cfg.train, "num_workers", 4)),
        ),
        "target_adapt_pool": _loader(
            TwoViewDataset(pool, weak_tf, strong_tf),
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=int(getattr(cfg.train, "num_workers", 4)),
        ),
        "target_test": _loader(
            target_test,
            batch_size=int(cfg.eval.batch_size),
            shuffle=False,
            num_workers=int(getattr(cfg.eval, "num_workers", 4)),
        ),
    }

    if len(pseudo_ids) > 0:
        pseudo = IdFilteredDataset(
            routed,
            mode="pseudo",
            queried_ids=set(round_state.queried_ids),
            pseudo_ids=pseudo_ids,
        )
        loaders["target_adapt_pseudo"] = _loader(
            TwoViewDataset(pseudo, weak_tf, strong_tf),
            batch_size=int(cfg.train.batch_size),
            shuffle=True,
            num_workers=int(getattr(cfg.train, "num_workers", 4)),
        )

    if bool(getattr(cfg.eval, "monitor_source_val", False)):
        source_val = build_dataset(
            cfg,
            dataset_name=dataset_name,
            split="source_val",
            domain=source_domain,
            transform=eval_tf,
            return_id=True,
        )
        loaders["source_val"] = _loader(
            source_val,
            batch_size=int(cfg.eval.batch_size),
            shuffle=False,
            num_workers=int(getattr(cfg.eval, "num_workers", 4)),
        )

    return loaders
