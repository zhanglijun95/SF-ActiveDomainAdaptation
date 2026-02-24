"""Split utilities and dataloader builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .datasets import build_dataset
from .transforms import build_eval_transform, build_strong_transform, build_weak_transform
from .wrappers import IdFilteredDataset, LabelRouterDataset, TwoViewDataset


def _loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def build_pretrain_loaders(cfg: Any) -> dict[str, DataLoader]:
    weak_tf = build_weak_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    train_ds = build_dataset(
        cfg,
        split="source_train",
        transform=weak_tf,
    )

    val_ds = build_dataset(
        cfg,
        split="source_val",
        transform=eval_tf,
    )

    batch_size = int(cfg.train.batch_size)
    workers = int(getattr(cfg.train, "num_workers", 4))
    return {
        "source_train": _loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
        "source_val": _loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
    }

def build_target_adapt_base(cfg: Any):
    """Build target_adapt base dataset once; shared across rounds."""
    return build_dataset(cfg, split="target_adapt", transform=None)


def build_static_eval_loaders(cfg: Any) -> dict[str, DataLoader]:
    """Build static eval loaders that do not change by round."""
    eval_tf = build_eval_transform(cfg)
    loaders: dict[str, DataLoader] = {}

    target_test = build_dataset(cfg, split="target_test", transform=eval_tf)
    loaders["target_test"] = _loader(
        target_test,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(getattr(cfg.eval, "num_workers", 4)),
    )

    if bool(getattr(cfg.eval, "monitor_source_val", False)):
        source_val = build_dataset(cfg, split="source_val", transform=eval_tf)
        loaders["source_val"] = _loader(
            source_val,
            batch_size=int(cfg.eval.batch_size),
            shuffle=False,
            num_workers=int(getattr(cfg.eval, "num_workers", 4)),
        )
    return loaders


def build_round_select_pool_loader(cfg: Any, target_adapt_gt, round_state: Any) -> DataLoader:
    """Build selection pool loader for planning:

    select_pool = target_adapt \\ queried_ids
    Pseudo-labeled samples remain in selection pool by design.
    """
    weak_tf = build_weak_transform(cfg)

    class _SingleViewDataset:
        def __init__(self, base_ds, tf):
            self.base_ds = base_ds
            self.tf = tf

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            item = self.base_ds[idx]
            return {
                "image": self.tf(item["image"]),
                "label": item["label"],
                "sample_id": item["sample_id"],
            }

    routed = LabelRouterDataset(
        base_ds=target_adapt_gt,
        queried_ids=set(round_state.queried_ids),
        pseudo_store={},  # ignore pseudo during selection pool construction
        unlabeled_label=-1,
    )
    pool = IdFilteredDataset(routed, mode="pool")
    select_ds = _SingleViewDataset(pool, weak_tf)
    return _loader(
        select_ds,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(getattr(cfg.eval, "num_workers", 4)),
    )


def build_round_train_loaders(cfg: Any, target_adapt_gt, round_state: Any) -> dict[str, DataLoader]:
    """Build round-dependent train loaders after query/pseudo plan is applied."""
    weak_tf = build_weak_transform(cfg)
    strong_tf = build_strong_transform(cfg)

    routed = LabelRouterDataset(
        base_ds=target_adapt_gt,
        queried_ids=set(round_state.queried_ids),
        pseudo_store=dict(round_state.pseudo_store),
        unlabeled_label=-1,
    )
    labeled = IdFilteredDataset(routed, mode="labeled")
    loaders: dict[str, DataLoader] = {
        "target_adapt_labeled": _loader(
            TwoViewDataset(labeled, weak_tf, strong_tf),
            batch_size=int(cfg.train.batch_size),
            shuffle=True,
            num_workers=int(getattr(cfg.train, "num_workers", 4)),
        )
    }
    if len(round_state.pseudo_store) > 0:
        pseudo = IdFilteredDataset(routed, mode="pseudo")
        loaders["target_adapt_pseudo"] = _loader(
            TwoViewDataset(pseudo, weak_tf, strong_tf),
            batch_size=int(cfg.train.batch_size),
            shuffle=True,
            num_workers=int(getattr(cfg.train, "num_workers", 4)),
        )
    return loaders

def build_adapt_loaders(cfg: Any, round_state: Any) -> dict[str, DataLoader]:
    """Backward-compatible combined builder.

    New code should prefer:
      - build_target_adapt_base
      - build_round_select_pool_loader
      - build_round_train_loaders
      - build_static_eval_loaders
    """
    target_adapt_gt = build_target_adapt_base(cfg)
    out = {}
    out["target_adapt_pool"] = build_round_select_pool_loader(cfg, target_adapt_gt, round_state)
    out.update(build_round_train_loaders(cfg, target_adapt_gt, round_state))
    out.update(build_static_eval_loaders(cfg))
    return out
