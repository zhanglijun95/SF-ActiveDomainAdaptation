"""Augmentation pipelines."""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def build_weak_transform(cfg: Any):
    size = int(getattr(getattr(cfg, "data", object()), "image_size", 224))
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_strong_transform(cfg: Any):
    size = int(getattr(getattr(cfg, "data", object()), "image_size", 224))
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(cfg: Any):
    size = int(getattr(getattr(cfg, "data", object()), "image_size", 224))
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
