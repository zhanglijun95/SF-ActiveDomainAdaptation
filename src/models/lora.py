"""LoRA helpers for model injection."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

from torch import nn
from peft import LoraConfig, get_peft_model


def _resolve_target_modules(model: nn.Module, target_modules: list[str]) -> list[str]:
    """Resolve exact module names from exact names and wildcard patterns."""
    if not target_modules:
        return []

    conv_names = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    resolved: list[str] = []
    seen: set[str] = set()
    for spec in target_modules:
        # Treat '*' and '?' as wildcard patterns; otherwise exact name.
        matches = (
            [n for n in conv_names if fnmatchcase(n, spec)]
            if ("*" in spec or "?" in spec)
            else [spec] if spec in conv_names else []
        )
        for name in matches:
            if name not in seen:
                resolved.append(name)
                seen.add(name)
    return resolved


def _build_rank_pattern(
    model: nn.Module,
    target_modules: list[str],
    rank_schedule: dict[str, int],
) -> dict[str, int]:
    """Build PEFT rank_pattern from stage-wise schedule.

    Example rank_schedule:
      {"layer1": 16, "layer2": 8, "layer3": 4, "layer4": 2}
    """
    if not rank_schedule:
        return {}

    target_set = set(target_modules)
    rank_pattern: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if target_set and name not in target_set:
            continue
        for stage, rank in rank_schedule.items():
            if name.startswith(f"{stage}."):
                rank_pattern[name] = int(rank)
                break
    return rank_pattern


def maybe_apply_lora(model: nn.Module, cfg: Any) -> nn.Module:
    lora_cfg = getattr(getattr(cfg, "model", object()), "lora", None)
    if lora_cfg is None or not getattr(lora_cfg, "enabled", False):
        return model

    target_modules_cfg = list(getattr(lora_cfg, "target_modules", []))
    target_modules = _resolve_target_modules(model, target_modules_cfg)
    r = int(getattr(lora_cfg, "r", 8))
    alpha = int(getattr(lora_cfg, "alpha", 16))
    dropout = float(getattr(lora_cfg, "dropout", 0.0))
    rank_schedule = dict(getattr(lora_cfg, "rank_schedule", {}))
    rank_pattern = _build_rank_pattern(model, target_modules, rank_schedule)

    lora_kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    if rank_pattern:
        lora_kwargs["rank_pattern"] = rank_pattern
    lora = LoraConfig(**lora_kwargs)
    return get_peft_model(model, lora)


def apply_finetune_mode(model: nn.Module, mode: str) -> None:
    if mode == "full_finetune" or mode == "source_train":
        for p in model.parameters():
            p.requires_grad = True
        return

    if mode != "lora_finetune":
        raise ValueError(f"Unsupported finetune mode: {mode}")

    for name, p in model.named_parameters():
        trainable = (
            "lora_" in name
            or "bn" in name.lower()
            or "batchnorm" in name.lower()
        )
        p.requires_grad = trainable
