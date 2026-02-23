"""LoRA helpers for model injection."""

from __future__ import annotations

from typing import Any

from torch import nn


def maybe_apply_lora(model: nn.Module, cfg: Any) -> nn.Module:
    lora_cfg = getattr(getattr(cfg, "model", object()), "lora", None)
    if lora_cfg is None or not getattr(lora_cfg, "enabled", False):
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PEFT is required when LoRA is enabled") from exc

    target_modules = list(getattr(lora_cfg, "target_modules", []))
    r = int(getattr(lora_cfg, "r", 8))
    alpha = int(getattr(lora_cfg, "alpha", 16))
    dropout = float(getattr(lora_cfg, "dropout", 0.0))

    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
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
