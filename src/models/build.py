"""Model builder and optimizer param group helpers."""

from __future__ import annotations

from typing import Any

from torch import nn

from .lora import apply_finetune_mode, maybe_apply_lora
from .resnet_head import ResNetBottleneckClassifier


def build_model(cfg: Any) -> nn.Module:
    model_cfg = cfg.model
    num_classes = int(cfg.data.num_classes)
    model = ResNetBottleneckClassifier(
        backbone_name=model_cfg.backbone,
        num_classes=num_classes,
        bottleneck_dim=int(getattr(model_cfg, "bottleneck_dim", 256)),
        pretrained=bool(getattr(model_cfg, "pretrained", True)),
        use_relu=bool(getattr(model_cfg, "bottleneck_relu", True)),
    )
    return maybe_apply_lora(model, cfg)


def get_param_groups(cfg: Any, model: nn.Module) -> list[dict]:
    mode = getattr(getattr(cfg, "train", object()), "finetune_mode", "full_finetune")
    apply_finetune_mode(model, mode=mode)
    params = [p for p in model.parameters() if p.requires_grad]
    return [{"params": params}]
