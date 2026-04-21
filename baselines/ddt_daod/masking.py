"""Block masking used by the DINO-adapted DDT baseline."""

from __future__ import annotations

import torch


@torch.no_grad()
def apply_block_mask(image: torch.Tensor, *, block_size: int, masked_ratio: float) -> torch.Tensor:
    """Mask one CHW image tensor with DDT-style block dropout."""

    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(image.shape)}")
    _, height, width = image.shape
    block_size = max(int(block_size), 1)
    grid_h = max(1, round(float(height) / float(block_size)))
    grid_w = max(1, round(float(width) / float(block_size)))
    keep = (torch.rand((1, 1, grid_h, grid_w), device=image.device) > float(masked_ratio)).float()
    mask = torch.nn.functional.interpolate(keep, size=(height, width), mode="nearest").squeeze(0)
    return image * mask


@torch.no_grad()
def apply_block_mask_to_inputs(
    inputs: list[dict],
    *,
    block_size: int,
    masked_ratio: float,
) -> list[dict]:
    masked_inputs = []
    for item in inputs:
        updated = dict(item)
        updated["image"] = apply_block_mask(
            item["image"],
            block_size=block_size,
            masked_ratio=masked_ratio,
        )
        masked_inputs.append(updated)
    return masked_inputs
