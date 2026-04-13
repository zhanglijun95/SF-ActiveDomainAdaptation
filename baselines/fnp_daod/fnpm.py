"""False Negative Prediction Module and its local training helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def normalize_fn_count(count: int | float, *, target_cap: float) -> float:
    capped = min(max(float(count), 0.0), float(target_cap))
    return float(capped / max(float(target_cap), 1e-12))


class FalseNegativePredictionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("FNPM requires at least two layers.")

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(num_layers) - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def fit_fnpm(
    model: FalseNegativePredictionModule,
    *,
    features: torch.Tensor,
    targets: torch.Tensor,
    cfg: Any,
    device: torch.device,
) -> list[dict[str, float]]:
    if features.numel() == 0 or targets.numel() == 0:
        return []

    fnpm_cfg = getattr(cfg.method, "fnpm", object())
    batch_size = int(getattr(fnpm_cfg, "batch_size", 32))
    epochs = int(getattr(fnpm_cfg, "epochs", 1))
    num_workers = int(getattr(fnpm_cfg, "num_workers", 0))
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(fnpm_cfg, "lr", 1e-4)),
        weight_decay=float(getattr(fnpm_cfg, "weight_decay", 1e-4)),
    )
    history: list[dict[str, float]] = []
    model.to(device)
    model.train()
    for epoch_idx in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)
            pred = model(feature_batch)
            loss = torch.nn.functional.mse_loss(pred, target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        history.append(
            {
                "epoch": float(epoch_idx),
                "loss": epoch_loss / max(steps, 1),
            }
        )
    return history
