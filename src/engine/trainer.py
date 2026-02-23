"""Trainer classes for supervised source and target fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .metrics import compute_classification_metrics


@dataclass
class TrainSummary:
    train_history: list[dict[str, float]]
    eval_history: list[dict[str, Any]]


class SupervisedTrainer:
    def __init__(
        self,
        cfg: Any,
        model,
        optimizer,
        scheduler,
        device: torch.device,
        num_classes: int,
        aux: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.aux = aux or {}

    def _extract_xy(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"] if "image" in batch else batch["x_w"]
        y = batch["label"]
        return x.to(self.device), y.to(self.device)

    def train_one_epoch(self, train_loader) -> dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        n_steps = 0
        n_skip = 0

        for batch in train_loader:
            x, y = self._extract_xy(batch)
            logits = self.model(x)
            mask = y != -1
            if int(mask.sum().item()) == 0:
                n_skip += 1
                continue

            loss = F.cross_entropy(logits[mask], y[mask])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item())
            n_steps += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = loss_sum / max(n_steps, 1)
        return {"loss": avg_loss, "steps": float(n_steps), "skipped": float(n_skip)}

    @torch.no_grad()
    def evaluate(self, loaders: dict[str, Any]) -> dict[str, Any]:
        self.model.eval()
        out: dict[str, Any] = {}
        for name, loader in loaders.items():
            logits_all = []
            labels_all = []
            for batch in loader:
                x, y = self._extract_xy(batch)
                logits = self.model(x)
                mask = y != -1
                if int(mask.sum().item()) == 0:
                    continue
                logits_all.append(logits[mask].detach().cpu())
                labels_all.append(y[mask].detach().cpu())

            if not logits_all:
                out[name] = {"acc_top1": 0.0, "mean_acc": 0.0, "acc_per_class": []}
                continue

            logits_cat = torch.cat(logits_all, dim=0)
            labels_cat = torch.cat(labels_all, dim=0)
            out[name] = compute_classification_metrics(
                logits=logits_cat,
                gt=labels_cat,
                num_classes=self.num_classes,
                with_per_class=True,
            )
        return out

    def fit(self, train_loader, eval_loaders: dict[str, Any], max_epochs: int) -> TrainSummary:
        train_hist: list[dict[str, float]] = []
        eval_hist: list[dict[str, Any]] = []
        for _ in range(max_epochs):
            train_hist.append(self.train_one_epoch(train_loader))
            eval_hist.append(self.evaluate(eval_loaders))
        return TrainSummary(train_hist, eval_hist)


class SourceTrainer(SupervisedTrainer):
    pass


class TargetFinetuneTrainer(SupervisedTrainer):
    def train_one_epoch(self, train_loader, pseudo_loader=None) -> dict[str, float]:
        labeled_metrics = super().train_one_epoch(train_loader)
        if pseudo_loader is None:
            return labeled_metrics

        self.model.train()
        pseudo_loss = 0.0
        pseudo_steps = 0
        for batch in pseudo_loader:
            x = batch["x_s"].to(self.device) if "x_s" in batch else batch["image"].to(self.device)
            y = batch["label"].to(self.device)
            mask = y != -1
            if int(mask.sum().item()) == 0:
                continue
            logits = self.model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            pseudo_loss += float(loss.item())
            pseudo_steps += 1

        if self.scheduler is not None:
            self.scheduler.step()

        labeled_metrics["pseudo_loss"] = pseudo_loss / max(pseudo_steps, 1)
        labeled_metrics["pseudo_steps"] = float(pseudo_steps)
        return labeled_metrics
