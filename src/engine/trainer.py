"""Trainer classes for supervised source and target fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .ckpt import save_checkpoint
from .metrics import compute_classification_metrics


@dataclass
class TrainSummary:
    train_history: list[dict[str, float]]
    eval_history: list[dict[str, Any]]
    best_epoch: int | None = None
    best_score: float | None = None


class SupervisedTrainer:
    def __init__(
        self,
        cfg: Any,
        model,
        optimizer,
        scheduler,
        device: torch.device,
        aux: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = int(cfg.data.num_classes)
        self.aux = aux or {}

    def _extract_xy(self, batch: dict[str, Any], use_strong: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if "image" in batch:
            x = batch["image"]
        elif use_strong:
            x = batch["x_s"]
        else:
            x = batch["x_w"]
        y = batch["label"]
        return x.to(self.device), y.to(self.device)

    def train_one_epoch(
        self,
        train_loader,
        log_writer=None,
        log_every: int | None = None,
    ) -> dict[str, float]:
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
            if log_writer is not None and log_every is not None and log_every > 0 and (n_steps % log_every == 0):
                log_writer.write(
                    json.dumps(
                        {
                            "iter": int(n_steps),
                            "loss": float(loss.item()),
                        }
                    )
                    + "\n"
                )

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

    def _extract_monitor_score(
        self,
        eval_metrics: dict[str, Any],
        monitor_loader: str | None,
        monitor_metric: str,
    ) -> float | None:
        if not monitor_loader:
            return None
        loader_metrics = eval_metrics.get(monitor_loader, {})
        score = loader_metrics.get(monitor_metric)
        return float(score) if score is not None else None

    def _maybe_save_ckpt(
        self,
        ckpt_path: str | None,
        epoch_idx: int,
        ckpt_extra: dict[str, Any],
    ) -> None:
        if not ckpt_path:
            return
        save_checkpoint(
            ckpt_path,
            self.model,
            self.optimizer,
            self.scheduler,
            scaler=None,
            step=0,
            epoch=epoch_idx,
            extra=ckpt_extra,
        )

    def fit(
        self,
        train_loader,
        eval_loaders: dict[str, Any],
        max_epochs: int,
        ckpt_last_path: str | None = None,
        ckpt_best_path: str | None = None,
        monitor_loader: str | None = None,
        monitor_metric: str = "acc_top1",
        ckpt_extra: dict[str, Any] | None = None,
        train_log_path: str | None = None,
        log_every_iters: int | None = None,
    ) -> TrainSummary:
        train_hist: list[dict[str, float]] = []
        eval_hist: list[dict[str, Any]] = []
        best_score = float("-inf")
        best_epoch: int | None = None
        extra = dict(ckpt_extra or {})
        log_f = None
        if train_log_path:
            Path(train_log_path).parent.mkdir(parents=True, exist_ok=True)
            log_f = open(train_log_path, "w", encoding="utf-8")
        for epoch_idx in range(1, max_epochs + 1):
            train_metrics = self.train_one_epoch(
                train_loader,
                log_writer=log_f,
                log_every=log_every_iters,
            )
            train_hist.append(train_metrics)
            if log_f is not None:
                log_f.write(json.dumps({"epoch": epoch_idx, **train_metrics}) + "\n")

            eval_metrics = self.evaluate(eval_loaders)
            eval_hist.append(eval_metrics)

            self._maybe_save_ckpt(ckpt_last_path, epoch_idx, {**extra, "kind": "last"})
            score = self._extract_monitor_score(eval_metrics, monitor_loader, monitor_metric)
            if score is not None and score >= best_score:
                best_score = score
                best_epoch = epoch_idx
                self._maybe_save_ckpt(
                    ckpt_best_path,
                    epoch_idx,
                    {**extra, "kind": "best", "best_loader": monitor_loader, "best_metric": monitor_metric, "best_score": best_score},
                )
        if log_f is not None:
            log_f.close()
        return TrainSummary(
            train_history=train_hist,
            eval_history=eval_hist,
            best_epoch=best_epoch,
            best_score=None if best_epoch is None else best_score,
        )


class SourceTrainer(SupervisedTrainer):
    pass


class TargetFinetuneTrainer(SupervisedTrainer):
    def __init__(
        self,
        cfg: Any,
        model,
        optimizer,
        scheduler,
        device: torch.device,
        aux: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg, model, optimizer, scheduler, device, aux=aux)
        mcfg = getattr(cfg, "method", object())
        self.pseudo_loss_weight = float(getattr(mcfg, "pseudo_loss_weight", 1.0))
        self.aml_lambda = float(getattr(mcfg, "aml_lambda", 1.0))

    def _aml_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        target_prior: torch.Tensor,
    ) -> torch.Tensor:
        # AML(z, y_hat, h_t) is CE on logits shifted by Delta_j = lambda * log(1 / h_j).
        eps = 1e-12
        prior = target_prior.to(logits.device, dtype=logits.dtype).clamp(min=eps)
        delta = self.aml_lambda * torch.log(1.0 / prior)
        return F.cross_entropy(logits - delta.unsqueeze(0), y)

    def fit(
        self,
        train_loaders,
        eval_loaders: dict[str, Any],
        max_epochs: int,
        ckpt_last_path: str | None = None,
        ckpt_best_path: str | None = None,
        monitor_loader: str | None = None,
        monitor_metric: str = "acc_top1",
        ckpt_extra: dict[str, Any] | None = None,
        train_log_path: str | None = None,
        log_every_iters: int | None = None,
    ) -> TrainSummary:
        if "target_adapt_labeled" not in train_loaders:
            raise KeyError("train_loaders must contain 'target_adapt_labeled'.")

        labeled_loader = train_loaders["target_adapt_labeled"]
        pseudo_loader = train_loaders.get("target_adapt_pseudo")
        # Reuse SupervisedTrainer.fit loop; train_one_epoch handles paired loaders.
        train_loader = labeled_loader if pseudo_loader is None else (labeled_loader, pseudo_loader)
        return super().fit(
            train_loader,
            eval_loaders,
            max_epochs,
            ckpt_last_path=ckpt_last_path,
            ckpt_best_path=ckpt_best_path,
            monitor_loader=monitor_loader,
            monitor_metric=monitor_metric,
            ckpt_extra=ckpt_extra,
            train_log_path=train_log_path,
            log_every_iters=log_every_iters,
        )

    def train_one_epoch(
        self,
        train_loader,
        log_writer=None,
        log_every: int | None = None,
    ) -> dict[str, float]:
        # Allow SupervisedTrainer.fit to pass a tuple (labeled_loader, pseudo_loader).
        if isinstance(train_loader, (tuple, list)) and len(train_loader) == 2:
            train_loader, pseudo_loader = train_loader
        else:
            return super().train_one_epoch(
                train_loader,
                log_writer=log_writer,
                log_every=log_every,
            )

        self.model.train()
        loss_sum = 0.0
        labeled_loss_sum = 0.0
        pseudo_loss_sum = 0.0
        n_steps = 0
        n_skip = 0
        target_prior_payload = self.aux.get("target_prior")
        target_prior = None
        if target_prior_payload is not None:
            target_prior = torch.tensor(target_prior_payload, dtype=torch.float32, device=self.device)

        for labeled_batch, pseudo_batch in zip(train_loader, pseudo_loader):
            x_l, y_l = self._extract_xy(labeled_batch)
            x_p, y_p = self._extract_xy(pseudo_batch, use_strong=True)

            mask_l = y_l != -1
            mask_p = y_p != -1
            if int(mask_l.sum().item()) == 0 or int(mask_p.sum().item()) == 0:
                n_skip += 1
                continue

            logits_l = self.model(x_l)
            logits_p = self.model(x_p)

            loss_l = F.cross_entropy(logits_l[mask_l], y_l[mask_l])
            if target_prior is None:
                loss_p = F.cross_entropy(logits_p[mask_p], y_p[mask_p])
            else:
                loss_p = self._aml_loss(logits_p[mask_p], y_p[mask_p], target_prior)

            loss = loss_l + self.pseudo_loss_weight * loss_p
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item())
            labeled_loss_sum += float(loss_l.item())
            pseudo_loss_sum += float(loss_p.item())
            n_steps += 1
            if log_writer is not None and log_every is not None and log_every > 0 and (n_steps % log_every == 0):
                log_writer.write(
                    json.dumps(
                        {
                            "iter": int(n_steps),
                            "loss": float(loss.item()),
                            "loss_labeled": float(loss_l.item()),
                            "loss_pseudo": float(loss_p.item()),
                        }
                    )
                    + "\n"
                )

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss_sum / max(n_steps, 1),
            "loss_labeled": labeled_loss_sum / max(n_steps, 1),
            "loss_pseudo": pseudo_loss_sum / max(n_steps, 1),
            "steps": float(n_steps),
            "skipped": float(n_skip),
            "use_aml": float(target_prior is not None),
        }
