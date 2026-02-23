"""Round-based SF-ADA method scaffold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.utils import build_adapt_loaders
from src.engine.ckpt import load_checkpoint, save_checkpoint
from src.engine.trainer import TargetFinetuneTrainer
from src.engine.utils import apply_train_mode, build_optimizer, build_scheduler, save_json
from src.models import build_model

from .utils import change_l1, debias_logits, estimate_prior, margin_from_prob, rank_norm, softmax


@dataclass
class RoundState:
    round_idx: int
    queried_ids: set[str]
    pseudo_store: dict[str, int]
    budget_total: int = 0
    budget_used: int = 0


@dataclass
class InferenceResult:
    sample_ids: list[str]
    logits: torch.Tensor
    pred_label: torch.Tensor
    margin: torch.Tensor
    change: torch.Tensor
    score_query: torch.Tensor
    target_prior: torch.Tensor | None


def save_round_state(path: str, state: RoundState) -> None:
    payload = asdict(state)
    payload["queried_ids"] = sorted(list(state.queried_ids))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_round_state(path: str) -> RoundState:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["queried_ids"] = set(payload["queried_ids"])
    payload["pseudo_store"] = {str(k): int(v) for k, v in payload["pseudo_store"].items()}
    return RoundState(**payload)


class OurMethod:
    def __init__(self, cfg: Any, num_classes: int, device: torch.device) -> None:
        self.cfg = cfg
        self.num_classes = num_classes
        self.device = device

        mcfg = cfg.method
        self.select_exclude = bool(getattr(mcfg, "select_exclude", True))
        self.use_debias = bool(getattr(mcfg, "use_debias", False))
        self.debias_lambda = float(getattr(mcfg, "debias_lambda", 1.0))
        self.prior_momentum = getattr(mcfg, "prior_momentum", None)
        self.score_use_margin = bool(getattr(mcfg, "score_use_margin", True))
        self.score_use_change = bool(getattr(mcfg, "score_use_change", True))
        self.w_margin = float(getattr(mcfg, "w_margin", 0.5))
        self.w_change = float(getattr(mcfg, "w_change", 0.5))
        self.use_pseudo = bool(getattr(mcfg, "use_pseudo", False))
        self.use_aml = bool(getattr(mcfg, "use_aml", False))
        self.aml_weight = float(getattr(mcfg, "aml_weight", 1.0))
        self.pseudo_keep_ratio = float(getattr(mcfg, "pseudo_keep_ratio", 0.5))

        self.prev_by_id: dict[str, torch.Tensor] = {}
        self.prior_ema: torch.Tensor | None = None

    def compute_round_budgets(self, budget_total: int, num_rounds: int) -> list[int]:
        base = budget_total // num_rounds
        ks = [base] * num_rounds
        ks[-1] += budget_total - sum(ks)
        return ks

    def build_select_pool_loader(self, target_adapt_gt, state: RoundState) -> DataLoader:
        class _PoolView(torch.utils.data.Dataset):
            def __init__(self, ds, queried):
                self.ds = ds
                self.queried = queried
                self.indices = [
                    i for i in range(len(ds)) if (not self.queried or ds[i]["sample_id"] not in queried)
                ]

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                item = self.ds[self.indices[idx]]
                return {"image": item["image"], "sample_id": item["sample_id"]}

        queried = state.queried_ids if self.select_exclude else set()
        ds = _PoolView(target_adapt_gt, queried)
        return DataLoader(
            ds,
            batch_size=int(self.cfg.eval.batch_size),
            shuffle=False,
            num_workers=int(getattr(self.cfg.eval, "num_workers", 4)),
        )

    @torch.no_grad()
    def infer_select_pool(self, model, select_pool_loader) -> InferenceResult:
        model.eval()
        sample_ids: list[str] = []
        logits_all = []

        for batch in select_pool_loader:
            x = batch["image"].to(self.device)
            logits = model(x)
            logits_all.append(logits.cpu())
            sample_ids.extend(list(batch["sample_id"]))

        if len(sample_ids) == 0:
            empty = torch.empty((0, self.num_classes), dtype=torch.float32)
            em = torch.empty((0,), dtype=torch.float32)
            return InferenceResult([], empty, torch.empty((0,), dtype=torch.long), em, em, em, None)

        logits = torch.cat(logits_all, dim=0)
        prob = softmax(logits)
        prior = None
        used_logits = logits
        if self.use_debias:
            prior = estimate_prior(prob)
            if self.prior_momentum is not None and self.prior_ema is not None:
                m = float(self.prior_momentum)
                prior = m * self.prior_ema + (1.0 - m) * prior
                prior = prior / prior.sum().clamp(min=1e-12)
            self.prior_ema = prior
            used_logits = debias_logits(logits, prior, self.debias_lambda)
            prob = softmax(used_logits)

        margin = margin_from_prob(prob)
        pred = prob.argmax(dim=1)

        changes = []
        for i, sid in enumerate(sample_ids):
            prev = self.prev_by_id.get(str(sid))
            if prev is None:
                changes.append(torch.tensor(0.0))
            else:
                changes.append(change_l1(prob[i : i + 1], prev[None, :])[0])
            self.prev_by_id[str(sid)] = prob[i].detach().clone()
        change = torch.stack(changes)

        u = rank_norm(1.0 - margin)
        c = rank_norm(change)
        score = torch.zeros_like(u)
        if self.score_use_margin:
            score += self.w_margin * u
        if self.score_use_change:
            score += self.w_change * c

        return InferenceResult(sample_ids, logits, pred, margin, change, score, prior)

    def plan_round(
        self,
        state: RoundState,
        infer: InferenceResult,
        budget_k: int,
    ) -> tuple[list[str], dict[str, int], dict[str, Any]]:
        rank = torch.argsort(infer.score_query, descending=True)

        new_queried_ids: list[str] = []
        for idx in rank.tolist():
            sid = str(infer.sample_ids[idx])
            if sid in state.queried_ids:
                continue
            new_queried_ids.append(sid)
            if len(new_queried_ids) >= budget_k:
                break

        pseudo_store_next: dict[str, int] = {}
        if self.use_pseudo and len(infer.sample_ids) > 0:
            keep_n = int(len(infer.sample_ids) * self.pseudo_keep_ratio)
            pseudo_rank = torch.argsort(infer.score_query, descending=False)
            for idx in pseudo_rank[:keep_n].tolist():
                sid = str(infer.sample_ids[idx])
                if sid in state.queried_ids or sid in new_queried_ids:
                    continue
                pseudo_store_next[sid] = int(infer.pred_label[idx].item())

        aux = {
            "target_prior": infer.target_prior.tolist() if infer.target_prior is not None else None,
            "stats": {
                "margin_mean": float(infer.margin.mean().item()) if infer.margin.numel() else 0.0,
                "change_mean": float(infer.change.mean().item()) if infer.change.numel() else 0.0,
            },
        }
        return new_queried_ids, pseudo_store_next, aux

    def apply_plan(
        self,
        state: RoundState,
        new_queried_ids: list[str],
        pseudo_store_next: dict[str, int],
        round_idx: int,
    ) -> RoundState:
        queried = set(state.queried_ids)
        queried.update(new_queried_ids)
        return RoundState(
            round_idx=round_idx,
            queried_ids=queried,
            pseudo_store=pseudo_store_next,
            budget_total=state.budget_total,
            budget_used=state.budget_used + len(new_queried_ids),
        )

    def build_train_loaders(self, state: RoundState) -> dict[str, DataLoader]:
        return build_adapt_loaders(self.cfg, state)

    def run_round(
        self,
        round_idx: int,
        ckpt_in: str,
        state_in: RoundState,
        budget_k: int,
    ) -> tuple[str, RoundState, dict[str, Any]]:
        model = build_model(self.cfg, num_classes=self.num_classes).to(self.device)
        load_checkpoint(ckpt_in, model, load_optimizer=False)

        select_loaders = build_adapt_loaders(self.cfg, state_in)
        infer = self.infer_select_pool(model, select_loaders["target_adapt_pool"])
        new_queried_ids, pseudo_store_next, aux = self.plan_round(state_in, infer, budget_k)
        state_out = self.apply_plan(state_in, new_queried_ids, pseudo_store_next, round_idx)

        loaders = self.build_train_loaders(state_out)
        apply_train_mode(self.cfg, model, mode=self.cfg.train.finetune_mode)
        optimizer = build_optimizer(self.cfg, model)
        scheduler = build_scheduler(self.cfg, optimizer)
        trainer = TargetFinetuneTrainer(
            self.cfg,
            model,
            optimizer,
            scheduler,
            self.device,
            self.num_classes,
            aux=aux,
        )

        pseudo_loader = loaders.get("target_adapt_pseudo")
        train_metrics = trainer.train_one_epoch(loaders["target_adapt_labeled"], pseudo_loader=pseudo_loader)
        eval_metrics = trainer.evaluate({k: v for k, v in loaders.items() if k in {"target_test", "source_val"}})

        run_dir = Path(self.cfg.run.dir)
        round_dir = run_dir / f"round_{round_idx}"
        (round_dir / "ckpt").mkdir(parents=True, exist_ok=True)

        ckpt_out = str(round_dir / "ckpt" / "ckpt_last.pt")
        save_checkpoint(
            ckpt_out,
            model,
            optimizer,
            scheduler,
            scaler=None,
            step=0,
            epoch=0,
            extra={"round": round_idx},
        )

        save_json(round_dir / "new_queried_ids.json", {"new_queried_ids": new_queried_ids})
        save_json(round_dir / "pseudo_store.json", {"pseudo_store": state_out.pseudo_store})
        save_json(round_dir / "metrics.json", {"train": train_metrics, "eval": eval_metrics})
        save_round_state(str(run_dir / "state_last.json"), state_out)

        return ckpt_out, state_out, {"train": train_metrics, "eval": eval_metrics}

    def run_all_rounds(self, ckpt_init: str, state_init: RoundState) -> tuple[str, RoundState]:
        budget_total = int(getattr(self.cfg.method, "budget_total", state_init.budget_total))
        rounds = int(self.cfg.method.num_rounds)
        k_list = self.compute_round_budgets(budget_total=budget_total, num_rounds=rounds)

        ckpt = ckpt_init
        state = state_init
        for r in range(rounds):
            ckpt, state, _ = self.run_round(r, ckpt, state, k_list[r])
        return ckpt, state
