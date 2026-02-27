"""Round-based SF-ADA method orchestration."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.utils import (
    build_round_select_pool_loader,
    build_round_train_loaders,
    build_static_eval_loaders,
    build_target_adapt_base,
)
from src.engine.ckpt import load_checkpoint
from src.engine.trainer import TargetFinetuneTrainer
from src.engine.utils import build_optimizer, build_scheduler, save_json
from src.models import build_model
from src.models.lora import apply_finetune_mode

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


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())


def _resolve_method_run_dir(cfg: Any) -> Path:
    run_cfg = getattr(cfg, "run", object())
    if hasattr(run_cfg, "dir"):
        return Path(run_cfg.dir)
    root = Path(getattr(run_cfg, "root_dir", "runs"))
    ds = _slug(str(cfg.data.dataset_name))
    src = _slug(str(cfg.data.source_domain))
    tgt = _slug(str(cfg.data.target_domain))
    return root / "ours" / ds / f"{src}_to_{tgt}"


class OurMethod:
    def __init__(self, cfg: Any, num_classes: int, device: torch.device) -> None:
        self.cfg = cfg
        self.num_classes = num_classes
        self.device = device

        mcfg = cfg.method
        self.use_debias = bool(getattr(mcfg, "use_debias", False))
        self.debias_lambda = float(getattr(mcfg, "debias_lambda", 1.0))
        self.prior_momentum = getattr(mcfg, "prior_momentum", None)
        self.score_use_margin = bool(getattr(mcfg, "score_use_margin", True))
        self.score_use_change = bool(getattr(mcfg, "score_use_change", True))
        self.w_margin = float(getattr(mcfg, "w_margin", 0.5))
        self.w_change = float(getattr(mcfg, "w_change", 0.5))
        self.use_pseudo = bool(getattr(mcfg, "use_pseudo", False))
        self.pseudo_keep_ratio = float(getattr(mcfg, "pseudo_keep_ratio", 0.5))
        self.round_epochs = int(getattr(self.cfg.method, "round_epochs", 1))

        self.prev_by_id: dict[str, torch.Tensor] = {}
        self.prior_ema: torch.Tensor | None = None

        # Static datasets/loaders: built once for all rounds.
        self.target_adapt_gt = build_target_adapt_base(cfg)
        self.static_eval_loaders = build_static_eval_loaders(cfg)
        self.run_dir = _resolve_method_run_dir(cfg)

    def compute_round_budgets(self, budget_total: int, num_rounds: int) -> list[int]:
        base = budget_total // num_rounds
        ks = [base] * num_rounds
        ks[-1] += budget_total - sum(ks)
        return ks

    def resolve_budget_total(self, budget_cfg: Any) -> int:
        n_target = len(self.target_adapt_gt)
        # ratio budget in (0, 1]: total_budget = |target_adapt| * ratio
        if isinstance(budget_cfg, float) and 0.0 < budget_cfg <= 1.0:
            return max(1, int(n_target * budget_cfg))
        # integer-like absolute budget
        return max(0, int(budget_cfg))

    def build_select_pool_loader(self, state: RoundState) -> DataLoader:
        return build_round_select_pool_loader(self.cfg, self.target_adapt_gt, state)

    @torch.no_grad()
    def infer_select_pool(self, model, select_pool_loader: DataLoader) -> InferenceResult:
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

        if self.use_debias:
            prior = estimate_prior(prob)
            if self.prior_momentum is not None and self.prior_ema is not None:
                m = float(self.prior_momentum)
                prior = m * self.prior_ema + (1.0 - m) * prior
                prior = prior / prior.sum().clamp(min=1e-12)
            self.prior_ema = prior
            prob = softmax(debias_logits(logits, prior, self.debias_lambda))

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
            pseudo_store=pseudo_store_next,  # refresh each round
            budget_total=state.budget_total,
            budget_used=state.budget_used + len(new_queried_ids),
        )

    def build_train_loaders(self, state: RoundState) -> dict[str, DataLoader]:
        return build_round_train_loaders(self.cfg, self.target_adapt_gt, state)

    def run_round(
        self,
        round_idx: int,
        ckpt_in: str,
        state_in: RoundState,
        budget_k: int,
    ) -> tuple[str, RoundState, dict[str, Any]]:
        # A) load model M_{r-1}
        model = build_model(self.cfg).to(self.device)
        strict = False if round_idx == 0 else True
        load_checkpoint(ckpt_in, model, load_optimizer=False, strict=strict)

        # B) selection pool from current state (exclude queried only)
        select_pool_loader = self.build_select_pool_loader(state_in)

        # C) infer + plan + state update
        infer = self.infer_select_pool(model, select_pool_loader)
        new_queried_ids, pseudo_store_next, aux = self.plan_round(state_in, infer, budget_k)
        state_out = self.apply_plan(state_in, new_queried_ids, pseudo_store_next, round_idx)

        # D) train loaders from updated state (labeled + pseudo)
        train_loaders = self.build_train_loaders(state_out)

        # E) finetune one round
        apply_finetune_mode(model, mode=self.cfg.train.finetune_mode)
        optimizer = build_optimizer(self.cfg, model)
        scheduler = build_scheduler(self.cfg, optimizer)
        trainer = TargetFinetuneTrainer(
            self.cfg,
            model,
            optimizer,
            scheduler,
            self.device,
            aux=aux,
        )
        round_dir = self.run_dir / f"round_{round_idx}"
        ckpt_dir = round_dir / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_out = str(ckpt_dir / "ckpt_last.pt")
        ckpt_best = str(ckpt_dir / "ckpt_best.pt")
        summary = trainer.fit(
            train_loaders=train_loaders,
            eval_loaders=self.static_eval_loaders,
            max_epochs=self.round_epochs,
            ckpt_last_path=ckpt_out,
            ckpt_best_path=ckpt_best,
            monitor_loader="target_test",
            monitor_metric="acc_top1",
            ckpt_extra={"round": round_idx, "ckpt_in": ckpt_in},
            train_log_path=str(round_dir / "train_log.jsonl"),
            log_every_iters=int(getattr(self.cfg.train, "log_every_iters", 0)),
        )
        eval_metrics = summary.eval_history[-1] if summary.eval_history else {}

        save_json(round_dir / "new_queried_ids.json", {"new_queried_ids": new_queried_ids})
        save_json(round_dir / "pseudo_store.json", {"pseudo_store": state_out.pseudo_store})
        save_json(
            round_dir / "metrics.json",
            {
                "eval_last": eval_metrics,
                "eval_history": summary.eval_history,
                "best_epoch": summary.best_epoch,
                "best_score": summary.best_score,
                "monitor_metric": "target_test.acc_top1",
            },
        )
        save_round_state(str(self.run_dir / "state_last.json"), state_out)

        return ckpt_out, state_out, {"eval": eval_metrics}

    def run_all_rounds(self, ckpt_init: str, state_init: RoundState) -> tuple[str, RoundState]:
        budget_cfg = getattr(self.cfg.method, "budget_total", state_init.budget_total)
        budget_total = self.resolve_budget_total(budget_cfg)
        rounds = int(self.cfg.method.num_rounds)
        k_list = self.compute_round_budgets(budget_total=budget_total, num_rounds=rounds)

        ckpt = ckpt_init
        state = RoundState(
            round_idx=state_init.round_idx,
            queried_ids=set(state_init.queried_ids),
            pseudo_store=dict(state_init.pseudo_store),
            budget_total=budget_total,
            budget_used=state_init.budget_used,
        )
        for r in range(rounds):
            ckpt, state, _ = self.run_round(r, ckpt, state, k_list[r])
        return ckpt, state
