"""Round-based SF-ADA method orchestration."""

from __future__ import annotations

import json
import re
import warnings
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
    mcfg = getattr(cfg, "method", object())
    exp_name = str(getattr(mcfg, "exp_name", "")).strip()
    if exp_name:
        exp_tag = _slug(exp_name)
    else:
        random_pick = bool(getattr(mcfg, "random_pick", False))
        score_margin = bool(getattr(mcfg, "score_use_margin", True))
        score_change = bool(getattr(mcfg, "score_use_change", True))
        use_debias = bool(getattr(mcfg, "use_debias", False))
        use_pseudo = bool(getattr(mcfg, "use_pseudo", False))
        budget = str(getattr(mcfg, "budget_total", "na")).replace(".", "p")
        parts = [
            "rand" if random_pick else "score",
            f"m{int(score_margin)}",
            f"c{int(score_change)}",
            f"d{int(use_debias)}",
            f"p{int(use_pseudo and not random_pick)}",
            f"b{budget}",
        ]
        exp_tag = "_".join(parts)
    return root / "method" / exp_tag / ds / f"{src}_to_{tgt}"


class RoundAdaptationMethod:
    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.num_classes = int(cfg.data.num_classes)
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
        self.random_pick = bool(getattr(mcfg, "random_pick", False))
        self.pseudo_keep_ratio = float(getattr(mcfg, "pseudo_keep_ratio", 0.5))
        self.round_epochs = int(getattr(self.cfg.method, "round_epochs", 1))
        if self.random_pick and self.use_pseudo:
            warnings.warn("method.random_pick=true disables pseudo labels; overriding method.use_pseudo to false.")
            self.use_pseudo = False

        self.prev_by_id: dict[str, torch.Tensor] = {}
        self.prior_ema: torch.Tensor | None = None

        # Static datasets/loaders: built once for all rounds.
        self.target_adapt_gt = build_target_adapt_base(cfg)
        self.static_eval_loaders = build_static_eval_loaders(cfg)
        self.run_dir = _resolve_method_run_dir(cfg)

    def compute_round_budgets(self, budget_total: int, num_rounds: int) -> list[int]:
        if budget_total <= 0 or num_rounds <= 0:
            return []
        base = budget_total // num_rounds
        if base == 0:
            effective_rounds = min(num_rounds, budget_total)
            warnings.warn(
                f"Budget too small for requested rounds: budget_total={budget_total}, "
                f"num_rounds={num_rounds}. Running {effective_rounds} rounds with budget 1 each."
            )
            return [1] * effective_rounds
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

        pred = prob.argmax(dim=1)
        score = torch.zeros((len(sample_ids),), dtype=torch.float32)
        if self.score_use_margin:
            margin = margin_from_prob(prob)
            u = rank_norm(1.0 - margin)
            score += self.w_margin * u
        else:
            margin = torch.empty((0,), dtype=torch.float32)

        if self.score_use_change:
            changes = []
            for i, sid in enumerate(sample_ids):
                prev = self.prev_by_id.get(str(sid))
                if prev is None:
                    changes.append(torch.tensor(0.0))
                else:
                    changes.append(change_l1(prob[i : i + 1], prev[None, :])[0])
                self.prev_by_id[str(sid)] = prob[i].detach().clone()
            change = torch.stack(changes)
            c = rank_norm(change)
            score += self.w_change * c
        else:
            change = torch.empty((0,), dtype=torch.float32)

        return InferenceResult(sample_ids, logits, pred, margin, change, score, prior)

    @torch.no_grad()
    def random_select_pool(self, select_pool_loader: DataLoader) -> InferenceResult:
        sample_ids: list[str] = []
        for batch in select_pool_loader:
            sample_ids.extend(list(batch["sample_id"]))

        n = len(sample_ids)
        if n == 0:
            empty = torch.empty((0, self.num_classes), dtype=torch.float32)
            em = torch.empty((0,), dtype=torch.float32)
            return InferenceResult([], empty, torch.empty((0,), dtype=torch.long), em, em, em, None)

        logits = torch.zeros((n, self.num_classes), dtype=torch.float32)
        pred = torch.randint(low=0, high=self.num_classes, size=(n,), dtype=torch.long)
        score = torch.rand((n,), dtype=torch.float32)
        em = torch.empty((0,), dtype=torch.float32)
        return InferenceResult(sample_ids, logits, pred, em, em, score, None)

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
        model_state_in: dict[str, torch.Tensor] | None = None,
        save_ckpt: bool = True,
    ) -> tuple[str, RoundState, dict[str, Any], dict[str, torch.Tensor] | None]:
        n_pool = len(self.target_adapt_gt) - len(state_in.queried_ids)
        print(
            f"[Round {round_idx + 1}] pool={n_pool} queried={len(state_in.queried_ids)} "
            f"budget={budget_k} mode={'random' if self.random_pick else 'score'}"
        )
        # A) load model M_{r-1}
        model = build_model(self.cfg).to(self.device)
        if model_state_in is not None:
            model.load_state_dict(model_state_in, strict=True)
        else:
            strict = False if round_idx == 0 else True
            load_checkpoint(ckpt_in, model, load_optimizer=False, strict=strict)

        # B) selection pool from current state (exclude queried only)
        select_pool_loader = self.build_select_pool_loader(state_in)

        # C) infer + plan + state update
        infer = self.random_select_pool(select_pool_loader) if self.random_pick else self.infer_select_pool(model, select_pool_loader)
        new_queried_ids, pseudo_store_next, aux = self.plan_round(state_in, infer, budget_k)
        state_out = self.apply_plan(state_in, new_queried_ids, pseudo_store_next, round_idx)
        print(
            f"[Round {round_idx + 1}] selected={len(new_queried_ids)} pseudo={len(pseudo_store_next)} "
            f"budget_used={state_out.budget_used}/{state_out.budget_total}"
        )

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
        if save_ckpt:
            ckpt_dir = round_dir / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_out = str(ckpt_dir / "ckpt_last.pt")
            ckpt_best = str(ckpt_dir / "ckpt_best.pt")
        else:
            ckpt_out = ckpt_in
            ckpt_best = None
        summary = trainer.fit(
            train_loaders=train_loaders,
            eval_loaders=self.static_eval_loaders,
            max_epochs=self.round_epochs,
            ckpt_last_path=ckpt_out if save_ckpt else None,
            ckpt_best_path=ckpt_best,
            monitor_loader="target_test",
            monitor_metric="acc_top1",
            ckpt_extra={"round": round_idx, "ckpt_in": ckpt_in},
            train_log_path=str(round_dir / "train_log.jsonl"),
            log_every_iters=int(getattr(self.cfg.train, "log_every_iters", 0)),
        )
        eval_metrics = summary.eval_history[-1] if summary.eval_history else {}
        for ep, (tr, ev) in enumerate(zip(summary.train_history, summary.eval_history), start=1):
            loss = float(tr.get("loss", 0.0))
            tgt_top1 = float(ev.get("target_test", {}).get("acc_top1", 0.0))
            src_top1 = ev.get("source_val", {}).get("acc_top1", None)
            src_txt = "" if src_top1 is None else f" src_val_top1={float(src_top1) * 100:.2f}%"
            print(
                f"[Round {round_idx + 1}][Epoch {ep}/{self.round_epochs}] "
                f"loss={loss:.4f} target_top1={tgt_top1 * 100:.2f}%{src_txt}"
            )
        if summary.best_epoch is not None and summary.best_score is not None:
            print(
                f"[Round {round_idx + 1}] best target_top1={summary.best_score * 100:.2f}% "
                f"at epoch {summary.best_epoch}"
            )

        save_json(
            round_dir / "metrics.json",
            {
                "round_idx": round_idx,
                "selected_count": len(new_queried_ids),
                "pseudo_count": len(pseudo_store_next),
                "budget_used": state_out.budget_used,
                "budget_total": state_out.budget_total,
                "train_last": summary.train_history[-1] if summary.train_history else {},
                "eval_last": eval_metrics,
                "eval_history": summary.eval_history,
                "best_epoch": summary.best_epoch,
                "best_score": summary.best_score,
                "monitor_metric": "target_test.acc_top1",
                "ckpt_saved": bool(save_ckpt),
            },
        )
        save_round_state(str(self.run_dir / "state_last.json"), state_out)

        next_model_state = None
        if not save_ckpt:
            next_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        return ckpt_out, state_out, {"eval": eval_metrics}, next_model_state

    def run_all_rounds(self, ckpt_init: str, state_init: RoundState) -> tuple[str, RoundState]:
        budget_cfg = getattr(self.cfg.method, "budget_total", state_init.budget_total)
        budget_total = self.resolve_budget_total(budget_cfg)
        rounds = int(self.cfg.method.num_rounds)
        k_list = self.compute_round_budgets(budget_total=budget_total, num_rounds=rounds)
        print(
            f"[Setup] dataset={self.cfg.data.dataset_name} source={self.cfg.data.source_domain} "
            f"target={self.cfg.data.target_domain} total_target={len(self.target_adapt_gt)} "
            f"budget={budget_total} requested_rounds={rounds} effective_rounds={len(k_list)} "
            f"round_epochs={self.round_epochs}"
        )

        ckpt = ckpt_init
        save_ckpt = bool(getattr(self.cfg.train, "save_ckpt", True))
        model_state: dict[str, torch.Tensor] | None = None
        state = RoundState(
            round_idx=state_init.round_idx,
            queried_ids=set(state_init.queried_ids),
            pseudo_store=dict(state_init.pseudo_store),
            budget_total=budget_total,
            budget_used=state_init.budget_used,
        )
        print(f"[Setup] init_ckpt={ckpt_init} save_ckpt={save_ckpt}")
        for r, k in enumerate(k_list):
            ckpt, state, _, model_state = self.run_round(
                r,
                ckpt,
                state,
                k,
                model_state_in=model_state,
                save_ckpt=save_ckpt,
            )
        final_ckpt = ckpt if save_ckpt else "<in-memory>"
        print(f"[Done] final_budget_used={state.budget_used}/{state.budget_total} final_ckpt={final_ckpt}")
        return ckpt, state


# Backward-compatible alias used by existing imports/callers.
OurMethod = RoundAdaptationMethod
