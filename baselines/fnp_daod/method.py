"""Round orchestration for the isolated FNP DAOD baseline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
import torch

from src.data.daod import build_daod_dataset
from src.models import build_daod_model

from .acquisition import apply_acquisition
from .config import resolve_fnp_daod_run_dir, resolve_fnp_daod_source_ckpt_path
from .dino_hooks import extract_pooled_backbone_features, mc_dropout_query_statistics
from .fnpm import FalseNegativePredictionModule, fit_fnpm, normalize_fn_count
from .metrics import count_false_negatives
from .state import FNPDAODState, FNPRoundPlan, FNPSamplePlan, save_fnp_state
from .utils import maybe_empty_cuda_cache, resolve_aux_device, save_json, save_resolved_config


def _resolve_budget_total(cfg: Any, total_target: int) -> int:
    budget_cfg = getattr(cfg.method, "budget_total", 0)
    if isinstance(budget_cfg, float) and 0.0 < budget_cfg <= 1.0:
        return max(1, int(total_target * budget_cfg))
    return max(0, int(budget_cfg))


def _compute_round_budgets(budget_total: int, num_rounds: int) -> list[int]:
    if budget_total <= 0 or num_rounds <= 0:
        return []
    base = budget_total // num_rounds
    if base == 0:
        effective_rounds = min(num_rounds, budget_total)
        return [1] * effective_rounds
    budgets = [base] * num_rounds
    budgets[-1] += budget_total - sum(budgets)
    return budgets


class FNPDAODMethod:
    def __init__(
        self,
        cfg: Any,
        device: torch.device,
        *,
        trainer: Any | None = None,
        source_train: list[dict[str, Any]] | None = None,
        target_train: list[dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.selection_device = resolve_aux_device(cfg, device)
        self.run_dir = resolve_fnp_daod_run_dir(cfg)
        if source_train is None:
            source_dataset = build_daod_dataset(cfg, split="source_train", transform=None)
            source_train = [dict(source_dataset[idx]) for idx in range(len(source_dataset))]
        if target_train is None:
            target_dataset = build_daod_dataset(cfg, split="target_train", transform=None)
            target_train = [dict(target_dataset[idx]) for idx in range(len(target_dataset))]
        self.source_train = source_train
        self.target_train = target_train
        if trainer is None:
            from .trainer import FNPDAODTrainer

            trainer = FNPDAODTrainer(cfg=cfg, device=device)
        self.trainer = trainer

    def build_initial_state(self) -> FNPDAODState:
        source_ckpt = resolve_fnp_daod_source_ckpt_path(
            self.cfg,
            which=str(getattr(getattr(self.cfg, "detector", object()), "source_ckpt", "best")),
        )
        if not Path(source_ckpt).exists():
            raise FileNotFoundError(
                f"Missing DAOD source checkpoint for FNP baseline: {source_ckpt}\n"
                "Run the DAOD source training first, or update detector.source_ckpt to an existing checkpoint."
            )
        return FNPDAODState(
            round_idx=0,
            queried_ids=set(),
            budget_total=_resolve_budget_total(self.cfg, len(self.target_train)),
            budget_used=0,
            student_checkpoint=str(source_ckpt),
            teacher_checkpoint=str(source_ckpt),
            optimizer_checkpoint=None,
            scheduler_checkpoint=None,
            discriminator_checkpoint=None,
            teacher_discriminator_checkpoint=None,
            initialized=False,
            global_step=0,
        )

    def _train_fnpm(self, *, state: FNPDAODState, round_dir: Path) -> tuple[FalseNegativePredictionModule, dict[str, Any]]:
        fnpm_cfg = getattr(self.cfg.method, "fnpm", object())
        target_cap = float(getattr(fnpm_cfg, "target_cap", 10.0))
        max_source_samples = int(getattr(fnpm_cfg, "max_source_samples", 0))
        max_target_samples = int(getattr(fnpm_cfg, "max_labeled_target_samples", 0))

        teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=self.selection_device)
        DetectionCheckpointer(teacher_adapter.model).load(str(state.teacher_checkpoint))
        teacher_adapter.model.eval()

        source_samples = self.source_train[: max_source_samples or None]
        labeled_target = [sample for sample in self.target_train if sample["sample_id"] in state.queried_ids]
        if max_target_samples > 0:
            labeled_target = labeled_target[:max_target_samples]
        training_samples = [*source_samples, *labeled_target]
        print(
            "[FNP-DAOD][FNPM] "
            f"round={state.round_idx} "
            f"source_samples={len(source_samples)} "
            f"labeled_target_samples={len(labeled_target)}"
        )

        feature_rows = []
        target_rows = []
        training_records = []
        for sample in training_samples:
            pooled = extract_pooled_backbone_features(teacher_adapter, sample, with_grad=False)[0]
            mc_stats = mc_dropout_query_statistics(
                teacher_adapter,
                sample,
                num_passes=int(getattr(self.cfg.method.acquisition, "mc_dropout_passes_train", getattr(self.cfg.method.acquisition, "mc_dropout_passes", 4))),
                dropout_rate=float(getattr(self.cfg.method.acquisition, "dropout_rate", 0.1)),
                score_floor=float(getattr(self.cfg.method.acquisition, "score_floor", 0.05)),
                max_queries=int(getattr(self.cfg.method.acquisition, "max_queries", 300)),
            )
            fn_count = count_false_negatives(
                sample["annotations"],
                mc_stats["rows"],
                iou_thresh=float(getattr(fnpm_cfg, "match_iou_thresh", 0.5)),
                score_floor=float(getattr(self.cfg.method.acquisition, "score_floor", 0.05)),
            )
            feature_rows.append(pooled.float())
            target_rows.append(normalize_fn_count(fn_count, target_cap=target_cap))
            training_records.append(
                {
                    "sample_id": sample["sample_id"],
                    "fn_count": int(fn_count),
                    "target": float(target_rows[-1]),
                }
            )

        if not feature_rows:
            raise RuntimeError("FNPM training requires at least one labeled sample. Check source dataset availability.")

        features = torch.stack(feature_rows, dim=0)
        targets = torch.tensor(target_rows, dtype=torch.float32)
        model = FalseNegativePredictionModule(
            input_dim=int(features.shape[-1]),
            hidden_dim=int(getattr(fnpm_cfg, "hidden_dim", 256)),
            num_layers=int(getattr(fnpm_cfg, "num_layers", 3)),
            dropout=float(getattr(fnpm_cfg, "dropout", 0.1)),
        )
        history = fit_fnpm(model, features=features, targets=targets, cfg=self.cfg, device=torch.device("cpu"))
        model.to(torch.device("cpu"))
        del teacher_adapter
        maybe_empty_cuda_cache()
        fnpm_ckpt_path = round_dir / "fnpm.pt"
        torch.save(model.state_dict(), fnpm_ckpt_path)
        summary = {
            "checkpoint": str(fnpm_ckpt_path),
            "history": history,
            "num_samples": len(training_samples),
            "records": training_records,
        }
        save_json(round_dir / "fnpm_summary.json", summary)
        return model, summary

    def _train_domain_discriminator(self, *, state: FNPDAODState, round_dir: Path) -> tuple[Any, dict[str, Any]]:
        unlabeled_target = [sample for sample in self.target_train if sample["sample_id"] not in state.queried_ids]
        print(
            "[FNP-DAOD][domain-disc] "
            f"round={state.round_idx} "
            f"source_samples={len(self.source_train)} "
            f"target_samples={len(unlabeled_target)}"
        )
        return self.trainer.train_domain_discriminator(
            teacher_checkpoint=str(state.teacher_checkpoint),
            run_dir=round_dir,
            source_samples=self.source_train,
            target_samples=unlabeled_target,
        )

    def plan_round(self, *, state: FNPDAODState, budget_k: int) -> FNPRoundPlan:
        round_dir = self.run_dir / f"round_{state.round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[FNP-DAOD][select] "
            f"round={state.round_idx} "
            f"budget_k={budget_k} "
            f"labeled_target={len(state.queried_ids)}"
        )
        fnpm_model, fnpm_summary = self._train_fnpm(state=state, round_dir=round_dir)
        domain_discriminator, domain_summary = self._train_domain_discriminator(state=state, round_dir=round_dir)

        teacher_adapter = build_daod_model(self.cfg, load_weights=False, device=self.selection_device)
        DetectionCheckpointer(teacher_adapter.model).load(str(state.teacher_checkpoint))
        teacher_adapter.model.eval()

        unlabeled_candidates = [sample for sample in self.target_train if sample["sample_id"] not in state.queried_ids]
        max_target_samples = int(getattr(self.cfg.method.acquisition, "max_target_samples", 0))
        if max_target_samples > 0:
            unlabeled_candidates = unlabeled_candidates[:max_target_samples]

        records = []
        fnpm_model.eval()
        for sample in unlabeled_candidates:
            pooled = extract_pooled_backbone_features(teacher_adapter, sample, with_grad=False)[0]
            fn_score = float(fnpm_model(pooled.unsqueeze(0)).detach().cpu().item())
            mc_stats = mc_dropout_query_statistics(
                teacher_adapter,
                sample,
                num_passes=int(getattr(self.cfg.method.acquisition, "mc_dropout_passes", 4)),
                dropout_rate=float(getattr(self.cfg.method.acquisition, "dropout_rate", 0.1)),
                score_floor=float(getattr(self.cfg.method.acquisition, "score_floor", 0.05)),
                max_queries=int(getattr(self.cfg.method.acquisition, "max_queries", 300)),
            )
            with torch.no_grad():
                logit = domain_discriminator(pooled.unsqueeze(0).cpu()).squeeze(0)
                prob_target = float(torch.sigmoid(logit).detach().cpu().item())
                div_score = float((1.0 - prob_target) / max(prob_target, 1e-6))
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "file_name": sample["file_name"],
                    "metrics": {
                        "fn": float(fn_score),
                        "loc": float(mc_stats["loc_score"]),
                        "ent": float(mc_stats["ent_score"]),
                        "div": float(div_score),
                    },
                }
            )

        scored_records = apply_acquisition(records)
        queried_ids = [record["sample_id"] for record in scored_records[:budget_k]]
        sample_plans = [
            FNPSamplePlan(
                sample_id=str(record["sample_id"]),
                file_name=str(record["file_name"]),
                acquisition_score=float(record["acquisition_score"]),
                metrics={key: float(value) for key, value in record["metrics"].items()},
                normalized_metrics={key: float(value) for key, value in record["normalized_metrics"].items()},
            )
            for record in scored_records
        ]
        save_json(
            round_dir / "selection_summary.json",
            {
                "budget_k": int(budget_k),
                "num_candidates": len(unlabeled_candidates),
                "fnpm": fnpm_summary,
                "domain_discriminator": domain_summary,
                "queried_ids": queried_ids,
                "sample_plans": [asdict(plan) for plan in sample_plans],
            },
        )
        del teacher_adapter
        maybe_empty_cuda_cache()
        print(
            "[FNP-DAOD][select] "
            f"round={state.round_idx} "
            f"selected={len(queried_ids)} "
            f"candidates={len(unlabeled_candidates)}"
        )
        return FNPRoundPlan(
            round_idx=state.round_idx,
            queried_ids=queried_ids,
            sample_plans=sample_plans,
        )

    def _run_initial_adaptation(self, state: FNPDAODState) -> FNPDAODState:
        init_cfg = getattr(self.cfg.method, "initial_adaptation", object())
        if not bool(getattr(init_cfg, "enabled", True)):
            state.initialized = True
            return state

        print(
            "[FNP-DAOD][init] "
            "skipped extra initial adaptation and reusing the source-trained checkpoint directly"
        )
        state.initialized = True
        save_fnp_state(self.run_dir / "state_last.json", state)
        return state

    def run_round(self, *, state: FNPDAODState, budget_k: int) -> FNPDAODState:
        print(
            "[FNP-DAOD][round] "
            f"start_round={state.round_idx} "
            f"budget_used={state.budget_used}/{state.budget_total}"
        )
        plan = self.plan_round(state=state, budget_k=budget_k)
        round_dir = self.run_dir / f"round_{state.round_idx}"
        save_json(
            round_dir / "plan.json",
            {
                "round_idx": int(plan.round_idx),
                "queried_ids": list(plan.queried_ids),
                "sample_plans": [asdict(sample_plan) for sample_plan in plan.sample_plans],
            },
        )

        labeled_target_ids = set(state.queried_ids).union(plan.queried_ids)
        backend_plan = SimpleNamespace(
            round_idx=state.round_idx,
            queried_ids=list(plan.queried_ids),
        )
        summary = self.trainer.fit_round(
            run_dir=round_dir / "train",
            state_in=state,
            plan=backend_plan,
        )
        state_out = FNPDAODState(
            round_idx=state.round_idx + 1,
            queried_ids=labeled_target_ids,
            budget_total=state.budget_total,
            budget_used=state.budget_used + len(plan.queried_ids),
            student_checkpoint=str(summary["student_checkpoint"]),
            teacher_checkpoint=str(summary["teacher_checkpoint"]),
            optimizer_checkpoint=summary.get("optimizer_checkpoint"),
            scheduler_checkpoint=summary.get("scheduler_checkpoint"),
            discriminator_checkpoint=None,
            teacher_discriminator_checkpoint=None,
            initialized=True,
            global_step=int(summary["global_step"]),
        )
        save_fnp_state(self.run_dir / "state_last.json", state_out)
        save_json(
            round_dir / "summary.json",
            {
                "round_idx": int(state.round_idx),
                "budget_k": int(budget_k),
                "selected_count": len(plan.queried_ids),
                "budget_used": int(state_out.budget_used),
                "budget_total": int(state_out.budget_total),
                "trainer": summary,
            },
        )
        print(
            "[FNP-DAOD][round] "
            f"done_round={state.round_idx} "
            f"budget_used={state_out.budget_used}/{state_out.budget_total}"
        )
        return state_out

    def run_all_rounds(self, *, config_path: str | Path | None = None, state_init: FNPDAODState | None = None) -> FNPDAODState:
        state = state_init or self.build_initial_state()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(self.run_dir / "resolved_config.yaml", self.cfg)
        if config_path is not None:
            config_src = Path(config_path).resolve()
            if config_src.is_file():
                shutil.copy2(config_src, self.run_dir / "config.yaml")

        if not state.initialized:
            state = self._run_initial_adaptation(state)

        budget_schedule = _compute_round_budgets(
            _resolve_budget_total(self.cfg, len(self.target_train)),
            int(getattr(self.cfg.method, "num_rounds", 1)),
        )
        print(
            "[FNP-DAOD] "
            f"run_dir={self.run_dir} "
            f"rounds={len(budget_schedule)} "
            f"budget_schedule={budget_schedule}"
        )
        for budget_k in budget_schedule:
            state = self.run_round(state=state, budget_k=budget_k)
        return state
