"""Event-driven DAOD method with step-wise label injection."""

from __future__ import annotations

from dataclasses import asdict
import torch

from src.engine.daod_stepwise_injection_trainer import DAODStepwiseInjectionTrainer
from src.engine.utils import save_json
from src.methods.daod_method import (
    DAODRoundMethod,
    DAODRoundPlan,
    DAODRoundState,
    _compute_round_budgets,
    _resolve_budget_total,
    build_default_daod_round_state,
    save_daod_round_state,
)


class DAODStepwiseInjectionMethod(DAODRoundMethod):
    """Continuous DAOD with label-injection events inside one training run."""

    def __init__(self, cfg, device: torch.device, trainer=None) -> None:
        super().__init__(
            cfg=cfg,
            device=device,
            trainer=trainer or DAODStepwiseInjectionTrainer(cfg=cfg, device=device),
        )

    def run_all_rounds(self, source_ckpt: str, state_init: DAODRoundState | None = None) -> DAODRoundState:
        budget_total = _resolve_budget_total(self.cfg, total_target=len(self.target_train))
        num_rounds = int(getattr(self.cfg.method, "num_rounds", 1))
        budget_schedule = _compute_round_budgets(budget_total, num_rounds)

        if state_init is None:
            state = DAODRoundState(
                round_idx=0,
                queried_ids=set(),
                budget_total=budget_total,
                budget_used=0,
                teacher_checkpoint=str(source_ckpt),
                student_checkpoint=str(source_ckpt),
                optimizer_checkpoint=None,
                scheduler_checkpoint=None,
                global_step=0,
            )
        else:
            state = state_init

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if budget_schedule:
            initial_plan = self.plan_round(state, budget_schedule[0])
        else:
            initial_plan = DAODRoundPlan(
                round_idx=0,
                queried_ids=[],
                sample_plans=[],
            )

        def _plan_callback(plan_state: DAODRoundState, budget_k: int, student_ckpt_path: str, stage_idx: int):
            adjusted_state = DAODRoundState(
                round_idx=stage_idx,
                queried_ids=set(plan_state.queried_ids),
                budget_total=plan_state.budget_total,
                budget_used=plan_state.budget_used,
                teacher_checkpoint=plan_state.teacher_checkpoint,
                student_checkpoint=student_ckpt_path,
                optimizer_checkpoint=plan_state.optimizer_checkpoint,
                scheduler_checkpoint=plan_state.scheduler_checkpoint,
                global_step=plan_state.global_step,
            )
            return self.plan_round(adjusted_state, budget_k)

        trainer_summary = self.trainer.fit_with_injections(
            cfg=self.cfg,
            run_dir=self.run_dir,
            state_in=state,
            initial_plan=initial_plan,
            remaining_budgets=budget_schedule[1:] if budget_schedule else [],
            plan_callback=_plan_callback,
        )

        state_out = DAODRoundState(
            round_idx=len(budget_schedule),
            queried_ids=set(trainer_summary.get("queried_ids", [])),
            budget_total=state.budget_total,
            budget_used=len(trainer_summary.get("queried_ids", [])),
            teacher_checkpoint=str(trainer_summary["teacher_checkpoint"]),
            student_checkpoint=str(trainer_summary["student_checkpoint"]),
            optimizer_checkpoint=trainer_summary.get("optimizer_checkpoint"),
            scheduler_checkpoint=trainer_summary.get("scheduler_checkpoint"),
            global_step=int(trainer_summary.get("global_step", state.global_step)),
        )
        save_daod_round_state(self.run_dir / "state_last.json", state_out)
        save_json(
            self.run_dir / "summary.json",
            {
                "mode": "stepwise_injection",
                "budget_schedule": budget_schedule,
                "initial_plan": {
                    "round_idx": initial_plan.round_idx,
                    "queried_ids": initial_plan.queried_ids,
                    "sample_plans": [asdict(sample_plan) for sample_plan in initial_plan.sample_plans],
                },
                "trainer": trainer_summary,
            },
        )
        return state_out


def build_default_daod_stepwise_injection_state(cfg) -> DAODRoundState:
    return build_default_daod_round_state(cfg)
