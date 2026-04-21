"""Serializable state for the isolated FNP baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class FNPSamplePlan:
    sample_id: str
    file_name: str
    acquisition_score: float
    metrics: dict[str, float]
    normalized_metrics: dict[str, float]


@dataclass
class FNPRoundPlan:
    round_idx: int
    queried_ids: list[str]
    sample_plans: list[FNPSamplePlan]


@dataclass
class FNPDAODState:
    round_idx: int
    queried_ids: set[str]
    budget_total: int
    budget_used: int
    student_checkpoint: str
    teacher_checkpoint: str
    optimizer_checkpoint: str | None = None
    scheduler_checkpoint: str | None = None
    discriminator_checkpoint: str | None = None
    teacher_discriminator_checkpoint: str | None = None
    initialized: bool = False
    global_step: int = 0


def save_fnp_state(path: str | Path, state: FNPDAODState) -> None:
    payload = asdict(state)
    payload["queried_ids"] = sorted(state.queried_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_fnp_state(path: str | Path) -> FNPDAODState:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["queried_ids"] = set(payload["queried_ids"])
    return FNPDAODState(**payload)
