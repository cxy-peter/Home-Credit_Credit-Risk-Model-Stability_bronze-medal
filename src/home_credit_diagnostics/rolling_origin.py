from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RollingOriginConfig:
    """Time-replay configuration for Track B model validation.

    ``label_maturity_lag_weeks`` is a simulation assumption, not a claim about
    the real Home Credit label-availability process.
    """

    label_maturity_lag_weeks: int = 8
    validation_weeks: int = 4
    step_weeks: int = 4
    min_mature_train_weeks: int = 32
    inner_early_stopping_weeks: int = 4
    outer_quarantine_weeks: int = 12
    max_folds: int | None = None


@dataclass(frozen=True)
class RollingFold:
    fold_id: int
    replay_origin_week: int | float
    fit_weeks: tuple[int | float, ...]
    early_stopping_weeks: tuple[int | float, ...]
    mature_train_weeks: tuple[int | float, ...]
    shadow_weeks: tuple[int | float, ...]
    validation_weeks: tuple[int | float, ...]

    @property
    def train_week_min(self) -> int | float:
        return self.mature_train_weeks[0]

    @property
    def train_week_max(self) -> int | float:
        return self.mature_train_weeks[-1]

    @property
    def validation_week_min(self) -> int | float:
        return self.validation_weeks[0]

    @property
    def validation_week_max(self) -> int | float:
        return self.validation_weeks[-1]


@dataclass(frozen=True)
class QuarantinePlan:
    mature_train_weeks: tuple[int | float, ...]
    shadow_weeks: tuple[int | float, ...]
    quarantine_weeks: tuple[int | float, ...]


def _sorted_unique_weeks(week_values: Iterable[int | float]) -> np.ndarray:
    try:
        weeks = np.asarray(list(week_values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Week values must be numeric") from exc
    if weeks.size == 0:
        raise ValueError("At least one week value is required")
    if not np.isfinite(weeks).all():
        raise ValueError("Week values must be finite")
    rounded = np.rint(weeks)
    if not np.allclose(weeks, rounded):
        raise ValueError("Week values must be integer complete-week identifiers")
    ordered = np.sort(np.unique(rounded.astype(np.int64)))
    if len(ordered) > 1 and not np.all(np.diff(ordered) == 1):
        raise ValueError(
            "Week values must be consecutive so label maturity represents real elapsed weeks"
        )
    return ordered


def build_rolling_origin_folds(
    week_values: Iterable[int | float],
    config: RollingOriginConfig | None = None,
) -> tuple[list[RollingFold], QuarantinePlan]:
    """Build expanding, complete-week folds with a label-maturity gap.

    Validation blocks are anchored backwards from the latest block whose labels
    can mature before the outer quarantine starts.  This leaves a final shadow
    interval immediately before the quarantine and never splits rows within a
    week.
    """

    config = config or RollingOriginConfig()
    integer_fields = {
        "label_maturity_lag_weeks": config.label_maturity_lag_weeks,
        "validation_weeks": config.validation_weeks,
        "step_weeks": config.step_weeks,
        "min_mature_train_weeks": config.min_mature_train_weeks,
        "inner_early_stopping_weeks": config.inner_early_stopping_weeks,
        "outer_quarantine_weeks": config.outer_quarantine_weeks,
    }
    if any(value < 0 for value in integer_fields.values()):
        raise ValueError(f"Rolling-origin values must be non-negative: {integer_fields}")
    if config.validation_weeks == 0 or config.step_weeks == 0:
        raise ValueError("validation_weeks and step_weeks must be positive")
    if config.label_maturity_lag_weeks == 0:
        raise ValueError("label_maturity_lag_weeks must be positive for Track B replay")
    if config.inner_early_stopping_weeks == 0:
        raise ValueError("inner_early_stopping_weeks must be positive")
    if config.outer_quarantine_weeks == 0:
        raise ValueError("outer_quarantine_weeks must be positive")
    if config.step_weeks != config.validation_weeks:
        raise ValueError(
            "step_weeks must equal validation_weeks so eligible weeks have exact, gap-free OOF coverage"
        )
    if config.max_folds is not None and config.max_folds <= 0:
        raise ValueError("max_folds must be positive when supplied")

    weeks = _sorted_unique_weeks(week_values)
    minimum_required = (
        config.min_mature_train_weeks
        + config.label_maturity_lag_weeks
        + config.validation_weeks
        + config.label_maturity_lag_weeks
        + config.outer_quarantine_weeks
    )
    if len(weeks) < minimum_required:
        raise ValueError(
            "Not enough complete weeks for mature training, rolling validation, "
            f"final label maturity, and quarantine: have={len(weeks)}, need>={minimum_required}"
        )
    if config.inner_early_stopping_weeks >= config.min_mature_train_weeks:
        raise ValueError(
            "inner_early_stopping_weeks must be smaller than min_mature_train_weeks"
        )

    quarantine_start = len(weeks) - config.outer_quarantine_weeks
    final_mature_end = quarantine_start - config.label_maturity_lag_weeks
    if final_mature_end <= 0:
        raise ValueError("The quarantine leaves no mature training weeks")

    quarantine = QuarantinePlan(
        mature_train_weeks=tuple(weeks[:final_mature_end].tolist()),
        shadow_weeks=tuple(weeks[final_mature_end:quarantine_start].tolist()),
        quarantine_weeks=tuple(weeks[quarantine_start:].tolist()),
    )

    latest_validation_end = final_mature_end
    latest_start = latest_validation_end - config.validation_weeks
    starts: list[int] = []
    start = latest_start
    while start - config.label_maturity_lag_weeks >= config.min_mature_train_weeks:
        starts.append(start)
        start -= config.step_weeks
    starts.reverse()
    if config.max_folds is not None:
        starts = starts[: config.max_folds]
    if not starts:
        raise ValueError("No eligible rolling-origin folds were produced")

    folds: list[RollingFold] = []
    for fold_number, validation_start in enumerate(starts, start=1):
        validation_end = validation_start + config.validation_weeks
        if validation_end > latest_validation_end:
            raise AssertionError("Validation extends beyond the frozen development window")
        mature_train_end = validation_start - config.label_maturity_lag_weeks
        inner_start = mature_train_end - config.inner_early_stopping_weeks
        if inner_start <= 0:
            raise ValueError("The fold leaves no rows for model fitting before early stopping")
        mature_train_weeks = tuple(weeks[:mature_train_end].tolist())
        shadow_weeks = tuple(weeks[mature_train_end:validation_start].tolist())
        validation_block = tuple(weeks[validation_start:validation_end].tolist())
        fold = RollingFold(
            fold_id=fold_number,
            replay_origin_week=weeks[validation_start - 1].item(),
            fit_weeks=tuple(weeks[:inner_start].tolist()),
            early_stopping_weeks=tuple(weeks[inner_start:mature_train_end].tolist()),
            mature_train_weeks=mature_train_weeks,
            shadow_weeks=shadow_weeks,
            validation_weeks=validation_block,
        )
        if len(fold.shadow_weeks) != config.label_maturity_lag_weeks:
            raise AssertionError("Each fold must retain the full label-maturity shadow gap")
        if set(fold.mature_train_weeks) & set(fold.validation_weeks):
            raise AssertionError("Mature training and validation weeks overlap")
        folds.append(fold)

    validation_weeks = [week for fold in folds for week in fold.validation_weeks]
    if len(validation_weeks) != len(set(validation_weeks)):
        raise AssertionError("A validation week appears in more than one fold")
    return folds, quarantine
