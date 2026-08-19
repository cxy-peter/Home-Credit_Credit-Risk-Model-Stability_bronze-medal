from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def make_synthetic_customer_data(
    n_rows: int = 12_000,
    n_weeks: int = 60,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic smoke data and simulated out-of-sample scores."""
    if n_rows < n_weeks * 10:
        raise ValueError("n_rows must provide at least ten rows per week")
    rng = np.random.default_rng(seed)
    weeks = np.resize(np.arange(n_weeks), n_rows)
    rng.shuffle(weeks)
    case_id = np.arange(1_000_000, 1_000_000 + n_rows, dtype=np.int64)

    age = np.clip(rng.normal(42, 11, n_rows), 18, 80)
    income = np.exp(rng.normal(10.2, 0.55, n_rows))
    credit = np.exp(rng.normal(9.6, 0.65, n_rows))
    debt_ratio = np.clip(rng.beta(2.0, 5.0, n_rows), 0, 1)
    applications = rng.poisson(1.5, n_rows)
    days_since = np.clip(rng.exponential(80, n_rows), 0, 500)
    sparse_present = rng.random(n_rows) < 0.03
    sparse_signal = np.where(sparse_present, rng.normal(1.0, 0.4, n_rows), np.nan)
    extreme_present = rng.random(n_rows) < 0.002
    extreme_sparse = np.where(extreme_present, rng.normal(0, 1, n_rows), np.nan)
    drift_signal = rng.normal(np.where(weeks >= int(n_weeks * 0.70), 1.4, 0.0), 1.0)

    region = rng.choice(["north", "south", "east", "west"], n_rows, p=[0.28, 0.25, 0.25, 0.22])
    channel = rng.choice(["branch", "web", "partner"], n_rows, p=[0.25, 0.55, 0.20]).astype(object)
    channel[weeks >= int(n_weeks * 0.70)] = rng.choice(
        ["web", "partner", "new_partner"],
        size=int((weeks >= int(n_weeks * 0.70)).sum()),
        p=[0.45, 0.30, 0.25],
    )
    occupation = np.array([f"occupation_{value:03d}" for value in rng.integers(0, 140, n_rows)])
    missing_category = rng.choice(["known", "nan", None], n_rows, p=[0.75, 0.15, 0.10])

    linear_risk = (
        -4.15
        + 2.0 * debt_ratio
        + 0.22 * applications
        + 0.65 * sparse_present
        + 0.38 * (region == "south")
        + 0.30 * (channel == "partner")
        + 0.004 * np.maximum(0, weeks - n_weeks * 0.65)
        - 0.012 * (age - 42)
    )
    probability = _sigmoid(linear_risk)
    target = rng.binomial(1, probability).astype(np.int8)
    frame = pd.DataFrame(
        {
            "case_id": case_id,
            "week_num": weeks,
            "target": target,
            "age_yrs": age,
            "income_amt": income,
            "income_duplicate_corr": income * 1.02 + rng.normal(0, income.std() * 0.01, n_rows),
            "credit_amt": credit,
            "debt_ratio": debt_ratio,
            "applications_90d": applications,
            "days_since_last_event": days_since,
            "region_cat": region,
            "channel_cat": channel,
            "occupation_cat": occupation,
            "sparse_signal": sparse_signal,
            "extreme_sparse": extreme_sparse,
            "drift_signal": drift_signal,
            "missing_category": missing_category,
            "constant_feature": 1,
        }
    )
    order = rng.permutation(n_rows)
    frame = frame.iloc[order].reset_index(drop=True)
    latent_ordered = linear_risk[order]
    oof = pd.DataFrame(
        {
            "case_id": frame["case_id"],
            "existing_simulated_oof": _sigmoid(latent_ordered + rng.normal(0, 0.85, n_rows)),
            "challenger_simulated_oof": _sigmoid(latent_ordered + rng.normal(0, 1.10, n_rows)),
        }
    )
    return frame, oof


def write_synthetic_demo(
    data_dir: str | Path,
    *,
    n_rows: int = 12_000,
    n_weeks: int = 60,
    seed: int = 42,
) -> tuple[Path, Path]:
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, oof = make_synthetic_customer_data(n_rows=n_rows, n_weeks=n_weeks, seed=seed)
    train_path = output / "synthetic_customer_features.parquet"
    oof_path = output / "synthetic_simulated_oof.parquet"
    frame.to_parquet(train_path, index=False)
    oof.to_parquet(oof_path, index=False)
    return train_path, oof_path
