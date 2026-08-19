from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _clean_binary_inputs(y_true: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(y_true, errors="coerce")
    scores = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & scores.notna()
    return y.loc[valid].astype(int).to_numpy(), scores.loc[valid].astype(float).to_numpy()


def tie_aware_lift(y_true: np.ndarray, score: np.ndarray, fraction: float) -> float:
    """Calculate Lift@fraction without arbitrary row ordering inside score ties."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if y_true.size == 0:
        return float("nan")
    overall_bad_rate = float(np.mean(y_true))
    if overall_bad_rate <= 0:
        return float("nan")
    capacity = y_true.size * fraction
    grouped = (
        pd.DataFrame({"target": y_true, "score": score})
        .groupby("score", sort=True, dropna=False)["target"]
        .agg(["count", "sum"])
        .sort_index(ascending=False)
    )
    remaining = capacity
    captured_bad = 0.0
    for row in grouped.itertuples():
        if remaining <= 0:
            break
        take = min(float(row.count), remaining)
        captured_bad += float(row.sum) * take / float(row.count)
        remaining -= take
    top_bad_rate = captured_bad / capacity
    return float(top_bad_rate / overall_bad_rate)


def binary_metrics(y_true: pd.Series, score: pd.Series) -> dict[str, float | int]:
    y, scores = _clean_binary_inputs(y_true, score)
    result: dict[str, float | int] = {
        "n_scored": int(y.size),
        "bad_rate": float(np.mean(y)) if y.size else float("nan"),
        "auc": float("nan"),
        "ks": float("nan"),
        "lift_10": float("nan"),
        "lift_20": float("nan"),
    }
    if y.size == 0 or np.unique(y).size < 2:
        return result
    result["auc"] = float(roc_auc_score(y, scores))
    false_positive_rate, true_positive_rate, _ = roc_curve(y, scores)
    result["ks"] = float(np.max(np.abs(true_positive_rate - false_positive_rate)))
    result["lift_10"] = tie_aware_lift(y, scores, 0.10)
    result["lift_20"] = tie_aware_lift(y, scores, 0.20)
    return result


def weekly_stability(
    target: pd.Series,
    score: pd.Series,
    week: pd.Series,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Home Credit weekly Gini stability using sorted, real week numbers."""
    frame = pd.DataFrame({"target": target, "score": score, "week_num": week})
    frame = frame.dropna(subset=["target", "score", "week_num"])
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    frame["week_num"] = pd.to_numeric(frame["week_num"], errors="raise").astype(float)
    rows: list[dict[str, float | int]] = []
    skipped = 0
    for week_value, group in frame.groupby("week_num", sort=True):
        if group["target"].nunique() < 2:
            skipped += 1
            continue
        auc = float(roc_auc_score(group["target"], group["score"]))
        rows.append(
            {
                "week_num": float(week_value),
                "n": int(len(group)),
                "bad_rate": float(group["target"].mean()),
                "auc": auc,
                "gini": 2.0 * auc - 1.0,
            }
        )
    weekly = pd.DataFrame(rows, columns=["week_num", "n", "bad_rate", "auc", "gini"])
    if weekly.empty:
        summary = {
            "stability": float("nan"),
            "mean_weekly_gini": float("nan"),
            "gini_slope": float("nan"),
            "gini_residual_std": float("nan"),
            "weeks_scored": 0,
            "weeks_skipped_single_class": skipped,
        }
        return summary, weekly

    x = weekly["week_num"].to_numpy(dtype=float)
    y = weekly["gini"].to_numpy(dtype=float)
    if len(weekly) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        residual_std = float(np.std(y - (slope * x + intercept)))
    else:
        slope = 0.0
        residual_std = 0.0
    mean_gini = float(np.mean(y))
    stability = mean_gini + 88.0 * min(0.0, float(slope)) - 0.5 * residual_std
    summary = {
        "stability": float(stability),
        "mean_weekly_gini": mean_gini,
        "gini_slope": float(slope),
        "gini_residual_std": residual_std,
        "weeks_scored": int(len(weekly)),
        "weeks_skipped_single_class": int(skipped),
    }
    return summary, weekly
