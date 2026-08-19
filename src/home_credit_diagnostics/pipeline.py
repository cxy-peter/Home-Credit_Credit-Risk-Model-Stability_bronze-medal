from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .binning import MISSING_BUCKET, add_actual_psi, decode_bucket, fit_feature_binner
from .io import restore_named_index_columns, write_csv
from .metrics import binary_metrics, weekly_stability


TIME_PROXY_NAMES = {
    "week_num",
    "month",
    "year",
    "day",
    "date_decision",
    "date",
}


def _feature_type(feature: str, series: pd.Series) -> str:
    upper = feature.upper()
    encoded_category = upper.startswith(("MODE(", "MONTH(", "SEASON(", "WEEKDAY("))
    if encoded_category or is_bool_dtype(series.dtype):
        return "categorical"
    return "numeric" if is_numeric_dtype(series.dtype) else "categorical"


def _is_time_proxy(feature: str) -> bool:
    lower = feature.lower()
    if lower in TIME_PROXY_NAMES or "date_decision" in lower:
        return True
    return bool(
        re.search(
            r"(^|[_.(])(week_num|month|year|weekday|season)(?=$|[_.()])|datemonth|dateyear",
            lower,
        )
    )


@dataclass(frozen=True)
class DiagnosticsConfig:
    id_col: str = "case_id"
    target_col: str = "target"
    week_col: str = "week_num"
    reference_fraction: float = 0.70
    n_bins: int = 10
    max_categories: int = 100
    sparse_threshold: float = 0.95
    extreme_sparse_threshold: float = 0.995
    high_psi_threshold: float = 0.25
    correlation_threshold: float = 0.90
    correlation_sample_rows: int = 50_000
    random_state: int = 42
    dataset_label: str = "User-supplied customer-level data"
    source_url: str | None = None
    synthetic_demo: bool = False


@dataclass
class DiagnosticsRun:
    feature_diagnostics: pd.DataFrame
    bin_diagnostics: pd.DataFrame
    correlation_recommendations: pd.DataFrame
    model_diagnostics: pd.DataFrame
    weekly_model_metrics: pd.DataFrame
    split_summary: pd.DataFrame
    summary: dict[str, object]


def _resolve_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    matches = [column for column in frame.columns if str(column).lower() == requested.lower()]
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise ValueError(f"Required column {requested!r} was not found")
    raise ValueError(f"Column name {requested!r} is ambiguous: {matches}")


def normalize_customer_frame(frame: pd.DataFrame, config: DiagnosticsConfig) -> pd.DataFrame:
    expected = [config.id_col, config.target_col, config.week_col]
    normalized = restore_named_index_columns(frame, expected).copy()
    rename: dict[str, str] = {}
    for requested in expected:
        actual = _resolve_column(normalized, requested)
        if actual != requested:
            rename[actual] = requested
    if rename:
        normalized = normalized.rename(columns=rename)
    if normalized[config.id_col].isna().any():
        raise ValueError(f"{config.id_col} contains missing values")
    if normalized[config.id_col].duplicated().any():
        raise ValueError(f"{config.id_col} must be unique in a customer-level matrix")
    target = pd.to_numeric(normalized[config.target_col], errors="raise")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{config.target_col} must contain only 0 and 1")
    normalized[config.target_col] = target.astype(np.int8)
    week = pd.to_numeric(normalized[config.week_col], errors="raise")
    if week.isna().any():
        raise ValueError(f"{config.week_col} contains missing values")
    normalized[config.week_col] = week
    if normalized[config.week_col].nunique() < 2:
        raise ValueError("At least two real week values are required")
    return normalized


def split_by_real_week(
    frame: pd.DataFrame,
    config: DiagnosticsConfig,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    if not 0 < config.reference_fraction < 1:
        raise ValueError("reference_fraction must be in (0, 1)")
    weeks = np.sort(frame[config.week_col].unique())
    reference_week_count = int(np.floor(len(weeks) * config.reference_fraction))
    reference_week_count = max(1, min(reference_week_count, len(weeks) - 1))
    reference_weeks = weeks[:reference_week_count]
    actual_weeks = weeks[reference_week_count:]
    reference_mask = frame[config.week_col].isin(reference_weeks)
    actual_mask = frame[config.week_col].isin(actual_weeks)
    return reference_mask, actual_mask, reference_weeks, actual_weeks


def _feature_status(row: dict[str, object], config: DiagnosticsConfig) -> str:
    if bool(row["constant_reference"]):
        return "CONSTANT_REVIEW"
    if bool(row["extreme_sparse"]):
        return "EXTREME_SPARSE_REVIEW"
    if bool(row["sparse_rescue_candidate"]):
        return "SPARSE_RESCUE_CANDIDATE"
    if bool(row["is_time_proxy"]):
        return "TIME_PROXY_REVIEW"
    if bool(row["high_drift"]):
        return "HIGH_DRIFT_REVIEW"
    return "KEEP_CANDIDATE"


def _recommend_correlations(
    frame: pd.DataFrame,
    reference_mask: pd.Series,
    features: Sequence[str],
    diagnostics: pd.DataFrame,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    columns = [
        "feature_a",
        "feature_b",
        "spearman",
        "abs_spearman",
        "recommend_keep",
        "recommend_review",
        "reason",
    ]
    type_lookup = diagnostics.set_index("feature")["feature_type"].to_dict()
    proxy_lookup = diagnostics.set_index("feature")["is_time_proxy"].to_dict()
    numeric_features = [
        feature
        for feature in features
        if type_lookup.get(feature) == "numeric" and not bool(proxy_lookup.get(feature))
    ]
    if len(numeric_features) < 2 or config.correlation_sample_rows <= 0:
        return pd.DataFrame(columns=columns)
    reference_positions = np.flatnonzero(reference_mask.to_numpy())
    if reference_positions.size > config.correlation_sample_rows:
        rng = np.random.default_rng(config.random_state)
        reference_positions = np.sort(
            rng.choice(reference_positions, size=config.correlation_sample_rows, replace=False)
        )
    sample = frame.iloc[reference_positions][numeric_features].apply(pd.to_numeric, errors="coerce")
    usable = [column for column in sample.columns if sample[column].nunique(dropna=True) > 1]
    if len(usable) < 2:
        return pd.DataFrame(columns=columns)
    correlation = sample[usable].corr(method="spearman", min_periods=30)
    lookup = diagnostics.set_index("feature").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for left_index, feature_a in enumerate(usable):
        for feature_b in usable[left_index + 1 :]:
            value = correlation.loc[feature_a, feature_b]
            if pd.isna(value) or abs(float(value)) < config.correlation_threshold:
                continue
            stats_a = lookup[feature_a]
            stats_b = lookup[feature_b]
            rank_a = (
                float(stats_a["iv"]),
                -float(stats_a["psi"]),
                -float(stats_a["reference_missing_rate"]),
                feature_a,
            )
            rank_b = (
                float(stats_b["iv"]),
                -float(stats_b["psi"]),
                -float(stats_b["reference_missing_rate"]),
                feature_b,
            )
            keep, review = (feature_a, feature_b) if rank_a >= rank_b else (feature_b, feature_a)
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "spearman": float(value),
                    "abs_spearman": abs(float(value)),
                    "recommend_keep": keep,
                    "recommend_review": review,
                    "reason": "higher IV, then lower PSI/missingness; confirm business meaning before removal",
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["abs_spearman", "feature_a", "feature_b"], ascending=[False, True, True]
    )


def _align_oof(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    score_cols: Sequence[str] | None,
    config: DiagnosticsConfig,
) -> tuple[pd.DataFrame, list[str]]:
    prediction = restore_named_index_columns(oof, [config.id_col]).copy()
    actual_id = _resolve_column(prediction, config.id_col)
    if actual_id != config.id_col:
        prediction = prediction.rename(columns={actual_id: config.id_col})
    if prediction[config.id_col].isna().any() or prediction[config.id_col].duplicated().any():
        raise ValueError("OOF case_id values must be present and unique")
    train_ids = pd.Index(frame[config.id_col])
    prediction_ids = pd.Index(prediction[config.id_col])
    missing = train_ids.difference(prediction_ids)
    extras = prediction_ids.difference(train_ids)
    if len(missing) or len(extras):
        raise ValueError(
            f"OOF alignment must be exact: missing={len(missing)}, extra={len(extras)}"
        )
    if score_cols:
        scores = list(score_cols)
    else:
        scores = [
            column
            for column in prediction.columns
            if column != config.id_col and is_numeric_dtype(prediction[column].dtype)
        ]
    if not scores:
        raise ValueError("No numeric OOF score columns were found")
    absent = [column for column in scores if column not in prediction.columns]
    if absent:
        raise ValueError(f"OOF score columns are missing: {absent}")
    if prediction[scores].isna().any().any():
        raise ValueError("OOF score columns contain missing values")
    metadata = frame[[config.id_col, config.target_col, config.week_col]]
    aligned = metadata.merge(
        prediction[[config.id_col, *scores]],
        on=config.id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    return aligned, scores


def _model_diagnostics(
    frame: pd.DataFrame,
    oof: pd.DataFrame | None,
    score_cols: Sequence[str] | None,
    config: DiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_columns = [
        "score",
        "n_scored",
        "bad_rate",
        "auc",
        "ks",
        "lift_10",
        "lift_20",
        "stability",
        "mean_weekly_gini",
        "gini_slope",
        "gini_residual_std",
        "weeks_scored",
        "weeks_skipped_single_class",
        "provenance",
    ]
    weekly_columns = ["score", "week_num", "n", "bad_rate", "auc", "gini"]
    if oof is None:
        return pd.DataFrame(columns=model_columns), pd.DataFrame(columns=weekly_columns)
    aligned, scores = _align_oof(frame, oof, score_cols, config)
    summaries: list[dict[str, object]] = []
    weekly_frames: list[pd.DataFrame] = []
    for score_col in scores:
        overall = binary_metrics(aligned[config.target_col], aligned[score_col])
        stability, weekly = weekly_stability(
            aligned[config.target_col], aligned[score_col], aligned[config.week_col]
        )
        summaries.append(
            {
                "score": score_col,
                **overall,
                **stability,
                "provenance": (
                    "deterministic synthetic simulated score; smoke validation only"
                    if config.synthetic_demo
                    else "user-supplied, exactly aligned OOF; provenance cannot be inferred from values"
                ),
            }
        )
        weekly.insert(0, "score", score_col)
        weekly_frames.append(weekly)
    model = pd.DataFrame(summaries, columns=model_columns)
    weekly_output = (
        pd.concat(weekly_frames, ignore_index=True)
        if weekly_frames
        else pd.DataFrame(columns=weekly_columns)
    )
    return model, weekly_output


def _format_markdown_rows(frame: pd.DataFrame, columns: Sequence[str], limit: int = 10) -> list[str]:
    if frame.empty:
        return ["No rows."]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [header, divider]
    for record in frame.head(limit).to_dict(orient="records"):
        values: list[str] = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _write_report(run: DiagnosticsRun, output_dir: Path, config: DiagnosticsConfig) -> None:
    summary = run.summary
    if config.synthetic_demo:
        title = "# Home Credit Diagnostics V3 — SYNTHETIC SMOKE TEST ONLY"
        caveat = (
            "> This report validates code paths with deterministic synthetic data. "
            "It is not the final Kaggle-data result and must not be quoted as model performance."
        )
    else:
        title = "# Home Credit Diagnostics V3"
        caveat = "> Results describe the supplied customer-level dataset; the original ML training was not rerun."
    lines = [
        title,
        "",
        caveat,
        "",
        f"Dataset: **{config.dataset_label}**",
        "",
        "## Split",
        "",
        f"- Rows: {summary['rows']:,}",
        f"- Features diagnosed: {summary['features']:,}",
        f"- Real week values: {summary['weeks']:,}",
        f"- Target positives: {summary['bad_count']:,}",
        f"- Bad rate: {summary['bad_rate']:.4%}",
        f"- Reference weeks: {summary['reference_week_min']} to {summary['reference_week_max']}",
        f"- Actual weeks: {summary['actual_week_min']} to {summary['actual_week_max']}",
        "- Bins, category frequency, WOE and IV were fitted on reference weeks only.",
        "- PSI and single-feature AUC/KS/Lift use the untouched later actual weeks.",
        "",
        "## Feature status",
        "",
    ]
    if config.source_url:
        lines.insert(5, f"Source: {config.source_url}")
        lines.insert(6, "")
    for status, count in summary["status_counts"].items():
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Leading actual-period single-feature diagnostics", ""])
    top = run.feature_diagnostics.loc[~run.feature_diagnostics["is_time_proxy"]].sort_values(
        ["auc", "iv"], ascending=False, na_position="last"
    )
    lines.extend(
        _format_markdown_rows(
            top,
            ["feature", "feature_type", "iv", "psi", "auc", "ks", "lift_10", "status"],
        )
    )
    lines.extend(["", "## Optional OOF diagnostics", ""])
    if run.model_diagnostics.empty:
        lines.append(
            "No model metrics were calculated. Supply a genuine, case_id-aligned OOF file with `--oof`; "
            "training-set predictions are not accepted as a substitute."
        )
    else:
        lines.extend(
            _format_markdown_rows(
                run.model_diagnostics,
                ["score", "auc", "ks", "lift_10", "lift_20", "stability", "weeks_scored"],
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This run consumes an existing customer-level parquet. It does not rebuild the original "
            "multi-table features or retrain LightGBM/CatBoost. Correlation output is a review "
            "recommendation, not an automatic feature deletion list.",
            "",
            "AMEX-style first/last/delta aggregation is available separately, but only when an "
            "explicit time column is supplied; parquet row order is never treated as time.",
            "",
            "## AMEX patterns: adopted and excluded",
            "",
            "Adopted: numeric mean/std/min/max, categorical count/nunique/mode, and explicit "
            "time-sorted first/last/last-minus-first/last-minus-mean. Existing OOF is preserved "
            "only when supplied with exact case_id alignment.",
            "",
            "Excluded: row-order-based temporal deltas, the AMEX competition metric, and the "
            "Transformer sequence model. Home Credit retains its real-week Gini stability metric "
            "and does not assume one homogeneous customer sequence.",
            "",
            "The source parquet is not redistributed with these lightweight results. Dataset use "
            "remains subject to its source page and the underlying Home Credit competition rules.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostics(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: DiagnosticsConfig | None = None,
    feature_cols: Sequence[str] | None = None,
    oof: pd.DataFrame | None = None,
    score_cols: Sequence[str] | None = None,
) -> DiagnosticsRun:
    config = config or DiagnosticsConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = normalize_customer_frame(frame, config)
    excluded = {config.id_col, config.target_col, config.week_col}
    if feature_cols is None:
        features = [column for column in data.columns if column not in excluded]
    else:
        features = list(feature_cols)
        absent = [column for column in features if column not in data.columns]
        if absent:
            raise ValueError(f"Requested feature columns are missing: {absent}")
        forbidden = [column for column in features if column in excluded]
        if forbidden:
            raise ValueError(f"Metadata columns cannot be diagnosed as features: {forbidden}")
    if not features:
        raise ValueError("No feature columns are available")

    reference_mask, actual_mask, reference_weeks, actual_weeks = split_by_real_week(data, config)
    reference_target = data.loc[reference_mask, config.target_col]
    actual_target = data.loc[actual_mask, config.target_col]
    feature_rows: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    for feature in features:
        reference = data.loc[reference_mask, feature]
        actual = data.loc[actual_mask, feature]
        fitted = fit_feature_binner(
            feature,
            reference,
            reference_target,
            n_bins=config.n_bins,
            max_categories=config.max_categories,
            feature_type=_feature_type(feature, reference),
        )
        bin_table, psi, actual_buckets = add_actual_psi(fitted, actual)
        bin_table["bucket_display"] = bin_table["bucket"].map(decode_bucket)
        bin_frames.append(bin_table)
        actual_score = fitted.risk_score_from_buckets(actual_buckets)
        metrics = binary_metrics(actual_target, actual_score)
        feature_type = fitted.binner.feature_type
        reference_missing_mask = fitted.reference_buckets.eq(MISSING_BUCKET)
        actual_missing_mask = actual_buckets.eq(MISSING_BUCKET)
        reference_missing = float(reference_missing_mask.mean())
        actual_missing = float(actual_missing_mask.mean())
        reference_unique = int(reference.loc[~reference_missing_mask].nunique())
        row: dict[str, object] = {
            "feature": feature,
            "feature_type": feature_type,
            "reference_missing_rate": reference_missing,
            "actual_missing_rate": actual_missing,
            "missing_rate_shift": actual_missing - reference_missing,
            "reference_unique_nonmissing": reference_unique,
            "constant_reference": reference_unique <= 1,
            "sparse_rescue_candidate": (
                config.sparse_threshold <= reference_missing < config.extreme_sparse_threshold
            ),
            "extreme_sparse": reference_missing >= config.extreme_sparse_threshold,
            "iv": fitted.iv,
            "psi": psi,
            "high_drift": psi >= config.high_psi_threshold,
            "is_time_proxy": _is_time_proxy(feature),
            **metrics,
        }
        row["status"] = _feature_status(row, config)
        feature_rows.append(row)

    feature_diagnostics = pd.DataFrame(feature_rows).sort_values(
        ["status", "iv", "feature"], ascending=[True, False, True]
    )
    bin_diagnostics = pd.concat(bin_frames, ignore_index=True)
    correlations = _recommend_correlations(
        data, reference_mask, features, feature_diagnostics, config
    )
    model_diagnostics, weekly_model_metrics = _model_diagnostics(
        data, oof, score_cols, config
    )
    split_summary = pd.DataFrame(
        [
            {
                "period": "reference",
                "week_min": reference_weeks[0],
                "week_max": reference_weeks[-1],
                "week_count": len(reference_weeks),
                "row_count": int(reference_mask.sum()),
                "bad_rate": float(reference_target.mean()),
            },
            {
                "period": "actual",
                "week_min": actual_weeks[0],
                "week_max": actual_weeks[-1],
                "week_count": len(actual_weeks),
                "row_count": int(actual_mask.sum()),
                "bad_rate": float(actual_target.mean()),
            },
        ]
    )
    status_counts = {
        str(key): int(value)
        for key, value in feature_diagnostics["status"].value_counts().sort_index().items()
    }
    summary: dict[str, object] = {
        "dataset_label": config.dataset_label,
        "synthetic_demo": config.synthetic_demo,
        "rows": int(len(data)),
        "features": int(len(features)),
        "weeks": int(data[config.week_col].nunique()),
        "bad_count": int(data[config.target_col].sum()),
        "bad_rate": float(data[config.target_col].mean()),
        "reference_week_min": float(reference_weeks[0]),
        "reference_week_max": float(reference_weeks[-1]),
        "actual_week_min": float(actual_weeks[0]),
        "actual_week_max": float(actual_weeks[-1]),
        "status_counts": status_counts,
        "oof_scores": list(model_diagnostics["score"]) if not model_diagnostics.empty else [],
    }
    run = DiagnosticsRun(
        feature_diagnostics=feature_diagnostics,
        bin_diagnostics=bin_diagnostics,
        correlation_recommendations=correlations,
        model_diagnostics=model_diagnostics,
        weekly_model_metrics=weekly_model_metrics,
        split_summary=split_summary,
        summary=summary,
    )
    write_csv(feature_diagnostics, output / "feature_diagnostics.csv")
    write_csv(bin_diagnostics, output / "bin_diagnostics.csv")
    write_csv(correlations, output / "correlation_recommendations.csv")
    write_csv(model_diagnostics, output / "model_diagnostics.csv")
    write_csv(weekly_model_metrics, output / "weekly_model_metrics.csv")
    write_csv(split_summary, output / "split_summary.csv")
    (output / "summary.json").write_text(
        json.dumps({"config": asdict(config), **summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(run, output, config)
    return run
