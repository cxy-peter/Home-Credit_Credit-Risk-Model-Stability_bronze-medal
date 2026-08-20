from __future__ import annotations

import gc
import importlib.metadata
import inspect
import json
import platform
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.metrics import average_precision_score

from .io import read_table, write_csv
from .metrics import binary_metrics, tie_aware_lift, weekly_stability
from .pipeline import DiagnosticsConfig, _is_time_proxy, normalize_customer_frame
from .rolling_origin import (
    QuarantinePlan,
    RollingFold,
    RollingOriginConfig,
    build_rolling_origin_folds,
)


SUPPORTED_MODELS = ("lightgbm", "xgboost")


@dataclass(frozen=True)
class TrackBTrainingConfig:
    id_col: str = "case_id"
    target_col: str = "target"
    week_col: str = "week_num"
    label_maturity_lag_weeks: int = 8
    validation_weeks: int = 4
    step_weeks: int = 4
    min_mature_train_weeks: int = 32
    inner_early_stopping_weeks: int = 4
    outer_quarantine_weeks: int = 12
    max_folds: int | None = None
    models: tuple[str, ...] = SUPPORTED_MODELS
    random_state: int = 42
    n_estimators: int = 2_000
    learning_rate: float = 0.03
    early_stopping_rounds: int = 150
    n_jobs: int = 12
    xgboost_device: str = "cpu"
    lightgbm_device: str = "cpu"
    high_cardinality_threshold: int = 10_000
    exclude_time_proxies: bool = False
    max_rows_per_week: int | None = None
    run_id: str | None = None
    dataset_label: str = "User-supplied customer-level feature matrix"
    input_path: str | None = None
    invocation_argv: tuple[str, ...] = ()

    def rolling_config(self) -> RollingOriginConfig:
        return RollingOriginConfig(
            label_maturity_lag_weeks=self.label_maturity_lag_weeks,
            validation_weeks=self.validation_weeks,
            step_weeks=self.step_weeks,
            min_mature_train_weeks=self.min_mature_train_weeks,
            inner_early_stopping_weeks=self.inner_early_stopping_weeks,
            outer_quarantine_weeks=self.outer_quarantine_weeks,
            max_folds=self.max_folds,
        )


@dataclass
class TrackBRun:
    oof_predictions: pd.DataFrame
    oof_labels: pd.DataFrame
    overall_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    weekly_metrics: pd.DataFrame
    fold_manifest: pd.DataFrame
    feature_manifest: pd.DataFrame
    feature_importance_by_fold: pd.DataFrame
    feature_importance_summary: pd.DataFrame


def _is_encoded_categorical(feature: str, series: pd.Series) -> bool:
    return feature.upper().startswith(("MODE(", "MONTH(", "SEASON(", "WEEKDAY(")) or is_bool_dtype(
        series.dtype
    )


def _validate_models(models: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(model.strip().lower() for model in models if model.strip()))
    if not normalized:
        raise ValueError("At least one model must be selected")
    unknown = sorted(set(normalized).difference(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported Track B models: {unknown}")
    return normalized


def _runtime_versions(models: Sequence[str]) -> dict[str, str | None]:
    packages = ["numpy", "pandas", "pyarrow", "scikit-learn"]
    if "lightgbm" in models:
        packages.append("lightgbm")
    if "xgboost" in models:
        packages.append("xgboost")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _preflight_training_frame(frame: pd.DataFrame, config: TrackBTrainingConfig) -> None:
    required = [config.id_col, config.target_col, config.week_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Track B input is missing required columns: {missing}")
    if frame[config.id_col].isna().any() or frame[config.id_col].duplicated().any():
        raise ValueError(f"{config.id_col} must be present and unique before training")
    target = pd.to_numeric(frame[config.target_col], errors="raise")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{config.target_col} must contain only 0 and 1")
    week = pd.to_numeric(frame[config.week_col], errors="raise")
    if week.isna().any():
        raise ValueError(f"{config.week_col} cannot contain missing values")


def load_track_b_frame(path: str | Path, config: TrackBTrainingConfig) -> pd.DataFrame:
    """Load, validate, and memory-downcast the customer-level matrix."""

    raw = read_table(path)
    normalized = normalize_customer_frame(
        raw,
        DiagnosticsConfig(
            id_col=config.id_col,
            target_col=config.target_col,
            week_col=config.week_col,
        ),
        copy=False,
    )
    del raw
    normalized[config.id_col] = pd.to_numeric(
        normalized[config.id_col], errors="raise", downcast="integer"
    )
    normalized[config.week_col] = pd.to_numeric(
        normalized[config.week_col], errors="raise", downcast="integer"
    )
    normalized[config.target_col] = normalized[config.target_col].astype(np.int8)
    metadata = {config.id_col, config.target_col, config.week_col}
    for feature in [column for column in normalized.columns if column not in metadata]:
        series = pd.to_numeric(normalized[feature], errors="coerce")
        if _is_encoded_categorical(feature, series) and not series.isna().any():
            normalized[feature] = pd.to_numeric(series, downcast="integer")
        else:
            numeric = series.astype(np.float32)
            normalized[feature] = numeric.mask(~np.isfinite(numeric), np.nan)
    normalized = normalized.sort_values(
        [config.week_col, config.id_col], kind="mergesort", ignore_index=True
    )
    return normalized


def _sample_rows_by_week(
    frame: pd.DataFrame,
    *,
    week_col: str,
    id_col: str,
    max_rows_per_week: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows_per_week is None:
        return frame
    if max_rows_per_week <= 0:
        raise ValueError("max_rows_per_week must be positive")
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby(week_col, sort=True):
        if len(group) > max_rows_per_week:
            group = group.sample(n=max_rows_per_week, random_state=random_state)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(
        [week_col, id_col], kind="mergesort", ignore_index=True
    )


def build_feature_manifest(
    frame: pd.DataFrame,
    config: TrackBTrainingConfig,
    *,
    feature_cols: Sequence[str] | None = None,
    exclude_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    metadata_roles = {
        config.id_col: "ALIGNMENT_ONLY",
        config.target_col: "LABEL_ONLY",
        config.week_col: "SPLIT_AND_MONITOR_ONLY",
    }
    excluded = set(exclude_cols or [])
    requested = set(feature_cols) if feature_cols is not None else None
    missing_requested = sorted((requested or set()).difference(frame.columns))
    if missing_requested:
        raise ValueError(f"Requested model features are missing: {missing_requested}")
    missing_excluded = sorted(excluded.difference(frame.columns))
    if missing_excluded:
        raise ValueError(f"Explicitly excluded features are missing: {missing_excluded}")

    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column in metadata_roles:
            role = metadata_roles[column]
            use_in_model = False
        elif requested is not None and column not in requested:
            role = "NOT_REQUESTED"
            use_in_model = False
        elif column in excluded:
            role = "USER_EXCLUDED"
            use_in_model = False
        elif config.exclude_time_proxies and _is_time_proxy(column):
            role = "MONITOR_ONLY"
            use_in_model = False
        else:
            role = "PREDICTOR_ALLOWED"
            use_in_model = True
        categorical = bool(column not in metadata_roles and _is_encoded_categorical(column, frame[column]))
        cardinality = int(frame[column].nunique(dropna=True))
        if not use_in_model:
            encoding = "not_modeled"
        elif categorical:
            encoding = "decided_from_each_fold_fit_cardinality"
        else:
            encoding = "native_numeric"
        rows.append(
            {
                "feature": column,
                "role": role,
                "use_in_model": use_in_model,
                "source_dtype": str(frame[column].dtype),
                "categorical": categorical,
                "cardinality_full_for_inventory_only": cardinality,
                "encoding": encoding,
            }
        )
    manifest = pd.DataFrame(rows)
    if not manifest["use_in_model"].any():
        raise ValueError("No model features remain after exclusions")
    return manifest


def _mask_for_weeks(frame: pd.DataFrame, week_col: str, weeks: Sequence[int | float]) -> pd.Series:
    return frame[week_col].isin(weeks)


def _prepare_fold_frames(
    frame: pd.DataFrame,
    fold: RollingFold,
    feature_manifest: pd.DataFrame,
    config: TrackBTrainingConfig,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    dict[str, object],
]:
    features = feature_manifest.loc[feature_manifest["use_in_model"], "feature"].tolist()
    categorical = feature_manifest.set_index("feature")["categorical"].to_dict()
    fit_mask = _mask_for_weeks(frame, config.week_col, fold.fit_weeks)
    early_mask = _mask_for_weeks(frame, config.week_col, fold.early_stopping_weeks)
    valid_mask = _mask_for_weeks(frame, config.week_col, fold.validation_weeks)
    X_fit = frame.loc[fit_mask, features].copy()
    y_fit = frame.loc[fit_mask, config.target_col].copy()
    X_early = frame.loc[early_mask, features].copy()
    y_early = frame.loc[early_mask, config.target_col].copy()
    X_valid = frame.loc[valid_mask, features].copy()
    y_valid = frame.loc[valid_mask, config.target_col].copy()
    unseen_rows: list[dict[str, object]] = []
    preprocessor: dict[str, object] = {
        "version": 1,
        "fit_week_min": fold.fit_weeks[0],
        "fit_week_max": fold.fit_weeks[-1],
        "feature_order": features,
        "categorical_columns": {},
    }

    for feature in features:
        if not categorical.get(feature, False):
            continue
        train_values = X_fit[feature]
        fold_encoding = (
            "fold_train_frequency"
            if train_values.nunique(dropna=True) > config.high_cardinality_threshold
            else "fold_train_native_category"
        )
        if fold_encoding == "fold_train_frequency":
            frequencies = train_values.value_counts(dropna=False, normalize=True)
            mapping_rows: list[dict[str, object]] = []
            missing_frequency = 0.0
            for value, frequency in frequencies.items():
                if pd.isna(value):
                    missing_frequency = float(frequency)
                else:
                    scalar = value.item() if isinstance(value, np.generic) else value
                    mapping_rows.append({"value": scalar, "frequency": float(frequency)})
            preprocessor["categorical_columns"][feature] = {
                "encoding": "fold_train_frequency",
                "source_dtype": str(train_values.dtype),
                "mapping": mapping_rows,
                "missing_frequency": missing_frequency,
                "unknown_frequency": 0.0,
            }
            X_fit[feature] = train_values.map(frequencies).fillna(0.0).astype(np.float32)
            for split_name, split in (("early_stopping", X_early), ("rolling_oof", X_valid)):
                original = split[feature]
                known = original.isin(frequencies.index)
                unseen_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "feature": feature,
                        "split": split_name,
                        "encoding": "fold_train_frequency",
                        "unseen_count": int((~known & original.notna()).sum()),
                        "row_count": int(len(original)),
                        "unseen_rate": float((~known & original.notna()).mean()),
                    }
                )
                split[feature] = original.map(frequencies).fillna(0.0).astype(np.float32)
        else:
            categories = pd.Index(train_values.dropna().unique()).sort_values()
            preprocessor["categorical_columns"][feature] = {
                "encoding": "fold_train_native_category",
                "source_dtype": str(train_values.dtype),
                "categories": [
                    value.item() if isinstance(value, np.generic) else value
                    for value in categories
                ],
                "unknown_value": None,
            }
            dtype = pd.CategoricalDtype(categories=categories, ordered=False)
            X_fit[feature] = train_values.astype(dtype)
            for split_name, split in (("early_stopping", X_early), ("rolling_oof", X_valid)):
                original = split[feature]
                known = original.isin(categories)
                unseen_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "feature": feature,
                        "split": split_name,
                        "encoding": "fold_train_native_category",
                        "unseen_count": int((~known & original.notna()).sum()),
                        "row_count": int(len(original)),
                        "unseen_rate": float((~known & original.notna()).mean()),
                    }
                )
                split[feature] = original.astype(dtype)
    return (
        X_fit,
        y_fit,
        X_early,
        y_early,
        X_valid,
        y_valid,
        pd.DataFrame(unseen_rows),
        preprocessor,
    )


def _assert_binary_split(y: pd.Series, split_name: str, fold_id: int) -> None:
    if y.nunique() != 2:
        raise ValueError(f"Fold {fold_id} {split_name} must contain both target classes")


def _importance_rows(
    model_name: str,
    fold_id: int,
    features: Sequence[str],
    values: np.ndarray,
    importance_type: str,
) -> list[dict[str, object]]:
    raw = np.asarray(values, dtype=float)
    denominator = float(raw.sum())
    normalized = raw / denominator if denominator > 0 else np.zeros_like(raw)
    return [
        {
            "model": model_name,
            "fold_id": fold_id,
            "feature": feature,
            "importance_type": importance_type,
            "importance_raw": float(value),
            "importance_normalized_within_fold": float(norm),
        }
        for feature, value, norm in zip(features, raw, normalized, strict=True)
    ]


def _train_lightgbm(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_early: pd.DataFrame,
    y_early: pd.Series,
    X_valid: pd.DataFrame,
    config: TrackBTrainingConfig,
    fold_id: int,
    model_dir: Path,
) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "LightGBM is not installed. Install the training extra with "
            "`python -m pip install -e \".[training]\"`."
        ) from exc

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        num_leaves=31,
        min_child_samples=1_500,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        max_bin=255,
        cat_smooth=20.0,
        cat_l2=10.0,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        device_type=config.lightgbm_device,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    fit_kwargs: dict[str, object] = {
        "eval_metric": "auc",
        "callbacks": [
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
        "categorical_feature": "auto",
    }
    if "eval_X" in inspect.signature(model.fit).parameters:
        fit_kwargs.update({"eval_X": X_early, "eval_y": y_early})
    else:  # LightGBM <= 4.6 compatibility
        fit_kwargs["eval_set"] = [(X_early, y_early)]
    model.fit(X_fit, y_fit, **fit_kwargs)
    score = model.predict_proba(X_valid)[:, 1]
    best_iteration = int(model.best_iteration_ or config.n_estimators)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(
        model_dir / f"fold_{fold_id:02d}.txt", num_iteration=best_iteration
    )
    importance: list[dict[str, object]] = []
    for importance_type in ("gain", "split"):
        importance.extend(
            _importance_rows(
                "lightgbm",
                fold_id,
                X_fit.columns,
                model.booster_.feature_importance(
                    importance_type=importance_type, iteration=best_iteration
                ),
                importance_type,
            )
        )
    del model
    return np.asarray(score, dtype=np.float64), best_iteration, importance


def _train_xgboost(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_early: pd.DataFrame,
    y_early: pd.Series,
    X_valid: pd.DataFrame,
    config: TrackBTrainingConfig,
    fold_id: int,
    model_dir: Path,
) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "XGBoost is not installed. Install the training extra with "
            "`python -m pip install -e \".[training]\"`."
        ) from exc

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=config.xgboost_device,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        max_depth=6,
        min_child_weight=50.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        gamma=0.05,
        max_bin=256,
        max_cat_to_onehot=4,
        max_cat_threshold=32,
        enable_categorical=True,
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_early, y_early)], verbose=False)
    score = model.predict_proba(X_valid)[:, 1]
    try:
        best_iteration = int(model.best_iteration) + 1
    except (AttributeError, ValueError):
        best_iteration = config.n_estimators
    booster = model.get_booster()
    best_booster = (
        booster[:best_iteration]
        if best_iteration < int(booster.num_boosted_rounds())
        else booster
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    best_booster.save_model(model_dir / f"fold_{fold_id:02d}.json")
    importance: list[dict[str, object]] = []
    for importance_type in ("gain", "weight"):
        lookup = best_booster.get_score(importance_type=importance_type)
        values = np.asarray([lookup.get(feature, 0.0) for feature in X_fit.columns])
        importance.extend(
            _importance_rows(
                "xgboost", fold_id, X_fit.columns, values, importance_type
            )
        )
    del best_booster, booster, model
    return np.asarray(score, dtype=np.float64), best_iteration, importance


def _score_summary(
    target: pd.Series,
    score: pd.Series,
    week: pd.Series,
) -> dict[str, float | int]:
    metrics = binary_metrics(target, score)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=int)
    prediction = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    metrics["average_precision"] = float(average_precision_score(y, prediction))
    metrics["lift_05"] = tie_aware_lift(y, prediction, 0.05)
    stability, _ = weekly_stability(target, score, week)
    return {**metrics, **stability}


def _evaluate_oof(
    evaluation: pd.DataFrame,
    score_cols: Sequence[str],
    config: TrackBTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    weekly_rows: list[dict[str, object]] = []
    for score_col in score_cols:
        model_name = score_col.removesuffix("_oof")
        overall_rows.append(
            {
                "model": model_name,
                "evaluation_scope": "pooled_rolling_oof",
                **_score_summary(
                    evaluation[config.target_col],
                    evaluation[score_col],
                    evaluation[config.week_col],
                ),
            }
        )
        for fold_id, fold_frame in evaluation.groupby("fold_id", sort=True):
            fold_rows.append(
                {
                    "model": model_name,
                    "fold_id": int(fold_id),
                    **_score_summary(
                        fold_frame[config.target_col],
                        fold_frame[score_col],
                        fold_frame[config.week_col],
                    ),
                }
            )
        for week_value, week_frame in evaluation.groupby(config.week_col, sort=True):
            values = binary_metrics(week_frame[config.target_col], week_frame[score_col])
            values["lift_05"] = tie_aware_lift(
                week_frame[config.target_col].to_numpy(dtype=int),
                week_frame[score_col].to_numpy(dtype=float),
                0.05,
            )
            values["average_precision"] = (
                float(average_precision_score(week_frame[config.target_col], week_frame[score_col]))
                if week_frame[config.target_col].nunique() == 2
                else float("nan")
            )
            weekly_rows.append(
                {
                    "model": model_name,
                    config.week_col: week_value,
                    **values,
                    "gini": 2.0 * float(values["auc"]) - 1.0
                    if not pd.isna(values["auc"])
                    else float("nan"),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(fold_rows), pd.DataFrame(weekly_rows)


def _week_ranges(weeks: Sequence[int | float]) -> tuple[object, object, int]:
    return weeks[0], weeks[-1], len(weeks)


def _build_fold_manifest_row(
    frame: pd.DataFrame,
    fold: RollingFold,
    config: TrackBTrainingConfig,
) -> dict[str, object]:
    fit_mask = _mask_for_weeks(frame, config.week_col, fold.fit_weeks)
    early_mask = _mask_for_weeks(frame, config.week_col, fold.early_stopping_weeks)
    mature_mask = _mask_for_weeks(frame, config.week_col, fold.mature_train_weeks)
    shadow_mask = _mask_for_weeks(frame, config.week_col, fold.shadow_weeks)
    validation_mask = _mask_for_weeks(frame, config.week_col, fold.validation_weeks)
    fit_min, fit_max, fit_count = _week_ranges(fold.fit_weeks)
    early_min, early_max, early_count = _week_ranges(fold.early_stopping_weeks)
    shadow_min, shadow_max, shadow_count = _week_ranges(fold.shadow_weeks)
    valid_min, valid_max, valid_count = _week_ranges(fold.validation_weeks)
    return {
        "fold_id": fold.fold_id,
        "replay_origin_week": fold.replay_origin_week,
        "fit_week_min": fit_min,
        "fit_week_max": fit_max,
        "fit_week_count": fit_count,
        "fit_rows": int(fit_mask.sum()),
        "fit_positives": int(frame.loc[fit_mask, config.target_col].sum()),
        "early_stopping_week_min": early_min,
        "early_stopping_week_max": early_max,
        "early_stopping_week_count": early_count,
        "early_stopping_rows": int(early_mask.sum()),
        "mature_train_week_max": fold.train_week_max,
        "mature_train_rows": int(mature_mask.sum()),
        "shadow_week_min": shadow_min,
        "shadow_week_max": shadow_max,
        "shadow_week_count": shadow_count,
        "shadow_rows": int(shadow_mask.sum()),
        "validation_week_min": valid_min,
        "validation_week_max": valid_max,
        "validation_week_count": valid_count,
        "validation_rows": int(validation_mask.sum()),
        "validation_positives": int(frame.loc[validation_mask, config.target_col].sum()),
    }


def _label_available_map(
    weeks: Sequence[int | float], lag: int
) -> dict[int | float, int | float]:
    ordered = list(np.sort(np.unique(np.asarray(weeks))))
    if lag == 0:
        return {week: week for week in ordered}
    return {week: ordered[index + lag] for index, week in enumerate(ordered[:-lag])}


def _summarize_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "importance_type",
                "feature",
                "fold_count",
                "mean_normalized_importance",
                "std_normalized_importance",
                "mean_rank",
            ]
        )
    ranked = importance.copy()
    ranked["rank_within_fold"] = ranked.groupby(
        ["model", "fold_id", "importance_type"]
    )["importance_normalized_within_fold"].rank(method="average", ascending=False)
    return (
        ranked.groupby(["model", "importance_type", "feature"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            mean_normalized_importance=("importance_normalized_within_fold", "mean"),
            std_normalized_importance=("importance_normalized_within_fold", "std"),
            mean_rank=("rank_within_fold", "mean"),
        )
        .sort_values(
            ["model", "importance_type", "mean_normalized_importance"],
            ascending=[True, True, False],
        )
    )


def _write_run_report(
    run: TrackBRun,
    output_dir: Path,
    config: TrackBTrainingConfig,
    quarantine: QuarantinePlan,
) -> None:
    folds = run.fold_manifest
    lines = [
        "# Home Credit Track B rolling OOF run",
        "",
        f"Dataset: **{config.dataset_label}**",
        "",
        f"- Run ID: `{config.run_id}`",
        f"- Models: {', '.join(config.models)}",
        f"- Label maturity lag (simulation): {config.label_maturity_lag_weeks} complete weeks",
        f"- Executed folds: {len(folds)}",
        f"- Rolling OOF weeks: {int(folds['validation_week_min'].min())} to {int(folds['validation_week_max'].max())}",
        f"- Rolling OOF rows: {len(run.oof_predictions):,}",
        f"- Outer quarantine weeks (not trained or evaluated here): {quarantine.quarantine_weeks[0]} to {quarantine.quarantine_weeks[-1]}",
        f"- Run completeness: {'PARTIAL DEVELOPMENT/SMOKE' if config.max_folds is not None or config.max_rows_per_week is not None else 'FULL INNER ROLLING OOF'}",
        f"- Row sampling smoke mode: {config.max_rows_per_week is not None}",
        "",
        "Validation uses complete future weeks. The label-maturity shadow gap is excluded from both fit and early stopping. "
        "Early stopping uses only the final mature weeks inside each training window.",
        "",
        "Warm-up training rows are intentionally absent from OOF scores; no training predictions are used to fill them.",
        "",
        "## Pooled rolling OOF metrics",
        "",
    ]
    if run.overall_metrics.empty:
        lines.append("No metrics were produced.")
    else:
        columns = [
            "model",
            "auc",
            "ks",
            "average_precision",
            "lift_05",
            "lift_10",
            "lift_20",
            "stability",
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in run.overall_metrics[columns].to_dict(orient="records"):
            lines.append(
                "| "
                + " | ".join(
                    str(value) if not isinstance(value, float) else f"{value:.6f}"
                    for value in row.values()
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is model validation conditional on the supplied prebuilt feature matrix. "
            "It does not rebuild the original 472-feature pipeline or claim that upstream feature selection was repeated inside every fold.",
            "",
            "The runner is Track B only: it does not import, execute, or score Competition Forensics/Track A artifacts.",
        ]
    )
    (output_dir / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_track_b_training_into(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: TrackBTrainingConfig | None = None,
    feature_cols: Sequence[str] | None = None,
    exclude_cols: Sequence[str] | None = None,
) -> TrackBRun:
    config = config or TrackBTrainingConfig()
    if config.run_id is None:
        config = replace(
            config, run_id=f"track-b-lag{config.label_maturity_lag_weeks}-v1"
        )
    models = _validate_models(config.models)
    _preflight_training_frame(frame, config)
    if config.n_estimators <= 0 or config.early_stopping_rounds <= 0:
        raise ValueError("n_estimators and early_stopping_rounds must be positive")
    if config.learning_rate <= 0 or config.n_jobs <= 0:
        raise ValueError("learning_rate and n_jobs must be positive")
    if config.high_cardinality_threshold <= 0:
        raise ValueError("high_cardinality_threshold must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_root = output / "models"
    preprocessing_root = output / "preprocessing"

    data = _sample_rows_by_week(
        frame,
        week_col=config.week_col,
        id_col=config.id_col,
        max_rows_per_week=config.max_rows_per_week,
        random_state=config.random_state,
    )
    feature_manifest = build_feature_manifest(
        data,
        config,
        feature_cols=feature_cols,
        exclude_cols=exclude_cols,
    )
    folds, quarantine = build_rolling_origin_folds(
        data[config.week_col].unique(), config.rolling_config()
    )
    label_available = _label_available_map(
        data[config.week_col].unique(), config.label_maturity_lag_weeks
    )
    prediction_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    unseen_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    iteration_rows: list[dict[str, object]] = []

    for fold in folds:
        print(
            f"Track B fold {fold.fold_id}/{len(folds)}: fit {fold.fit_weeks[0]}-{fold.fit_weeks[-1]}, "
            f"early-stop {fold.early_stopping_weeks[0]}-{fold.early_stopping_weeks[-1]}, "
            f"OOF {fold.validation_weeks[0]}-{fold.validation_weeks[-1]}",
            flush=True,
        )
        manifest_rows.append(_build_fold_manifest_row(data, fold, config))
        (
            X_fit,
            y_fit,
            X_early,
            y_early,
            X_valid,
            y_valid,
            unseen,
            preprocessor,
        ) = _prepare_fold_frames(data, fold, feature_manifest, config)
        _assert_binary_split(y_fit, "fit", fold.fold_id)
        _assert_binary_split(y_early, "early_stopping", fold.fold_id)
        unseen_frames.append(unseen)
        preprocessing_root.mkdir(parents=True, exist_ok=True)
        (preprocessing_root / f"fold_{fold.fold_id:02d}.json").write_text(
            json.dumps(preprocessor, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        valid_mask = _mask_for_weeks(data, config.week_col, fold.validation_weeks)
        prediction = data.loc[valid_mask, [config.id_col, config.week_col]].copy()
        prediction["fold_id"] = fold.fold_id
        prediction["model_fit_asof_week"] = fold.replay_origin_week
        prediction["fit_week_max"] = fold.fit_weeks[-1]
        prediction["mature_train_week_max"] = fold.train_week_max
        prediction["label_available_week"] = prediction[config.week_col].map(label_available)
        evaluation_asof_week = label_available[fold.validation_weeks[-1]]
        prediction["evaluation_asof_week"] = evaluation_asof_week

        if "lightgbm" in models:
            score, best_iteration, importance = _train_lightgbm(
                X_fit,
                y_fit,
                X_early,
                y_early,
                X_valid,
                config,
                fold.fold_id,
                model_root / "lightgbm",
            )
            prediction["lightgbm_oof"] = score
            iteration_rows.append(
                {"model": "lightgbm", "fold_id": fold.fold_id, "best_iteration": best_iteration}
            )
            print(f"  LightGBM best_iteration={best_iteration}", flush=True)
            importance_rows.extend(importance)
            gc.collect()
        if "xgboost" in models:
            score, best_iteration, importance = _train_xgboost(
                X_fit,
                y_fit,
                X_early,
                y_early,
                X_valid,
                config,
                fold.fold_id,
                model_root / "xgboost",
            )
            prediction["xgboost_oof"] = score
            iteration_rows.append(
                {"model": "xgboost", "fold_id": fold.fold_id, "best_iteration": best_iteration}
            )
            print(f"  XGBoost best_iteration={best_iteration}", flush=True)
            importance_rows.extend(importance)
            gc.collect()
        _assert_binary_split(y_valid, "rolling_oof", fold.fold_id)
        prediction[config.target_col] = y_valid.to_numpy(dtype=np.int8)
        prediction_frames.append(prediction)
        del X_fit, y_fit, X_early, y_early, X_valid, y_valid, prediction, preprocessor
        gc.collect()

    evaluation = pd.concat(prediction_frames, ignore_index=True).sort_values(
        [config.week_col, config.id_col], kind="mergesort", ignore_index=True
    )
    if evaluation[config.id_col].duplicated().any():
        raise AssertionError("A customer received more than one rolling OOF prediction")
    score_cols = [f"{model}_oof" for model in models]
    if evaluation[score_cols].isna().any().any():
        raise AssertionError("Rolling OOF scores contain missing values")
    if len(score_cols) == 2:
        evaluation["blend_oof"] = evaluation[score_cols].mean(axis=1)
        score_cols.append("blend_oof")
    if not (evaluation["model_fit_asof_week"] < evaluation[config.week_col]).all():
        raise AssertionError("Predictions must be created before their validation week")
    if not (evaluation["label_available_week"] <= evaluation["evaluation_asof_week"]).all():
        raise AssertionError("The configured OOF evaluation time precedes label maturity")

    prediction_columns = [
        config.id_col,
        config.week_col,
        "fold_id",
        "model_fit_asof_week",
        "fit_week_max",
        "mature_train_week_max",
        "label_available_week",
        "evaluation_asof_week",
        *score_cols,
    ]
    label_columns = [
        config.id_col,
        config.week_col,
        config.target_col,
        "label_available_week",
        "evaluation_asof_week",
    ]
    oof_predictions = evaluation[prediction_columns].copy()
    oof_labels = evaluation[label_columns].copy()
    overall, fold_metrics, weekly = _evaluate_oof(evaluation, score_cols, config)
    fold_manifest = pd.DataFrame(manifest_rows)
    iterations = pd.DataFrame(iteration_rows)
    fold_manifest = fold_manifest.merge(
        iterations.pivot(index="fold_id", columns="model", values="best_iteration")
        .add_suffix("_best_iteration")
        .reset_index(),
        on="fold_id",
        how="left",
        validate="one_to_one",
    )
    importance = pd.DataFrame(importance_rows)
    importance_summary = _summarize_importance(importance)
    unseen_columns = [
        "fold_id",
        "feature",
        "split",
        "encoding",
        "unseen_count",
        "row_count",
        "unseen_rate",
    ]
    categorical_unseen = (
        pd.concat(unseen_frames, ignore_index=True).reindex(columns=unseen_columns)
        if unseen_frames
        else pd.DataFrame(columns=unseen_columns)
    )
    run = TrackBRun(
        oof_predictions=oof_predictions,
        oof_labels=oof_labels,
        overall_metrics=overall,
        fold_metrics=fold_metrics,
        weekly_metrics=weekly,
        fold_manifest=fold_manifest,
        feature_manifest=feature_manifest,
        feature_importance_by_fold=importance,
        feature_importance_summary=importance_summary,
    )

    oof_predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    oof_labels.to_parquet(output / "oof_labels.parquet", index=False)
    evaluation.to_parquet(output / "oof_evaluation.parquet", index=False)
    write_csv(overall, output / "overall_metrics.csv")
    write_csv(fold_metrics, output / "fold_metrics.csv")
    write_csv(weekly, output / "weekly_metrics.csv")
    write_csv(fold_manifest, output / "fold_manifest.csv")
    write_csv(feature_manifest, output / "feature_manifest.csv")
    write_csv(categorical_unseen, output / "categorical_unseen_by_fold.csv")
    write_csv(importance, output / "feature_importance_by_fold.csv")
    write_csv(importance_summary, output / "feature_importance_summary.csv")
    run_config = {
        **asdict(config),
        "models": list(models),
        "feature_cols_requested": list(feature_cols) if feature_cols is not None else None,
        "exclude_cols_requested": list(exclude_cols) if exclude_cols is not None else [],
        "runtime_versions": _runtime_versions(models),
        "rows_loaded": int(len(frame)),
        "rows_modeled": int(len(data)),
        "features_modeled": int(feature_manifest["use_in_model"].sum()),
        "rolling_oof_rows": int(len(oof_predictions)),
        "rolling_oof_week_min": int(oof_predictions[config.week_col].min()),
        "rolling_oof_week_max": int(oof_predictions[config.week_col].max()),
        "quarantine_week_min": int(quarantine.quarantine_weeks[0]),
        "quarantine_week_max": int(quarantine.quarantine_weeks[-1]),
        "quarantine_evaluated": False,
        "partial_development_run": bool(
            config.max_folds is not None or config.max_rows_per_week is not None
        ),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_run_report(run, output, config, quarantine)
    return run


def run_track_b_training(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: TrackBTrainingConfig | None = None,
    feature_cols: Sequence[str] | None = None,
    exclude_cols: Sequence[str] | None = None,
) -> TrackBRun:
    """Run Track B into a fresh output directory via an atomic staging path."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Track B output already exists: {destination}. Use a new --out path so stale models cannot mix with this run."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.inprogress-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        run = _run_track_b_training_into(
            frame,
            staging,
            config=config,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
        )
        staging.replace(destination)
    return run
