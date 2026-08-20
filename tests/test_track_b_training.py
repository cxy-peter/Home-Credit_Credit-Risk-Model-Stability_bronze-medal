from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from home_credit_diagnostics import training
from home_credit_diagnostics.rolling_origin import (
    RollingOriginConfig,
    build_rolling_origin_folds,
)
from home_credit_diagnostics.training import (
    TrackBTrainingConfig,
    _prepare_fold_frames,
    build_feature_manifest,
    run_track_b_training,
)


def _small_customer_frame() -> pd.DataFrame:
    rows: list[dict[str, int | float]] = []
    case_id = 0
    for week in range(92):
        for target in (0, 1):
            rows.append(
                {
                    "case_id": case_id,
                    "week_num": week,
                    "target": target,
                    "numeric_signal": float(target + week / 100),
                    "MODE(channel)": 0 if week < 40 else 1,
                    "MODE(high_card)": case_id,
                    "MONTH(date_decision)": week // 4,
                }
            )
            case_id += 1
    return pd.DataFrame(rows)


def test_default_track_b_timeline_has_label_delay_and_quarantine() -> None:
    folds, quarantine = build_rolling_origin_folds(range(92), RollingOriginConfig())

    assert len(folds) == 8
    first = folds[0]
    assert first.fit_weeks == tuple(range(28))
    assert first.early_stopping_weeks == tuple(range(28, 32))
    assert first.mature_train_weeks == tuple(range(32))
    assert first.shadow_weeks == tuple(range(32, 40))
    assert first.validation_weeks == tuple(range(40, 44))
    assert folds[-1].validation_weeks == tuple(range(68, 72))
    assert quarantine.mature_train_weeks == tuple(range(72))
    assert quarantine.shadow_weeks == tuple(range(72, 80))
    assert quarantine.quarantine_weeks == tuple(range(80, 92))

    with pytest.raises(ValueError, match="gap-free OOF coverage"):
        build_rolling_origin_folds(
            range(92), RollingOriginConfig(validation_weeks=4, step_weeks=8)
        )
    with pytest.raises(ValueError, match="must be consecutive"):
        build_rolling_origin_folds(
            [week for week in range(93) if week != 20], RollingOriginConfig()
        )


def test_fold_categories_fit_vocabulary_on_training_only() -> None:
    frame = _small_customer_frame()
    config = TrackBTrainingConfig(high_cardinality_threshold=10)
    manifest = build_feature_manifest(frame, config)
    folds, _ = build_rolling_origin_folds(frame["week_num"].unique(), config.rolling_config())

    X_fit, _, _, _, X_valid, _, unseen, preprocessor = _prepare_fold_frames(
        frame, folds[0], manifest, config
    )

    assert str(X_fit["MODE(channel)"].dtype) == "category"
    assert (
        manifest.set_index("feature").loc["MODE(high_card)", "encoding"]
        == "decided_from_each_fold_fit_cardinality"
    )
    assert X_valid["MODE(channel)"].isna().all()
    assert (X_valid["MODE(high_card)"] == 0.0).all()
    assert (
        preprocessor["categorical_columns"]["MODE(high_card)"]["encoding"]
        == "fold_train_frequency"
    )
    assert preprocessor["categorical_columns"]["MODE(high_card)"]["source_dtype"] == "int64"
    channel = unseen.query(
        "feature == 'MODE(channel)' and split == 'rolling_oof'"
    ).iloc[0]
    assert channel["unseen_rate"] == 1.0


def test_time_proxy_exclusion_is_explicit_not_implicit() -> None:
    frame = _small_customer_frame()
    included = build_feature_manifest(frame, TrackBTrainingConfig())
    stricter = build_feature_manifest(
        frame, TrackBTrainingConfig(exclude_time_proxies=True)
    )

    assert bool(
        included.set_index("feature").loc["MONTH(date_decision)", "use_in_model"]
    )
    strict_row = stricter.set_index("feature").loc["MONTH(date_decision)"]
    assert not bool(strict_row["use_in_model"])
    assert strict_row["role"] == "MONITOR_ONLY"


def test_training_outputs_only_future_week_oof(
    tmp_path: Path, monkeypatch
) -> None:
    frame = _small_customer_frame()

    def fake_trainer(
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        X_early: pd.DataFrame,
        y_early: pd.Series,
        X_valid: pd.DataFrame,
        config: TrackBTrainingConfig,
        fold_id: int,
        model_dir: Path,
    ) -> tuple[np.ndarray, int, list[dict[str, object]]]:
        del X_fit, y_fit, X_early, y_early, config, model_dir
        score = X_valid["numeric_signal"].to_numpy(dtype=float)
        importance = [
            {
                "model": "lightgbm",
                "fold_id": fold_id,
                "feature": feature,
                "importance_type": "gain",
                "importance_raw": 1.0,
                "importance_normalized_within_fold": 1.0 / len(X_valid.columns),
            }
            for feature in X_valid.columns
        ]
        return score, 7, importance

    monkeypatch.setattr(training, "_train_lightgbm", fake_trainer)
    config = TrackBTrainingConfig(models=("lightgbm",), max_folds=2, n_estimators=10)
    run = run_track_b_training(frame, tmp_path / "track_b", config=config)

    assert set(run.oof_predictions["week_num"]) == set(range(40, 48))
    assert run.oof_predictions["case_id"].is_unique
    assert run.oof_predictions["lightgbm_oof"].notna().all()
    assert set(run.oof_predictions["evaluation_asof_week"]) == {51, 55}
    assert not set(range(40)).intersection(run.oof_predictions["week_num"])
    assert (tmp_path / "track_b" / "oof_predictions.parquet").exists()
    assert (tmp_path / "track_b" / "preprocessing" / "fold_01.json").exists()
    assert (tmp_path / "track_b" / "RUN_REPORT.md").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        run_track_b_training(frame, tmp_path / "track_b", config=config)


def test_optional_real_model_adapters_smoke(tmp_path: Path) -> None:
    lgb = pytest.importorskip("lightgbm")
    xgb = pytest.importorskip("xgboost")
    dtype = pd.CategoricalDtype(categories=[0, 1])

    def matrix(rows: int, offset: int) -> tuple[pd.DataFrame, pd.Series]:
        target = pd.Series(np.arange(rows) % 2, dtype=np.int8)
        data = pd.DataFrame(
            {
                "numeric": target.to_numpy(dtype=np.float32)
                + np.arange(rows, dtype=np.float32) / (rows + offset + 1),
                "MODE(channel)": pd.Series(np.arange(rows) % 2).astype(dtype),
            }
        )
        return data, target

    X_fit, y_fit = matrix(400, 0)
    X_early, y_early = matrix(100, 1)
    X_valid, _ = matrix(80, 2)
    config = TrackBTrainingConfig(n_estimators=2, early_stopping_rounds=1, n_jobs=2)

    lgb_score, lgb_best, _ = training._train_lightgbm(
        X_fit,
        y_fit,
        X_early,
        y_early,
        X_valid,
        config,
        1,
        tmp_path / "lgb",
    )
    xgb_score, xgb_best, _ = training._train_xgboost(
        X_fit,
        y_fit,
        X_early,
        y_early,
        X_valid,
        config,
        1,
        tmp_path / "xgb",
    )

    assert len(lgb_score) == len(X_valid)
    assert len(xgb_score) == len(X_valid)
    assert 1 <= lgb_best <= 2
    assert 1 <= xgb_best <= 2
    assert (tmp_path / "lgb" / "fold_01.txt").exists()
    assert (tmp_path / "xgb" / "fold_01.json").exists()

    loaded_lgb = lgb.Booster(model_file=str(tmp_path / "lgb" / "fold_01.txt"))
    reloaded_lgb_score = loaded_lgb.predict(X_valid)
    loaded_xgb = xgb.Booster()
    loaded_xgb.load_model(tmp_path / "xgb" / "fold_01.json")
    reloaded_xgb_score = loaded_xgb.predict(
        xgb.DMatrix(X_valid, enable_categorical=True)
    )
    assert np.allclose(lgb_score, reloaded_lgb_score)
    assert np.allclose(xgb_score, reloaded_xgb_score)
