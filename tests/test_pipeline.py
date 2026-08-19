import pandas as pd
import pytest

from home_credit_diagnostics.pipeline import DiagnosticsConfig, _is_time_proxy, run_diagnostics
from home_credit_diagnostics.synthetic import make_synthetic_customer_data


def test_pipeline_uses_real_week_order_and_strict_oof_alignment(tmp_path) -> None:
    frame, oof = make_synthetic_customer_data(n_rows=2_400, n_weeks=24, seed=7)
    parquet_style = frame.rename(columns={"week_num": "WEEK_NUM"}).set_index("case_id")
    config = DiagnosticsConfig(
        correlation_sample_rows=1_000,
        dataset_label="test data",
        synthetic_demo=True,
    )

    run = run_diagnostics(parquet_style, tmp_path / "run", config=config, oof=oof)

    split = run.split_summary.set_index("period")
    assert split.loc["reference", "week_max"] < split.loc["actual", "week_min"]
    assert run.summary["weeks"] == 24
    assert run.summary["bad_count"] == int(frame["target"].sum())
    sparse = run.feature_diagnostics.set_index("feature")
    assert bool(sparse.loc["sparse_signal", "sparse_rescue_candidate"])
    assert bool(sparse.loc["extreme_sparse", "extreme_sparse"])
    unseen = run.bin_diagnostics.query(
        "feature == 'channel_cat' and bucket == '__UNSEEN__'"
    )
    assert unseen["actual_count"].iloc[0] > 0
    assert set(run.model_diagnostics["score"]) == {
        "existing_simulated_oof",
        "challenger_simulated_oof",
    }
    assert run.model_diagnostics["provenance"].str.contains("synthetic").all()
    assert run.weekly_model_metrics.groupby("score")["week_num"].apply(
        lambda values: values.is_monotonic_increasing
    ).all()
    assert (tmp_path / "run" / "report.md").exists()

    with pytest.raises(ValueError, match="alignment must be exact"):
        run_diagnostics(
            parquet_style,
            tmp_path / "bad_oof",
            config=config,
            oof=oof.iloc[:-1],
        )


def test_time_proxy_tokens_are_reviewed_not_recommended(tmp_path) -> None:
    assert _is_time_proxy("MONTH(date_decision)")
    assert _is_time_proxy("SEASON(date_decision)")
    assert _is_time_proxy("WEEKDAY(date_decision)")
    assert _is_time_proxy("MAX_MIN_DELTA(credit_bureau_a_1.dpdmaxdatemonth_442T)")
    assert _is_time_proxy("MAX_MIN_DELTA(credit_bureau_a_1.dpdmaxdateyear_596T)")
    assert not _is_time_proxy("MAX(static_0.maininc_215A)")

    frame, _ = make_synthetic_customer_data(n_rows=1_200, n_weeks=12, seed=11)
    frame["MONTH(date_decision)"] = frame["week_num"] // 4
    run = run_diagnostics(
        frame,
        tmp_path / "time_proxy",
        config=DiagnosticsConfig(correlation_sample_rows=500),
    )
    row = run.feature_diagnostics.set_index("feature").loc["MONTH(date_decision)"]
    assert bool(row["is_time_proxy"])
    assert row["status"] == "TIME_PROXY_REVIEW"
    assert "MONTH(date_decision)" not in set(
        run.correlation_recommendations.get("recommend_keep", pd.Series(dtype=str))
    )
