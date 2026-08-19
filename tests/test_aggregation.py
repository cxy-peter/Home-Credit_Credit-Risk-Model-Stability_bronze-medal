import pandas as pd
import pytest

from home_credit_diagnostics.aggregation import aggregate_history


def test_temporal_aggregates_require_and_sort_explicit_time() -> None:
    history = pd.DataFrame(
        {
            "case_id": [1, 1, 1, 2, 2],
            "event_time": [3, 1, 2, 2, 1],
            "amount": [30.0, 10.0, 20.0, 8.0, 5.0],
            "status": ["c", "a", "b", "y", "x"],
            "product_code": [1, 1, 2, 3, 3],
        }
    )

    order_free = aggregate_history(history, id_col="case_id")
    temporal = aggregate_history(
        history,
        id_col="case_id",
        time_col="event_time",
        categorical_cols=["product_code"],
    )

    assert not any(column.startswith("first_") for column in order_free.columns)
    first = temporal.set_index("case_id")
    assert first.loc[1, "first_amount"] == 10.0
    assert first.loc[1, "last_amount"] == 30.0
    assert first.loc[1, "last_minus_first_amount"] == 20.0
    assert first.loc[1, "last_status"] == "c"
    assert first.loc[1, "mode_product_code"] == 1


def test_duplicate_times_require_explicit_unique_tie_breaker() -> None:
    history = pd.DataFrame(
        {
            "case_id": [1, 1],
            "event_time": [1, 1],
            "event_sequence": [2, 1],
            "amount": [20.0, 10.0],
        }
    )
    with pytest.raises(ValueError, match="require tie_breaker_col"):
        aggregate_history(history, id_col="case_id", time_col="event_time")

    result = aggregate_history(
        history,
        id_col="case_id",
        time_col="event_time",
        tie_breaker_col="event_sequence",
    ).set_index("case_id")
    assert result.loc[1, "first_amount"] == 10.0
    assert result.loc[1, "last_amount"] == 20.0
