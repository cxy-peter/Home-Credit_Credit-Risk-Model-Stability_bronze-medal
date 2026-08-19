import pandas as pd

from home_credit_diagnostics.binning import (
    MISSING_BUCKET,
    UNSEEN_BUCKET,
    add_actual_psi,
    fit_feature_binner,
)


def test_reference_categories_keep_missing_and_unseen_explicit() -> None:
    reference = pd.Series(["a", "a", "b", None, "nan", "b", "a", "b"])
    target = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
    actual = pd.Series(["a", "new", None, "nan", "b"])

    fitted = fit_feature_binner("category", reference, target, max_categories=10)
    transformed = fitted.binner.transform(actual)
    table, psi, _ = add_actual_psi(fitted, actual)

    assert transformed.iloc[1] == UNSEEN_BUCKET
    assert transformed.iloc[2] == MISSING_BUCKET
    assert transformed.iloc[3] == MISSING_BUCKET
    assert {MISSING_BUCKET, UNSEEN_BUCKET}.issubset(set(table["bucket"]))
    assert table.loc[table["bucket"] == UNSEEN_BUCKET, "actual_count"].iloc[0] == 1
    assert psi > 0
