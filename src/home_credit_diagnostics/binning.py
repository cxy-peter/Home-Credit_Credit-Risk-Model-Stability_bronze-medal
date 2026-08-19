from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


MISSING_BUCKET = "__MISSING__"
OTHER_BUCKET = "__OTHER__"
UNSEEN_BUCKET = "__UNSEEN__"
DEFAULT_MISSING_SENTINELS = frozenset({"", "nan", "none", "null", "<na>"})


def _categorical_values(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = series.astype("string")
    normalized = values.str.strip().str.lower()
    missing = values.isna() | normalized.isin(DEFAULT_MISSING_SENTINELS)
    keys = "VALUE::" + values.fillna("")
    return keys, missing


def effective_missing_mask(series: pd.Series, feature_type: str | None = None) -> pd.Series:
    """Treat stored string sentinels from the original notebook as missing."""
    if feature_type == "numeric" or (
        feature_type is None and is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)
    ):
        return pd.to_numeric(series, errors="coerce").isna()
    _, missing = _categorical_values(series)
    return missing


@dataclass(frozen=True)
class FeatureBinner:
    feature: str
    feature_type: str
    bucket_order: tuple[str, ...]
    numeric_edges: tuple[float, ...] = ()
    kept_categories: tuple[str, ...] = ()
    known_categories: tuple[str, ...] = ()

    def transform(self, series: pd.Series) -> pd.Series:
        if self.feature_type == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            labels = [f"BIN_{index:02d}" for index in range(len(self.numeric_edges) - 1)]
            bucketed = pd.cut(
                numeric,
                bins=np.asarray(self.numeric_edges, dtype=float),
                labels=labels,
                include_lowest=True,
                right=True,
            ).astype("string")
            return bucketed.fillna(MISSING_BUCKET).astype(str)

        keys, missing = _categorical_values(series)
        known = set(self.known_categories)
        kept = set(self.kept_categories)
        output = pd.Series(UNSEEN_BUCKET, index=series.index, dtype="object")
        output.loc[missing] = MISSING_BUCKET
        present = ~missing
        output.loc[present & keys.isin(known) & ~keys.isin(kept)] = OTHER_BUCKET
        output.loc[present & keys.isin(kept)] = keys.loc[present & keys.isin(kept)]
        return output

    def bounds_for(self, bucket: str) -> tuple[float, float]:
        if self.feature_type != "numeric" or not bucket.startswith("BIN_"):
            return float("nan"), float("nan")
        index = int(bucket.split("_")[1])
        return self.numeric_edges[index], self.numeric_edges[index + 1]


@dataclass
class BinningResult:
    binner: FeatureBinner
    woe_map: dict[str, float]
    table: pd.DataFrame
    iv: float
    reference_buckets: pd.Series

    def risk_score(self, series: pd.Series) -> pd.Series:
        buckets = self.binner.transform(series)
        return self.risk_score_from_buckets(buckets)

    def risk_score_from_buckets(self, buckets: pd.Series) -> pd.Series:
        return buckets.map(self.woe_map).fillna(0.0).astype(float)


def _fit_numeric(feature: str, series: pd.Series, n_bins: int) -> FeatureBinner:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.dropna().to_numpy(dtype=float)
    if values.size == 0 or np.unique(values).size <= 1:
        edges = np.array([-np.inf, np.inf], dtype=float)
    else:
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        internal = np.unique(quantiles[1:-1])
        internal = internal[np.isfinite(internal)]
        edges = np.concatenate(([-np.inf], internal, [np.inf])).astype(float)
    bins = tuple(f"BIN_{index:02d}" for index in range(len(edges) - 1))
    return FeatureBinner(
        feature=feature,
        feature_type="numeric",
        bucket_order=(MISSING_BUCKET, *bins),
        numeric_edges=tuple(float(value) for value in edges),
    )


def _fit_categorical(feature: str, series: pd.Series, max_categories: int) -> FeatureBinner:
    keys, missing = _categorical_values(series)
    counts = keys.loc[~missing].value_counts(dropna=False)
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    kept = tuple(str(key) for key, _ in ranked[:max_categories])
    known = tuple(str(key) for key, _ in ranked)
    has_other = len(known) > len(kept)
    order: list[str] = [MISSING_BUCKET, *kept]
    if has_other:
        order.append(OTHER_BUCKET)
    order.append(UNSEEN_BUCKET)
    return FeatureBinner(
        feature=feature,
        feature_type="categorical",
        bucket_order=tuple(order),
        kept_categories=kept,
        known_categories=known,
    )


def fit_feature_binner(
    feature: str,
    reference: pd.Series,
    target: pd.Series,
    *,
    n_bins: int = 10,
    max_categories: int = 100,
    smoothing: float = 0.5,
    feature_type: str | None = None,
) -> BinningResult:
    """Fit bins and WOE only on the reference period."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if max_categories < 1:
        raise ValueError("max_categories must be positive")
    y = pd.to_numeric(target, errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("target must contain only 0 and 1")

    if feature_type not in {None, "numeric", "categorical"}:
        raise ValueError("feature_type must be numeric, categorical, or None")
    inferred_type = (
        "numeric"
        if is_numeric_dtype(reference.dtype) and not is_bool_dtype(reference.dtype)
        else "categorical"
    )
    resolved_type = feature_type or inferred_type
    if resolved_type == "numeric":
        binner = _fit_numeric(feature, reference, n_bins)
    else:
        binner = _fit_categorical(feature, reference, max_categories)

    buckets = binner.transform(reference)
    active_buckets = [bucket for bucket in binner.bucket_order if int((buckets == bucket).sum()) > 0]
    active_count = max(len(active_buckets), 1)
    total_bad = int(y.sum())
    total_good = int((1 - y).sum())
    bad_denominator = total_bad + smoothing * active_count
    good_denominator = total_good + smoothing * active_count

    rows: list[dict[str, object]] = []
    woe_map: dict[str, float] = {}
    iv = 0.0
    for order, bucket in enumerate(binner.bucket_order):
        mask = buckets.eq(bucket)
        count = int(mask.sum())
        bad = int(y.loc[mask].sum())
        good = count - bad
        if count == 0:
            bad_share = 0.0
            good_share = 0.0
            woe = 0.0
            component = 0.0
        else:
            bad_share = (bad + smoothing) / bad_denominator
            good_share = (good + smoothing) / good_denominator
            woe = float(np.log(bad_share / good_share))
            component = float((bad_share - good_share) * woe)
        woe_map[bucket] = woe
        iv += component
        lower, upper = binner.bounds_for(bucket)
        rows.append(
            {
                "feature": feature,
                "feature_type": binner.feature_type,
                "bucket_order": order,
                "bucket": bucket,
                "lower_bound": lower,
                "upper_bound": upper,
                "reference_count": count,
                "reference_bad": bad,
                "reference_good": good,
                "reference_bad_rate": bad / count if count else np.nan,
                "bad_share": bad_share,
                "good_share": good_share,
                "woe": woe,
                "iv_component": component,
            }
        )

    return BinningResult(
        binner=binner,
        woe_map=woe_map,
        table=pd.DataFrame(rows),
        iv=float(iv),
        reference_buckets=buckets,
    )


def add_actual_psi(
    result: BinningResult,
    actual: pd.Series,
    *,
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, float, pd.Series]:
    """Apply reference-fitted bins to actual data and calculate fixed-bin PSI."""
    reference_buckets = result.reference_buckets
    actual_buckets = result.binner.transform(actual)
    reference_total = max(len(reference_buckets), 1)
    actual_total = max(len(actual_buckets), 1)
    table = result.table.copy()
    actual_counts: list[int] = []
    actual_shares: list[float] = []
    reference_shares: list[float] = []
    components: list[float] = []
    for bucket in table["bucket"]:
        reference_count = int(reference_buckets.eq(bucket).sum())
        actual_count = int(actual_buckets.eq(bucket).sum())
        reference_share = reference_count / reference_total
        actual_share = actual_count / actual_total
        adjusted_reference = max(reference_share, epsilon)
        adjusted_actual = max(actual_share, epsilon)
        component = float(
            (adjusted_actual - adjusted_reference)
            * np.log(adjusted_actual / adjusted_reference)
        )
        actual_counts.append(actual_count)
        reference_shares.append(reference_share)
        actual_shares.append(actual_share)
        components.append(component)
    table["reference_share"] = reference_shares
    table["actual_count"] = actual_counts
    table["actual_share"] = actual_shares
    table["psi_component"] = components
    return table, float(np.sum(components)), actual_buckets


def decode_bucket(bucket: str) -> str:
    """Make categorical value buckets readable in CSV and Markdown output."""
    return bucket.removeprefix("VALUE::")


def ordered_bucket_union(*collections: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for collection in collections:
        for value in collection:
            if value not in seen:
                seen.add(value)
                output.append(value)
    return output
