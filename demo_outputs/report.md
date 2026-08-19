# Home Credit Diagnostics V3 — SYNTHETIC SMOKE TEST ONLY

> This report validates code paths with deterministic synthetic data. It is not the final Kaggle-data result and must not be quoted as model performance.

Dataset: **Deterministic synthetic smoke data — not final Kaggle results**

## Split

- Rows: 12,000
- Features diagnosed: 15
- Real week values: 60
- Target positives: 591
- Bad rate: 4.9250%
- Reference weeks: 0.0 to 41.0
- Actual weeks: 42.0 to 59.0
- Bins, category frequency, WOE and IV were fitted on reference weeks only.
- PSI and single-feature AUC/KS/Lift use the untouched later actual weeks.

## Feature status

- CONSTANT_REVIEW: 2
- EXTREME_SPARSE_REVIEW: 1
- HIGH_DRIFT_REVIEW: 2
- KEEP_CANDIDATE: 9
- SPARSE_RESCUE_CANDIDATE: 1

## Leading actual-period single-feature diagnostics

| feature | feature_type | iv | psi | auc | ks | lift_10 | status |
|---|---|---|---|---|---|---|---|
| debt_ratio | numeric | 0.1569 | 0.0013 | 0.5978 | 0.1971 | 1.7845 | KEEP_CANDIDATE |
| region_cat | categorical | 0.0371 | 0.0019 | 0.5612 | 0.0945 | 1.2332 | KEEP_CANDIDATE |
| applications_90d | numeric | 0.0693 | 0.0008 | 0.5558 | 0.0860 | 1.2582 | KEEP_CANDIDATE |
| occupation_cat | categorical | 0.2438 | 0.0413 | 0.5370 | 0.0931 | 1.1148 | KEEP_CANDIDATE |
| sparse_signal | numeric | 0.0494 | 0.0055 | 0.5199 | 0.0367 | 1.3248 | SPARSE_RESCUE_CANDIDATE |
| channel_cat | categorical | 0.0196 | 6.1035 | 0.5181 | 0.0724 | 1.2280 | HIGH_DRIFT_REVIEW |
| income_amt | numeric | 0.0281 | 0.0050 | 0.5175 | 0.0512 | 1.0903 | KEEP_CANDIDATE |
| drift_signal | numeric | 0.0179 | 1.6931 | 0.5103 | 0.0305 | 1.0411 | HIGH_DRIFT_REVIEW |
| income_duplicate_corr | numeric | 0.0188 | 0.0060 | 0.5096 | 0.0320 | 1.0758 | KEEP_CANDIDATE |
| age_yrs | numeric | 0.0432 | 0.0047 | 0.5077 | 0.0327 | 1.2640 | KEEP_CANDIDATE |

## Optional OOF diagnostics

| score | auc | ks | lift_10 | lift_20 | stability | weeks_scored |
|---|---|---|---|---|---|---|
| existing_simulated_oof | 0.5684 | 0.1106 | 1.3875 | 1.4213 | 0.0488 | 60 |
| challenger_simulated_oof | 0.5637 | 0.0938 | 1.5567 | 1.3706 | 0.0415 | 60 |

## Scope

This run consumes an existing customer-level parquet. It does not rebuild the original multi-table features or retrain LightGBM/CatBoost. Correlation output is a review recommendation, not an automatic feature deletion list.

AMEX-style first/last/delta aggregation is available separately, but only when an explicit time column is supplied; parquet row order is never treated as time.

## AMEX patterns: adopted and excluded

Adopted: numeric mean/std/min/max, categorical count/nunique/mode, and explicit time-sorted first/last/last-minus-first/last-minus-mean. Existing OOF is preserved only when supplied with exact case_id alignment.

Excluded: row-order-based temporal deltas, the AMEX competition metric, and the Transformer sequence model. Home Credit retains its real-week Gini stability metric and does not assume one homogeneous customer sequence.

The source parquet is not redistributed with these lightweight results. Dataset use remains subject to its source page and the underlying Home Credit competition rules.
