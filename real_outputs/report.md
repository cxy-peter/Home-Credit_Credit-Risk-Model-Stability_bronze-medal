# Home Credit Diagnostics V3

> Results describe the supplied customer-level dataset; the original ML training was not rerun.

Dataset: **Kaggle diarray base_100features.parquet**
Source: https://www.kaggle.com/datasets/diarray/deep-feature-synthesis-home-credit-stability


## Split

- Rows: 1,526,659
- Features diagnosed: 100
- Real week values: 92
- Target positives: 47,994
- Bad rate: 3.1437%
- Reference weeks: 0.0 to 63.0
- Actual weeks: 64.0 to 91.0
- Bins, category frequency, WOE and IV were fitted on reference weeks only.
- PSI and single-feature AUC/KS/Lift use the untouched later actual weeks.

## Feature status

- CONSTANT_REVIEW: 2
- HIGH_DRIFT_REVIEW: 26
- KEEP_CANDIDATE: 67
- TIME_PROXY_REVIEW: 5

## Leading actual-period single-feature diagnostics

| feature | feature_type | iv | psi | auc | ks | lift_10 | status |
|---|---|---|---|---|---|---|---|
| MEAN(credit_bureau_a_2.pmts_overdue_1152A) | numeric | 0.2630 | 0.5326 | 0.6952 | 0.3238 | 2.3754 | HIGH_DRIFT_REVIEW |
| MAX(credit_bureau_a_1.dpdmax_757P) | numeric | 0.2291 | 0.5344 | 0.6762 | 0.3090 | 2.0451 | HIGH_DRIFT_REVIEW |
| MAX(credit_bureau_a_1.overdueamountmax2_398A) | numeric | 0.1656 | 0.5345 | 0.6516 | 0.2298 | 1.9649 | HIGH_DRIFT_REVIEW |
| MODE(applprev_1.status_219L) | categorical | 0.2026 | 0.0424 | 0.6468 | 0.2639 | 1.7635 | KEEP_CANDIDATE |
| MAX(applprev_1.maxdpdtolerance_577P) | numeric | 0.2283 | 0.0495 | 0.6427 | 0.2150 | 2.4566 | KEEP_CANDIDATE |
| SUM(static_0.avgdbddpdlast24m_3658932P) | numeric | 0.2155 | 0.0264 | 0.6405 | 0.1986 | 2.7251 | KEEP_CANDIDATE |
| MAX_MIN_DELTA(credit_bureau_a_1.dpdmax_757P) | numeric | 0.1447 | 0.2319 | 0.6405 | 0.2203 | 1.9545 | KEEP_CANDIDATE |
| SUM(credit_bureau_a_1.overdueamountmax2_398A) | numeric | 0.1216 | 0.2441 | 0.6334 | 0.2221 | 1.9698 | KEEP_CANDIDATE |
| MAX(credit_bureau_a_1.dpdmax_139P) | numeric | 0.2501 | 0.0043 | 0.6333 | 0.2214 | 2.4844 | KEEP_CANDIDATE |
| STD(applprev_1.maxdpdtolerance_577P) | numeric | 0.1741 | 0.0563 | 0.6201 | 0.1958 | 1.9538 | KEEP_CANDIDATE |

## Optional OOF diagnostics

No model metrics were calculated. Supply a genuine, case_id-aligned OOF file with `--oof`; training-set predictions are not accepted as a substitute.

## Scope

This run consumes an existing customer-level parquet. It does not rebuild the original multi-table features or retrain LightGBM/CatBoost. Correlation output is a review recommendation, not an automatic feature deletion list.

AMEX-style first/last/delta aggregation is available separately, but only when an explicit time column is supplied; parquet row order is never treated as time.

## AMEX patterns: adopted and excluded

Adopted: numeric mean/std/min/max, categorical count/nunique/mode, and explicit time-sorted first/last/last-minus-first/last-minus-mean. Existing OOF is preserved only when supplied with exact case_id alignment.

Excluded: row-order-based temporal deltas, the AMEX competition metric, and the Transformer sequence model. Home Credit retains its real-week Gini stability metric and does not assume one homogeneous customer sequence.

The source parquet is not redistributed with these lightweight results. Dataset use remains subject to its source page and the underlying Home Credit competition rules.
