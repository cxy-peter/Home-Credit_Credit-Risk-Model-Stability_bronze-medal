# Home Credit Track B rolling OOF run

Dataset: **Kaggle base_100features.parquet full 8-fold rolling OOF**

- Run ID: `track-b-lag8-full-8fold-v1`
- Models: lightgbm, xgboost
- Label maturity lag (simulation): 8 complete weeks
- Executed folds: 8
- Rolling OOF weeks: 40 to 71
- Rolling OOF rows: 559,660
- Outer quarantine weeks (not trained or evaluated here): 80 to 91
- Run completeness: FULL INNER ROLLING OOF
- Row sampling smoke mode: False

Validation uses complete future weeks. The label-maturity shadow gap is excluded from both fit and early stopping. Early stopping uses only the final mature weeks inside each training window.

Warm-up training rows are intentionally absent from OOF scores; no training predictions are used to fill them.

## Pooled rolling OOF metrics

| model | auc | ks | average_precision | lift_05 | lift_10 | lift_20 | stability |
|---|---|---|---|---|---|---|---|
| lightgbm | 0.801606 | 0.458083 | 0.148646 | 4.947032 | 4.016746 | 3.050146 | 0.577893 |
| xgboost | 0.799394 | 0.453157 | 0.146993 | 4.884119 | 4.001018 | 3.024009 | 0.575973 |
| blend | 0.801969 | 0.457923 | 0.149348 | 4.932229 | 4.017671 | 3.037424 | 0.579424 |

## Interpretation boundary

This is model validation conditional on the supplied prebuilt feature matrix. It does not rebuild the original 472-feature pipeline or claim that upstream feature selection was repeated inside every fold.

The runner is Track B only: it does not import, execute, or score Competition Forensics/Track A artifacts.
