# Track B full rolling OOF — lightweight checked snapshot

This directory is the Git-friendly result snapshot from the checked full run:

```text
run_id: track-b-lag8-full-8fold-v1
input: data/kaggle/base_100features.parquet
rows loaded: 1,526,659
features modeled: 100
rolling folds: 8
rolling OOF rows: 559,660
rolling OOF weeks: 40–71
outer quarantine: 80–91, not evaluated
models: LightGBM 4.7.0 / XGBoost 3.4.1, CPU
tree cap: 800
inner early stopping: 100 rounds
```

## Pooled rolling OOF

| Model | AUC | KS | AP | Lift@5% | Lift@10% | Lift@20% | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.801606 | 0.458083 | 0.148646 | 4.947032 | 4.016746 | 3.050146 | 0.577893 |
| XGBoost | 0.799394 | 0.453157 | 0.146993 | 4.884119 | 4.001018 | 3.024009 | 0.575973 |
| Fixed 50/50 blend | 0.801969 | 0.457923 | 0.149348 | 4.932229 | 4.017671 | 3.037424 | 0.579424 |

All eight model folds produced AUC above 0.79. The pooled weekly-Gini slopes are close to zero and non-negative for all three scores. This supports model effectiveness on the inner time replay, while the gain from the fixed blend over LightGBM alone is small.

The checked run included all 100 prebuilt features (`exclude_time_proxies=false`) because this iteration focused on model training and OOF validation. It is a temporal compatibility baseline, not a completed field-governance or deployment approval.

## Included

- [`overall_metrics.csv`](overall_metrics.csv)
- [`fold_metrics.csv`](fold_metrics.csv)
- [`weekly_metrics.csv`](weekly_metrics.csv)
- [`fold_manifest.csv`](fold_manifest.csv)
- [`feature_manifest.csv`](feature_manifest.csv)
- [`categorical_unseen_by_fold.csv`](categorical_unseen_by_fold.csv)
- [`feature_importance_by_fold.csv`](feature_importance_by_fold.csv)
- [`feature_importance_summary.csv`](feature_importance_summary.csv)
- [`run_config.json`](run_config.json)
- [`RUN_REPORT.md`](RUN_REPORT.md)

## Kept local

The following are generated and validated locally but intentionally not committed:

- 559,660-row `oof_predictions.parquet`, `oof_labels.parquet`, and `oof_evaluation.parquet`;
- eight LightGBM and eight XGBoost fold models;
- eight fold-specific category/frequency preprocessing maps.

They are reproducible with the command in `run_config.json`. The raw Kaggle Parquet and large generated artifacts remain subject to their source terms and repository size constraints.

This snapshot does not contain an outer-quarantine score, a final week 0–71 deployment model, probability calibration, threshold selection, or business-cost optimization.
