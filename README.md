# Home Credit Risk Diagnostics V3

> [中文完整结果报告：真实数据、执行过程、关键结果与限制](docs/RESULT_REPORT_ZH.md)
>
> [Track B 中文训练手册：LightGBM/XGBoost Rolling OOF](docs/TRACK_B_MODELING_ZH.md)
>
> [Track B 已核验的完整 8 折轻量结果快照](track_b_results/README.md)

This repository keeps the two original competition notebooks unchanged. The
Diagnostics V3 `run` path analyzes an **existing customer-level feature parquet**
without rebuilding the raw tables or rerunning the original LightGBM/CatBoost flow.
The separate Track B `train-track-b` path does train new LightGBM/XGBoost models.

The diagnostic layer provides:

- chronological reference/actual splitting using sorted, real week values;
- reference-fit numeric and categorical bins;
- WOE and IV on reference data;
- fixed-bin PSI and later-period single-feature AUC, KS, Lift@10% and Lift@20%;
- explicit missing, rare-category and unseen-category buckets;
- sparse-rescue, extreme-sparse, constant and high-drift flags;
- deterministic sampled Spearman correlation review recommendations;
- optional, exactly `case_id`-aligned OOF AUC/KS/Lift and Home Credit weekly Gini stability.

Track B now adds an independent `train-track-b` path for label-delay rolling-origin
LightGBM/XGBoost validation. It does not change or execute either original notebook.

Training-set predictions must not be supplied as OOF. The tool validates keys and
coverage, but prediction provenance cannot be inferred from values alone.

## Install

```bash
python -m pip install -e ".[test]"
```

Install the optional model-training dependencies when running Track B:

```bash
python -m pip install -e ".[training,test]"
```

## Track B LightGBM/XGBoost rolling OOF

The default schedule simulates an eight-complete-week label maturity lag, uses
four-week rolling validation blocks, and reserves weeks 80–91 as an unevaluated
outer quarantine. Early stopping uses only the last mature weeks inside each
training window; the future OOF block is never passed to `fit()`.

Full real-data run:

```bash
home-credit-diagnostics train-track-b \
  --train data/kaggle/base_100features.parquet \
  --out track_b_outputs_full \
  --models lightgbm,xgboost \
  --label-maturity-lag-weeks 8 \
  --validation-weeks 4 \
  --outer-quarantine-weeks 12 \
  --xgboost-device cpu \
  --lightgbm-device cpu \
  --run-id track-b-lag8-full-v1
```

For a quick code-path check, add `--max-folds 1 --max-rows-per-week 1000
--n-estimators 50 --early-stopping-rounds 10`. Any run with
`--max-folds` or `--max-rows-per-week` is labeled partial development/smoke and must
not be reported as final model performance. See the
[Chinese Track B guide](docs/TRACK_B_MODELING_ZH.md) for the fold timeline, model
parameters, output schema, and exact interpretation boundary. The checked full run
is available as a [lightweight result snapshot](track_b_results/README.md).

## Real Kaggle-data run

The checked real run uses the Kaggle dataset
[`diarray/deep-feature-synthesis-home-credit-stability`](https://www.kaggle.com/datasets/diarray/deep-feature-synthesis-home-credit-stability), file
`base_100features.parquet`. Download it without committing it:

The download command assumes the external Kaggle CLI is installed (for example,
`python -m pip install kaggle`) and any required login/rules acceptance is complete;
the CLI is not installed by this project's Python dependencies.

```bash
kaggle datasets download \
  -d diarray/deep-feature-synthesis-home-credit-stability \
  -f base_100features.parquet \
  -p data/kaggle \
  --unzip
```

Run Diagnostics V3:

```bash
home-credit-diagnostics run \
  --train data/kaggle/base_100features.parquet \
  --out real_outputs \
  --dataset-label "Kaggle diarray base_100features.parquet" \
  --source-url "https://www.kaggle.com/datasets/diarray/deep-feature-synthesis-home-credit-stability"
```

The loader restores `case_id` when parquet metadata stores it as an index and
resolves `WEEK_NUM` case-insensitively to the configured `week_num` field.

With genuine OOF scores:

```bash
home-credit-diagnostics run \
  --train data/kaggle/base_100features.parquet \
  --oof /path/to/oof_predictions.parquet \
  --score-cols lgb_oof,cat_oof,blend_oof \
  --out real_outputs_with_oof
```

The current original notebooks do not save assembled OOF predictions, so model-level
diagnostics are intentionally skipped in the checked real run.

Track B rolling OOF covers only its eligible future weeks and therefore must not be
passed to this older Diagnostics V3 `run --oof` interface, which intentionally
requires exact `case_id` coverage of the full input matrix.

### Checked real-run evidence

The committed `real_outputs/` report was generated from the downloaded Kaggle file,
not the synthetic generator. Input validation observed:

- 1,526,659 unique customer rows;
- 103 stored fields including metadata, with 100 diagnosed features;
- 92 real `WEEK_NUM` values from 0 through 91;
- 47,994 target positives, a 3.1437% bad rate;
- `case_id` restored from the parquet index and validated as unique.

No OOF file is present in the source repository, so the real run correctly leaves
model diagnostics empty rather than substituting training predictions.

The uploader page labels its dataset MIT. That label does not replace the underlying
[Home Credit competition rules](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/rules).
Code licensing and data-use terms are separate; this repository does not redistribute
the downloaded parquet or Kaggle zip.

## Synthetic smoke test

Synthetic data exists only to validate the pipeline without Kaggle credentials. Its
report is labeled **SYNTHETIC SMOKE TEST ONLY** and is not a final result:

```bash
home-credit-diagnostics demo \
  --data-dir demo_data \
  --out demo_outputs
```

## AMEX patterns: adopted and not migrated

The implementation was cross-checked against the Kaggle notebooks
[`Amex Agg Data How It Created`](https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created)
and [`Lag Features Are All You Need`](https://www.kaggle.com/code/thedevastator/lag-features-are-all-you-need).
Only their general aggregation patterns are referenced; their derived AMEX data is
not copied or redistributed here.

Adopted:

- numeric mean/std/min/max and existing last values;
- categorical count/last/nunique/mode;
- last-step difference and last-minus-mean only after sorting by customer and an
  explicitly supplied time column;
- preservation of genuine OOF scores for separate evaluation.

Not migrated:

- any first/last/difference calculation based on current parquet row order;
- the AMEX competition metric—Home Credit keeps sorted real-week Gini stability;
- the Transformer sequence model, because Home Credit does not have one standardized,
  homogeneous customer sequence.

## Optional AMEX-style history aggregation

The original Home Credit matrix already contains many `COUNT`, `MODE`, `STD`, `SUM`,
`MAX`, `MIN` and delta features. Diagnostics V3 analyzes those existing fields rather
than regenerating them.

For a separate history table, the optional helper supports numeric
`count/mean/std/min/max/sum/nunique`, categorical `count/nunique/mode`, and—only when
an explicit time column is supplied—`first/last/last-minus-first/last-minus-mean`:

```bash
home-credit-diagnostics aggregate \
  --input /path/to/history.parquet \
  --out /path/to/history_aggregated.parquet \
  --id-col case_id \
  --time-col event_date \
  --tie-breaker-col event_sequence \
  --categorical-cols status,product_code
```

Without `--time-col`, temporal features are not generated. Parquet row order is never
used to imitate a time series. Duplicate customer/time rows fail fast unless an
explicit tie-breaker column uniquely orders them. This corrects the unsafe row-order
assumption seen in some AMEX examples while retaining the justified aggregation
patterns. Transformer sequence modeling is deliberately excluded: Home Credit
history tables do not form one standardized, homogeneous customer sequence.

## Diagnostics V3 outputs

- `feature_diagnostics.csv`
- `bin_diagnostics.csv`
- `correlation_recommendations.csv`
- `model_diagnostics.csv`
- `weekly_model_metrics.csv`
- `split_summary.csv`
- `summary.json`
- `report.md`

Correlation recommendations are review aids, not automatic deletion decisions.
Changing upstream aggregation or feature filtering would change the matrix and require
a full model rerun, so that work is intentionally outside this minimal integration.
