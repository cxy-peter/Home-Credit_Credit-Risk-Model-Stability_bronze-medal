from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .aggregation import aggregate_history
from .io import read_table
from .pipeline import DiagnosticsConfig, run_diagnostics
from .synthetic import write_synthetic_demo
from .training import TrackBTrainingConfig, load_track_b_frame, run_track_b_training


def _comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _add_common_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--id-col", default="case_id")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--week-col", default="week_num")
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-categories", type=int, default=100)
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    parser.add_argument("--correlation-sample-rows", type=int, default=50_000)
    parser.add_argument("--feature-cols", help="Optional comma-separated feature subset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="home-credit-diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Diagnose an existing customer-level parquet without retraining"
    )
    run_parser.add_argument("--train", required=True, type=Path)
    run_parser.add_argument(
        "--oof",
        type=Path,
        help="Optional genuine OOF table with one unique case_id per training row",
    )
    run_parser.add_argument("--score-cols", help="Comma-separated OOF score columns")
    run_parser.add_argument("--dataset-label", default="User-supplied customer-level data")
    run_parser.add_argument("--source-url", help="Optional dataset source URL recorded in the report")
    _add_common_run_options(run_parser)

    demo_parser = subparsers.add_parser(
        "demo", help="Generate deterministic synthetic smoke data and exercise the pipeline"
    )
    demo_parser.add_argument("--data-dir", required=True, type=Path)
    demo_parser.add_argument("--rows", type=int, default=12_000)
    demo_parser.add_argument("--weeks", type=int, default=60)
    demo_parser.add_argument("--seed", type=int, default=42)
    _add_common_run_options(demo_parser)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Build optional AMEX-style aggregates from a history table"
    )
    aggregate_parser.add_argument("--input", required=True, type=Path)
    aggregate_parser.add_argument("--out", required=True, type=Path)
    aggregate_parser.add_argument("--id-col", default="case_id")
    aggregate_parser.add_argument(
        "--time-col",
        help="Explicit time column required for first/last/delta aggregates",
    )
    aggregate_parser.add_argument(
        "--categorical-cols",
        help="Comma-separated categorical columns, including integer-encoded categories",
    )
    aggregate_parser.add_argument(
        "--tie-breaker-col",
        help="Explicit deterministic order for duplicate customer/time rows",
    )

    training_parser = subparsers.add_parser(
        "train-track-b",
        help="Train LightGBM/XGBoost with label-delay rolling-origin OOF validation",
    )
    training_parser.add_argument("--train", required=True, type=Path)
    training_parser.add_argument("--out", required=True, type=Path)
    training_parser.add_argument("--id-col", default="case_id")
    training_parser.add_argument("--target-col", default="target")
    training_parser.add_argument("--week-col", default="week_num")
    training_parser.add_argument("--models", default="lightgbm,xgboost")
    training_parser.add_argument("--feature-cols", help="Optional comma-separated feature subset")
    training_parser.add_argument("--exclude-cols", help="Optional comma-separated model exclusions")
    training_parser.add_argument("--label-maturity-lag-weeks", type=int, default=8)
    training_parser.add_argument("--validation-weeks", type=int, default=4)
    training_parser.add_argument("--step-weeks", type=int, default=4)
    training_parser.add_argument("--min-mature-train-weeks", type=int, default=32)
    training_parser.add_argument("--inner-early-stopping-weeks", type=int, default=4)
    training_parser.add_argument("--outer-quarantine-weeks", type=int, default=12)
    training_parser.add_argument(
        "--max-folds",
        type=int,
        help="Development/smoke limit; omitted runs every eligible rolling fold",
    )
    training_parser.add_argument("--n-estimators", type=int, default=2_000)
    training_parser.add_argument("--learning-rate", type=float, default=0.03)
    training_parser.add_argument("--early-stopping-rounds", type=int, default=150)
    training_parser.add_argument("--n-jobs", type=int, default=12)
    training_parser.add_argument("--xgboost-device", default="cpu", choices=["cpu", "cuda"])
    training_parser.add_argument(
        "--lightgbm-device", default="cpu", choices=["cpu", "gpu", "cuda"]
    )
    training_parser.add_argument("--high-cardinality-threshold", type=int, default=10_000)
    training_parser.add_argument(
        "--exclude-time-proxies",
        action="store_true",
        help="Optional stricter run: keep recognized calendar/time proxies out of the model",
    )
    training_parser.add_argument(
        "--max-rows-per-week",
        type=int,
        help="Deterministic real-data smoke sampling; never quote sampled metrics as final",
    )
    training_parser.add_argument(
        "--run-id", help="Audit run ID; defaults to track-b-lag<configured lag>-v1"
    )
    training_parser.add_argument(
        "--dataset-label", default="User-supplied customer-level feature matrix"
    )
    return parser


def _config_from_args(args: argparse.Namespace, *, synthetic_demo: bool) -> DiagnosticsConfig:
    return DiagnosticsConfig(
        id_col=args.id_col,
        target_col=args.target_col,
        week_col=args.week_col,
        reference_fraction=args.reference_fraction,
        n_bins=args.n_bins,
        max_categories=args.max_categories,
        correlation_threshold=args.correlation_threshold,
        correlation_sample_rows=args.correlation_sample_rows,
        dataset_label=(
            "Deterministic synthetic smoke data — not final Kaggle results"
            if synthetic_demo
            else args.dataset_label
        ),
        source_url=None if synthetic_demo else args.source_url,
        synthetic_demo=synthetic_demo,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "aggregate":
        history = read_table(args.input)
        aggregated = aggregate_history(
            history,
            id_col=args.id_col,
            time_col=args.time_col,
            tie_breaker_col=args.tie_breaker_col,
            categorical_cols=_comma_list(args.categorical_cols),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_parquet(args.out, index=False)
        print(f"Wrote {len(aggregated):,} aggregated rows to {args.out}")
        if args.time_col is None:
            print("Temporal first/last/delta features were not generated because --time-col was omitted.")
        return 0

    if args.command == "train-track-b":
        if args.out.exists():
            parser.error(
                f"Track B output already exists: {args.out}. Choose a fresh --out path."
            )
        config = TrackBTrainingConfig(
            id_col=args.id_col,
            target_col=args.target_col,
            week_col=args.week_col,
            label_maturity_lag_weeks=args.label_maturity_lag_weeks,
            validation_weeks=args.validation_weeks,
            step_weeks=args.step_weeks,
            min_mature_train_weeks=args.min_mature_train_weeks,
            inner_early_stopping_weeks=args.inner_early_stopping_weeks,
            outer_quarantine_weeks=args.outer_quarantine_weeks,
            max_folds=args.max_folds,
            models=tuple(_comma_list(args.models) or []),
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            early_stopping_rounds=args.early_stopping_rounds,
            n_jobs=args.n_jobs,
            xgboost_device=args.xgboost_device,
            lightgbm_device=args.lightgbm_device,
            high_cardinality_threshold=args.high_cardinality_threshold,
            exclude_time_proxies=args.exclude_time_proxies,
            max_rows_per_week=args.max_rows_per_week,
            run_id=args.run_id or f"track-b-lag{args.label_maturity_lag_weeks}-v1",
            dataset_label=args.dataset_label,
            input_path=str(args.train.resolve()),
            invocation_argv=tuple(sys.argv),
        )
        train = load_track_b_frame(args.train, config)
        run = run_track_b_training(
            train,
            args.out,
            config=config,
            feature_cols=_comma_list(args.feature_cols),
            exclude_cols=_comma_list(args.exclude_cols),
        )
        print(
            f"Track B wrote {len(run.oof_predictions):,} rolling OOF rows "
            f"for {len(run.fold_manifest)} folds to {args.out}"
        )
        if args.max_rows_per_week is not None or args.max_folds is not None:
            print(
                "PARTIAL DEVELOPMENT/SMOKE ONLY: fold limiting or row sampling was enabled; "
                "do not quote these metrics as final."
            )
        print("Outer quarantine was reserved and was not evaluated by this command.")
        return 0

    if args.command == "demo":
        train_path, oof_path = write_synthetic_demo(
            args.data_dir,
            n_rows=args.rows,
            n_weeks=args.weeks,
            seed=args.seed,
        )
        train = read_table(train_path)
        oof = read_table(oof_path)
        config = _config_from_args(args, synthetic_demo=True)
        run = run_diagnostics(
            train,
            args.out,
            config=config,
            feature_cols=_comma_list(args.feature_cols),
            oof=oof,
            score_cols=["existing_simulated_oof", "challenger_simulated_oof"],
        )
        print(f"Synthetic smoke test wrote {run.summary['features']} feature diagnostics to {args.out}")
        print("This is smoke validation only, not the final Kaggle-data result.")
        return 0

    train = read_table(args.train)
    oof = read_table(args.oof) if args.oof else None
    config = _config_from_args(args, synthetic_demo=False)
    run = run_diagnostics(
        train,
        args.out,
        config=config,
        feature_cols=_comma_list(args.feature_cols),
        oof=oof,
        score_cols=_comma_list(args.score_cols),
    )
    print(f"Wrote {run.summary['features']} feature diagnostics to {args.out}")
    if oof is None:
        print("Model diagnostics skipped: no genuine OOF file was supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
