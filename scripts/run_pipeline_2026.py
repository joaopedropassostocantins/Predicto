#!/usr/bin/env python3
"""
scripts/run_pipeline_2026.py
============================
Unified entry point for the March Machine Learning Mania 2026 pipeline.

Usage
-----
# Full pipeline: backtest + final model + submission
python scripts/run_pipeline_2026.py

# Backtest only
python scripts/run_pipeline_2026.py --mode backtest

# Submission only (skips backtest)
python scripts/run_pipeline_2026.py --mode submit

# Backtest on a subset of seasons (faster development cycle)
python scripts/run_pipeline_2026.py --mode backtest --seasons 2021 2022 2023 2024

Options
-------
--mode       backtest | submit | all  (default: all)
--seasons    Space-separated list of seasons for backtest (default: config)
--output     Output path for submission CSV (default: submission.csv)
--data-dir   Override data_dir from config
"""

import argparse
import os
import sys
import time

# Ensure project root is on the path whether run from project root or scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.config import CONFIG
from src.backtest import rolling_backtest, save_backtest_outputs
from src.submit import generate_submission


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="March ML Mania 2026 — Full Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "submit", "all"],
        default="all",
        help="Which part of the pipeline to run.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Seasons for backtest (default: config backtest_seasons).",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output path for submission CSV.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override CONFIG['data_dir'].",
    )
    parser.add_argument(
        "--backtest-dir",
        default="backtest_results",
        help="Directory to save backtest CSVs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = dict(CONFIG)  # shallow copy so we can override keys
    if args.data_dir:
        cfg["data_dir"] = args.data_dir

    seasons = args.seasons or cfg["backtest_seasons"]

    # ── Banner ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  March Machine Learning Mania 2026 — Predicto Pipeline v3.0")
    print("=" * 70)
    print(f"  data_dir       : {cfg['data_dir']}")
    print(f"  target_season  : {cfg['target_season']}")
    print(f"  backtest_seasons: {seasons}")
    print(f"  mode           : {args.mode}")
    print("=" * 70)
    print()

    t0 = time.time()

    # ── Backtest ───────────────────────────────────────────────────────────
    if args.mode in ("backtest", "all"):
        print(">>> STEP 1: Rolling Backtest")
        t1 = time.time()
        results = rolling_backtest(
            seasons=seasons,
            cfg=cfg,
            genders=("M", "W"),
            calibrate=True,
        )
        save_backtest_outputs(results, args.backtest_dir)

        print()
        print("━" * 60)
        print("BACKTEST SUMMARY")
        print("━" * 60)
        print(results["summary"].to_string(index=False))
        print()

        summary = results["summary"]
        if len(summary) > 0:
            print("── Aggregate Metrics ──")
            print(f"  Mean Brier   : {summary['brier'].mean():.4f}")
            print(f"  Mean Accuracy: {summary['accuracy'].mean():.4f}")
            print(f"  Mean LogLoss : {summary['log_loss'].mean():.4f}")
            print(f"  Best Brier   : {summary['brier'].min():.4f}  "
                  f"(season {summary.loc[summary['brier'].idxmin(), 'Season']})")
            print()

        print("── Calibration Table (all seasons pooled) ──")
        print(results["calibration_table"]
              .dropna(subset=["avg_pred"])
              .to_string(index=False))
        print()

        print("── Probability Bands ──")
        print(results["probability_bands"]
              .dropna(subset=["avg_pred"])
              .to_string(index=False))
        print()

        print(f"Backtest done in {time.time() - t1:.1f}s")
        print()

    # ── Submission ─────────────────────────────────────────────────────────
    if args.mode in ("submit", "all"):
        print(">>> STEP 2: Generate Submission")
        t2 = time.time()
        submission = generate_submission(cfg, output_path=args.output)
        print(f"Submission done in {time.time() - t2:.1f}s")
        print()
        print("── Submission Preview ──")
        print(submission.head(10).to_string(index=False))
        print()

    print(f"Total elapsed: {time.time() - t0:.1f}s")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
