#!/usr/bin/env python3
# scripts/validate.py — Temporal walk-forward validation pipeline v4.0
#
# Runs rolling backtest, prints calibration audit, and saves all outputs.
#
# Usage:
#   python scripts/validate.py [--data-dir DIR] [--output-dir DIR]
#                               [--seasons 2021 2022 2023] [--no-calibrate]
#
# Output:
#   output_dir/
#     summary.csv              — metrics per season
#     predictions.csv          — all predictions with components
#     calibration_table.csv    — reliability plot data
#     probability_bands.csv    — empirical win rate by confidence band
#     seasonal_metrics.csv     — per-season breakdown
#     blend_sensitivity.csv    — blend weight sensitivity analysis

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, reload_config
from src.backtest import rolling_backtest, save_backtest_outputs
from src.metrics import comparison_table


def parse_args():
    p = argparse.ArgumentParser(description="Run temporal walk-forward validation")
    p.add_argument("--data-dir",    type=str, default=None)
    p.add_argument("--output-dir",  type=str, default="backtest_results")
    p.add_argument("--seasons",     type=int, nargs="+", default=None,
                   help="Backtest seasons (default: all in config)")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Disable post-model calibration")
    p.add_argument("--genders",     type=str, nargs="+", default=["M", "W"])
    p.add_argument("--baseline",    action="store_true",
                   help="Run baseline comparison table alongside main backtest")
    return p.parse_args()


def main():
    args  = parse_args()
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    cfg = reload_config(overrides)

    seasons = args.seasons or cfg["backtest_seasons"]

    print("=" * 60)
    print("Predicto v4.0 — Walk-Forward Validation")
    print("=" * 60)
    print(f"Seasons: {seasons}")
    print(f"Genders: {args.genders}")
    print(f"Calibration: {'enabled' if not args.no_calibrate else 'DISABLED'}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 60)

    results = rolling_backtest(
        seasons=seasons,
        cfg=cfg,
        genders=tuple(args.genders),
        calibrate=not args.no_calibrate,
        verbose=True,
    )

    # ── Save all outputs ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_backtest_outputs(results, args.output_dir)

    # ── Print calibration summary ─────────────────────────────────────────
    print("\n=== Calibration Table ===")
    cal_tbl = results["calibration_table"]
    print(cal_tbl[["bin_left", "bin_right", "n", "avg_pred", "emp_rate", "abs_gap"]]
          .dropna()
          .to_string(index=False, float_format="%.3f"))

    print("\n=== Probability Bands ===")
    print(results["probability_bands"].dropna().to_string(index=False, float_format="%.3f"))

    # ── Baseline comparison (optional) ────────────────────────────────────
    if args.baseline:
        from src.tuning import baseline_comparison
        print("\n=== Baseline Comparison ===")
        cmp = baseline_comparison(seasons[-5:], base_cfg=cfg, verbose=True)
        cmp.to_csv(os.path.join(args.output_dir, "baseline_comparison.csv"), index=False)

    # ── Component-level comparison ────────────────────────────────────────
    all_preds = results["predictions"]
    if "p_elo" in all_preds.columns:
        component_preds = {
            "p_elo":     all_preds["p_elo"].values,
            "p_poisson": all_preds["p_poisson"].values,
            "p_xgb":     all_preds["p_xgb"].values,
            "p_manual":  all_preds["p_manual"].values,
            "Ensemble":  all_preds["Pred"].values,
        }
        cmp_tbl = comparison_table(all_preds["ActualLowWin"].values, component_preds)
        print("\n=== Component Comparison ===")
        print(cmp_tbl.to_string(index=False))
        cmp_tbl.to_csv(os.path.join(args.output_dir, "component_comparison.csv"), index=False)

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
