#!/usr/bin/env python3
# scripts/train.py — Final model training pipeline v4.0
#
# Trains the final ensemble model on ALL available backtest seasons and
# saves the trained artefacts for inference.
#
# Usage:
#   python scripts/train.py [--data-dir DIR] [--output-dir DIR]
#
# Output:
#   output_dir/
#     feature_importance.csv   — XGBoost feature importance
#     training_metrics.csv     — per-season OOF metrics
#     calibrator_info.txt      — selected calibrator and parameters
#     submission.csv           — final submission file

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, reload_config
from src.submit import generate_submission


def parse_args():
    p = argparse.ArgumentParser(description="Train final Predicto v4 model and generate submission")
    p.add_argument("--data-dir",    type=str, default=None,
                   help="Override data directory")
    p.add_argument("--output-dir",  type=str, default="output",
                   help="Directory for output files (default: output/)")
    p.add_argument("--output",      type=str, default=None,
                   help="Submission CSV path (default: output_dir/submission.csv)")
    p.add_argument("--target-season", type=int, default=None,
                   help="Override target season (default: from config)")
    p.add_argument("--seasons",     type=int, nargs="+", default=None,
                   help="Override backtest seasons (space-separated)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Apply overrides ───────────────────────────────────────────────────
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.target_season:
        overrides["target_season"] = args.target_season
    if args.seasons:
        overrides["backtest_seasons"] = args.seasons

    cfg = reload_config(overrides)

    # ── Output directory ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output or os.path.join(args.output_dir, "submission.csv")

    print("=" * 60)
    print("Predicto v4.0 — Training Pipeline")
    print("=" * 60)
    print(f"Target season:    {cfg['target_season']}")
    print(f"Backtest seasons: {cfg['backtest_seasons']}")
    print(f"Data directory:   {cfg['data_dir']}")
    print(f"Blend weights:    {cfg['blend_weights']}")
    print(f"Output:           {output_path}")
    print("=" * 60)

    # ── Generate submission (includes full training pipeline) ─────────────
    submission = generate_submission(cfg, output_path=output_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nDone. Submission: {output_path} ({len(submission)} rows)")
    print(f"Range: [{submission['Pred'].min():.4f}, {submission['Pred'].max():.4f}]  "
          f"Mean: {submission['Pred'].mean():.4f}")


if __name__ == "__main__":
    main()
