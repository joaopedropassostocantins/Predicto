#!/usr/bin/env python3
# scripts/make_submission.py — Submission generation v4.0
#
# Full pipeline: train on all backtest seasons → calibrate → write submission.csv
#
# Usage:
#   python scripts/make_submission.py [--data-dir DIR] [--output FILE]
#                                      [--target-season YEAR]
#
# Equivalent to running scripts/train.py — kept as a separate entry point for
# compatibility with Kaggle notebooks and competition workflows.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, reload_config
from src.submit import generate_submission


def parse_args():
    p = argparse.ArgumentParser(description="Generate Kaggle submission CSV")
    p.add_argument("--data-dir",      type=str, default=None,
                   help="Override data directory path")
    p.add_argument("--output",        type=str, default="submission.csv",
                   help="Output CSV path (default: submission.csv)")
    p.add_argument("--target-season", type=int, default=None,
                   help="Override target season (default: from config)")
    p.add_argument("--seasons",       type=int, nargs="+", default=None,
                   help="Override backtest seasons")
    return p.parse_args()


def main():
    args      = parse_args()
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.target_season:
        overrides["target_season"] = args.target_season
    if args.seasons:
        overrides["backtest_seasons"] = args.seasons

    cfg = reload_config(overrides)

    print("=" * 60)
    print("Predicto v4.0 — Submission Generator")
    print("=" * 60)
    print(f"Target season:    {cfg['target_season']}")
    print(f"Backtest seasons: {cfg['backtest_seasons']}")
    print(f"Data directory:   {cfg['data_dir']}")
    print(f"Output:           {args.output}")
    print("=" * 60)

    submission = generate_submission(cfg, output_path=args.output)
    print(f"\nSubmission generated: {args.output} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
