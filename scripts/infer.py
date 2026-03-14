#!/usr/bin/env python3
# scripts/infer.py — Inference pipeline v4.0
#
# Generates predictions for the target season from a set of explicit matchups.
# Useful for: predicting specific games, bracket simulation, or testing.
#
# Usage:
#   python scripts/infer.py [--data-dir DIR] [--matchups FILE] [--output FILE]
#
# Matchups file (CSV) format:
#   Season, TeamIDLow, TeamIDHigh, Gender
#   2026, 1101, 1242, M
#   2026, 3101, 3242, W
#
# If --matchups is not provided, uses sample_submission.csv matchups.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import CONFIG, reload_config
from src.data import (
    load_regular_season_detailed,
    load_seeds,
    load_sample_submission,
    parse_submission_ids,
)
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.models import compute_all_probabilities, get_feature_importance
from src.calibration import apply_calibrator, fit_calibrator
from src.ratings import precompute_starting_elo
from src.submit import generate_submission, validate_submission


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on explicit matchups or submission template")
    p.add_argument("--data-dir",  type=str, default=None)
    p.add_argument("--matchups",  type=str, default=None,
                   help="CSV file with matchups (Season, TeamIDLow, TeamIDHigh, Gender)")
    p.add_argument("--output",    type=str, default="predictions.csv")
    p.add_argument("--target-season", type=int, default=None)
    p.add_argument("--use-calibrator", type=str, default="temperature",
                   choices=["identity", "temperature", "platt", "isotonic"],
                   help="Calibrator to apply to predictions")
    p.add_argument("--explain",   action="store_true",
                   help="Print top feature importances")
    return p.parse_args()


def main():
    args      = parse_args()
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.target_season:
        overrides["target_season"] = args.target_season
    cfg = reload_config(overrides)

    target = cfg["target_season"]
    print(f"Inference for season {target} ...")

    # ── Load data ──────────────────────────────────────────────────────────
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_seeds   = load_seeds(cfg["data_dir"], "M")
    w_seeds   = load_seeds(cfg["data_dir"], "W")

    # ── Cross-season Elo ───────────────────────────────────────────────────
    elo_params = dict(
        k_factor=cfg["elo_k_factor"],
        initial_rating=cfg["elo_initial_rating"],
        carry_factor=cfg["elo_carry_factor"],
        use_margin=cfg.get("elo_use_margin", True),
        margin_cap=cfg.get("elo_margin_cap", 15.0),
    )

    def _elo_games(df):
        return (
            df[df["Win"] == 1][["Season", "DayNum", "TeamID", "OppTeamID", "Margin"]]
            .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
            .drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID"])
            .reset_index(drop=True)
        )

    starting_elo_m = precompute_starting_elo(_elo_games(m_regular), **elo_params)
    starting_elo_w = precompute_starting_elo(_elo_games(w_regular), **elo_params)

    # ── Load or generate matchups ──────────────────────────────────────────
    if args.matchups:
        matchup_df = pd.read_csv(args.matchups)
        # Ensure required columns
        for col in ["Season", "TeamIDLow", "TeamIDHigh"]:
            if col not in matchup_df.columns:
                raise ValueError(f"Matchups file missing column: {col}")
        if "Gender" not in matchup_df.columns:
            matchup_df["Gender"] = "M"
        if "ID" not in matchup_df.columns:
            matchup_df["ID"] = (
                matchup_df["Season"].astype(str) + "_"
                + matchup_df["TeamIDLow"].astype(str) + "_"
                + matchup_df["TeamIDHigh"].astype(str)
            )
    else:
        # Use sample submission
        sample_sub = load_sample_submission(cfg["data_dir"])
        matchup_df = parse_submission_ids(sample_sub)
        matchup_df = matchup_df[matchup_df["Season"] == target].copy()

    if len(matchup_df) == 0:
        print("No matchups found. Exiting.")
        return

    print(f"Predicting {len(matchup_df)} matchups...")

    # ── Build features ─────────────────────────────────────────────────────
    m_feat = build_team_features(
        m_regular, target, cfg["recent_games_window"], cfg["alpha_ci"],
        starting_elo=starting_elo_m, cfg=cfg,
    )
    w_feat = build_team_features(
        w_regular, target, cfg["recent_games_window"], cfg["alpha_ci"],
        starting_elo=starting_elo_w, cfg=cfg,
    )

    pred_df = attach_team_features(matchup_df, m_feat, w_feat, m_seeds, w_seeds, cfg)
    pred_df = make_matchup_features(pred_df, cfg=cfg)

    # ── Predict (no training data → auxiliary blend only) ─────────────────
    # For full ensemble, run train.py which trains XGBoost on all backtest seasons.
    pred_df = compute_all_probabilities(pred_df, cfg, train_df=None)

    # ── Apply calibrator ───────────────────────────────────────────────────
    dummy_p = pred_df["Pred"].values
    dummy_y = (pred_df["Pred"].values > 0.5).astype(int)  # pseudo labels
    cal = fit_calibrator(args.use_calibrator, dummy_p, dummy_y, cfg=cfg)
    pred_df["PredCalibrated"] = apply_calibrator(cal, pred_df["Pred"].values)

    # ── Output ────────────────────────────────────────────────────────────
    out_cols = ["ID", "Season", "TeamIDLow", "TeamIDHigh",
                "p_elo", "p_poisson", "p_xgb", "p_manual", "Pred", "PredCalibrated"]
    out_cols = [c for c in out_cols if c in pred_df.columns]
    output   = pred_df[out_cols].copy()
    output.to_csv(args.output, index=False)
    print(f"Predictions saved: {args.output}")

    if args.explain:
        print("\n=== Feature Signal Summary ===")
        for col in ["p_elo", "p_poisson", "p_xgb", "p_manual"]:
            if col in pred_df.columns:
                print(f"  {col}: mean={pred_df[col].mean():.3f}  "
                      f"std={pred_df[col].std():.3f}  "
                      f"range=[{pred_df[col].min():.3f}, {pred_df[col].max():.3f}]")


if __name__ == "__main__":
    main()
