# src/config.py — March Machine Learning Mania 2026 — v3.0

import os

# ---------------------------------------------------------------------------
# Auto-detect Kaggle vs local environment
# ---------------------------------------------------------------------------
_ON_KAGGLE = os.path.exists("/kaggle/input")

CONFIG = {
    # ── Data paths ─────────────────────────────────────────────────────────
    "data_dir": (
        "/kaggle/input/march-machine-learning-mania-2026"
        if _ON_KAGGLE
        else "/home/ubuntu/predicto_local/data"
    ),

    # ── Competition target ─────────────────────────────────────────────────
    "target_season": 2026,

    # Seasons used for rolling backtest.
    # • Exclude 2020 (COVID — no tournament).
    # • Verify your data files cover all listed seasons before running.
    # • Earlier seasons provide XGBoost training diversity.
    "backtest_seasons": [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],

    # ── Feature windows ────────────────────────────────────────────────────
    "recent_games_window": 3,
    "poisson_windows": [3, 5, "season"],
    "poisson_blend_weights": {
        "recent3": 0.35,
        "recent5": 0.30,
        "season": 0.35,
    },

    # ── Poisson settings ───────────────────────────────────────────────────
    "alpha_ci": 0.10,
    "max_points_poisson": 220,
    # Shrink per-team lambda toward league mean (Bayesian regularisation).
    # 0.0 = no shrinkage; 1.0 = always use league mean.
    "poisson_shrinkage": 0.20,

    # ── Probability clipping ───────────────────────────────────────────────
    # Tighter than 1e-6 to prevent Brier/LogLoss blowup on confident errors.
    "pred_clip_min": 0.025,
    "pred_clip_max": 0.975,

    # ── Elo ────────────────────────────────────────────────────────────────
    "elo_k_factor": 25.0,
    "elo_initial_rating": 1500.0,
    # Fraction of end-of-season Elo carried into the next season.
    # new_start = carry * end_elo + (1 - carry) * 1500
    "elo_carry_factor": 0.75,
    # Weight Elo update by log(margin+1) instead of binary win/loss.
    "elo_use_margin": True,
    # Cap the raw point margin used for the log-margin factor.
    "elo_margin_cap": 30.0,

    # ── Manual model ───────────────────────────────────────────────────────
    "manual_temperature": 10.0,

    # ── Temperature scaling candidates ────────────────────────────────────
    "temperature_candidates": [0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 2.00],

    # ── Calibration methods ────────────────────────────────────────────────
    "calibration_methods": ["identity", "temperature", "platt", "isotonic"],

    # ── Fallback values (for teams with missing data) ──────────────────────
    "fallback_points_for": 70.0,
    "fallback_points_against": 70.0,
    "fallback_seed": 8.5,
    "fallback_elo": 1500.0,

    # ── XGBoost hyperparameters ────────────────────────────────────────────
    "tabular_model": "xgb_or_hgb",
    "xgb_params": {
        "n_estimators": 700,
        "learning_rate": 0.02,
        "max_depth": 4,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.2,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
    },
    # Fallback if XGBoost is not installed.
    "hgb_params": {
        "learning_rate": 0.03,
        "max_depth": 4,
        "max_iter": 600,
        "min_samples_leaf": 15,
        "l2_regularization": 0.5,
        "random_state": 42,
    },

    # ── Blend weights ──────────────────────────────────────────────────────
    # XGBoost is the PRIMARY model; others are auxiliary calibrating signals.
    "blend_weights": {
        "xgb":     0.55,
        "poisson": 0.20,
        "seed":    0.10,
        "elo":     0.10,
        "manual":  0.05,
    },

    # ── Manual model feature weights ───────────────────────────────────────
    "manual_feature_weights": {
        "seed_diff":              0.90,
        "elo_diff":               1.10,
        "season_margin_diff":     1.20,
        "season_win_pct_diff":    0.80,
        "recent3_margin_diff":    1.25,
        "recent5_margin_diff":    1.00,
        "matchup_diff":           1.10,
        "poisson_win_prob_centered": 2.20,
        "quality_diff":           1.00,
        "sos_diff":               0.70,
        "rank_diff_signed":       0.60,
        "consistency_edge":       0.35,
    },

    # ── XGBoost feature columns ────────────────────────────────────────────
    "feature_cols": [
        # Seeding (one of the strongest signals)
        "seed_diff",

        # Elo strength
        "elo_diff",

        # Full-season aggregates
        "season_points_for_diff",
        "season_points_against_diff",
        "season_margin_diff",
        "season_win_pct_diff",

        # Recent form — last 3 games
        "recent3_points_for_diff",
        "recent3_points_against_diff",
        "recent3_margin_diff",

        # Recent form — last 5 games
        "recent5_points_for_diff",
        "recent5_points_against_diff",
        "recent5_margin_diff",

        # Matchup: offense vs opponent defense
        "matchup_diff",
        "offense_vs_defense_low",
        "offense_vs_defense_high",

        # Strength of schedule & quality
        "sos_diff",
        "quality_diff",
        "rank_diff_signed",

        # Poisson-derived signals
        "poisson_lambda_low",
        "poisson_lambda_high",
        "poisson_expected_margin",
        "poisson_win_prob",
        "poisson_win_prob_centered",

        # Consistency (scoring variance)
        "consistency_edge",

        # ── v3 new features ──────────────────────────────────────────────
        # Season trajectory: late-season margin minus early-season margin
        "season_trajectory_diff",
        # Win rate vs quality opponents (above-median margin teams)
        "quality_win_pct_diff",
        # Blowout rate: fraction of games won by >15 pts
        "blowout_pct_diff",
        # Clutch: win rate in games decided by <5 pts
        "close_game_win_pct_diff",
    ],

    # ── Massey Ordinals ────────────────────────────────────────────────────
    # Prioritised ranking systems to use as a composite massey_rank feature.
    # Add "massey_rank_diff" to feature_cols if this file is available.
    "massey_system_priority": ["POM", "SAG", "MOR", "DOL", "WOL", "RTH", "COL"],
}
