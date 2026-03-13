# src/config.py

CONFIG = {
    "data_dir": "/kaggle/input/competitions/march-machine-learning-mania-2026",
    "target_season": 2026,
    "backtest_seasons": [2022, 2023, 2024, 2025],

    # windows
    "recent_games_window": 3,
    "poisson_windows": [3, 5, "season"],
    "poisson_blend_weights": {
        "recent3": 0.40,
        "recent5": 0.30,
        "season": 0.30,
    },

    # uncertainty
    "alpha_ci": 0.10,
    "max_points_poisson": 220,

    # clipping
    "pred_clip_min": 1e-6,
    "pred_clip_max": 1 - 1e-6,

    # manual model
    "manual_temperature": 10.0,

    # temperature scaling over logits
    "temperature_candidates": [0.80, 1.00, 1.25, 1.50, 2.00],

    # calibration methods
    "calibration_methods": ["identity", "temperature", "platt", "isotonic"],

    # defaults
    "fallback_points_for": 70.0,
    "fallback_points_against": 70.0,
    "fallback_seed": 8.5,
    "fallback_elo": 1500.0,

    # elo
    "elo_k_factor": 20.0,
    "elo_initial_rating": 1500.0,

    # tabular model
    "tabular_model": "xgb_or_hgb",
    "xgb_params": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    },
    "hgb_params": {
        "learning_rate": 0.04,
        "max_depth": 4,
        "max_iter": 400,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 42,
    },

    # blend final
    "blend_weights": {
        "tabular": 0.45,
        "poisson": 0.25,
        "manual": 0.10,
        "seed": 0.08,
        "elo": 0.07,
        "rank": 0.05,
    },

    # manual weights
    "manual_feature_weights": {
        "seed_diff": 0.90,
        "elo_diff": 1.10,
        "season_margin_diff": 1.20,
        "season_win_pct_diff": 0.80,
        "recent3_margin_diff": 1.25,
        "recent5_margin_diff": 1.00,
        "matchup_diff": 1.10,
        "poisson_win_prob_centered": 2.20,
        "quality_diff": 1.00,
        "sos_diff": 0.70,
        "rank_diff_signed": 0.60,
        "consistency_edge": 0.35,
    },

    # feature columns for tabular model
    "feature_cols": [
        "seed_diff",
        "elo_diff",
        "season_points_for_diff",
        "season_points_against_diff",
        "season_margin_diff",
        "season_win_pct_diff",
        "recent3_points_for_diff",
        "recent3_points_against_diff",
        "recent3_margin_diff",
        "recent5_points_for_diff",
        "recent5_points_against_diff",
        "recent5_margin_diff",
        "matchup_diff",
        "offense_vs_defense_low",
        "offense_vs_defense_high",
        "sos_diff",
        "quality_diff",
        "rank_diff_signed",
        "poisson_lambda_low",
        "poisson_lambda_high",
        "poisson_expected_margin",
        "poisson_win_prob",
        "poisson_win_prob_centered",
        "consistency_edge",
    ],
}
