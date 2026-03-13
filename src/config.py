# src/config.py

CONFIG = {
    "data_dir": "/kaggle/input/march-machine-learning-mania-2026",
    "target_season": 2026,
    "recent_games_window": 3,
    "alpha_ci": 0.10,  # IC 90%
    "max_points_poisson": 220,
    "pred_clip_min": 0.02,
    "pred_clip_max": 0.98,
    "temperature_manual": 8.0,
    "temperature_poisson": 6.0,
    "blend_manual": 0.50,
    "blend_poisson": 0.35,
    "blend_seed": 0.15,
    "fallback_points_for": 70.0,
    "fallback_points_against": 70.0,
    "fallback_seed": 8.5,
    "weights": {
        "recent_offense_diff": 1.20,
        "recent_defense_diff": 1.00,
        "recent_net_rating_diff": 1.50,
        "season_win_pct_diff": 0.90,
        "season_avg_margin_diff": 1.10,
        "seed_diff": 0.80,
        "matchup_attack_vs_defense_diff": 1.40,
        "consistency_diff": -0.25,
        "ci_width_diff": -0.15,
    },
}
