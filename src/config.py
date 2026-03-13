
CONFIG = {
    "data_dir": "/kaggle/input/competitions/march-machine-learning-mania-2026",
    "target_season": 2026,
    "recent_games_window": 3,
    "alpha_ci": 0.10,
    "max_points_poisson": 220,
    "pred_clip_min": 0.02,
    "pred_clip_max": 0.98,
    "fallback_points_for": 70.0,
    "fallback_points_against": 70.0,
    "fallback_seed": 8.5,
    "poisson_windows": [3, 5, "season"], # New: multiple windows for Poisson
    "poisson_blend_weights": {"recent3": 0.4, "recent5": 0.3, "season": 0.3}, # New: Poisson blend weights
    "backtest_seasons": [2022, 2023, 2024, 2025],
    "calibration_methods": ["identity", "platt", "isotonic"],
    "elo_k_factor": 20, # New: Elo K-factor
    "elo_initial_rating": 1500, # New: Elo initial rating
}
