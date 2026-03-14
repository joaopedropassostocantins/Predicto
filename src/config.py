# src/config.py — Predicto v4.0 — March Machine Learning Mania 2026
#
# Central configuration with YAML-based loading and auto-detection.
# Falls back gracefully when PyYAML is not installed.

from __future__ import annotations

import os
from pathlib import Path
from copy import deepcopy

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

_ON_KAGGLE    = os.path.exists("/kaggle/input")
PROJECT_ROOT  = Path(__file__).resolve().parent.parent   # Predicto/
_CONFIG_YAML  = PROJECT_ROOT / "configs" / "default.yaml"

# ---------------------------------------------------------------------------
# Data directory auto-detection
# ---------------------------------------------------------------------------

def _find_data_dir() -> str:
    """Resolve data directory: CONFIG override → Kaggle → local candidates."""
    # 1. Kaggle standard path
    if _ON_KAGGLE:
        for sub in [
            "march-machine-learning-mania-2026",
            "march-machine-learning-mania-2025",
        ]:
            p = f"/kaggle/input/{sub}"
            if os.path.exists(p):
                return p
        return "/kaggle/input/march-machine-learning-mania-2026"

    # 2. Local candidates
    candidates = [
        str(PROJECT_ROOT / "data"),
        os.path.expanduser("~/data/march-machine-learning-mania-2026"),
        os.path.expanduser("~/predicto_local/data"),
        "/home/ubuntu/predicto_local/data",
        "/tmp/predicto_data",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c

    # 3. Fallback — return project/data even if it doesn't exist yet
    return str(PROJECT_ROOT / "data")


# ---------------------------------------------------------------------------
# YAML loader with graceful fallback
# ---------------------------------------------------------------------------

def _load_yaml_config(path: Path) -> dict:
    """Load YAML config; return empty dict if file missing or PyYAML absent."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        # PyYAML not installed — continue with Python dict defaults below
        return {}
    except Exception as e:
        print(f"[config] WARNING: Failed to load {path}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Build CONFIG dict — YAML → Python dict → runtime injection
# ---------------------------------------------------------------------------

def _build_config() -> dict:
    """Build the full CONFIG dict from YAML + runtime overrides."""
    yaml_cfg = _load_yaml_config(_CONFIG_YAML)

    # Python-dict defaults (used when YAML is unavailable).
    # These mirror configs/default.yaml exactly.
    defaults: dict = {
        # ── Competition target ──────────────────────────────────────────────
        "target_season": 2026,
        "backtest_seasons": [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
        "genders": ["M", "W"],

        # ── Feature windows ─────────────────────────────────────────────────
        "recent_games_window": 5,
        "poisson_windows": [3, 5, "season"],
        "poisson_blend_weights": {
            "recent3": 0.40,
            "recent5": 0.35,
            "season":  0.25,
        },

        # ── Poisson ─────────────────────────────────────────────────────────
        "alpha_ci": 0.10,
        "max_points_poisson": 155,
        "poisson_shrinkage_k": 8.0,       # adaptive shrinkage prior k
        "poisson_shrinkage": 0.20,         # legacy fixed shrinkage (fallback)

        # ── Probability clipping ────────────────────────────────────────────
        "pred_win_min": 0.05,
        "pred_win_max": 0.95,
        "pred_clip_min": 0.05,             # legacy alias
        "pred_clip_max": 0.95,             # legacy alias

        # ── Elo ─────────────────────────────────────────────────────────────
        "elo_initial_rating": 1500.0,
        "elo_k_factor": 20.0,              # was 25 — too high
        "elo_carry_factor": 0.82,          # was 0.75 — too low
        "elo_use_margin": True,
        "elo_margin_cap": 15.0,            # was 30 — too high for NCAA basketball

        # ── Temperature scaling ─────────────────────────────────────────────
        # Only values >= 1.0 — never compress probabilities artificially
        "temperature_candidates": [1.00, 1.05, 1.08, 1.12, 1.15, 1.20, 1.25, 1.35, 1.50],
        "manual_temperature": 1.12,

        # ── Calibration ─────────────────────────────────────────────────────
        "calibration_methods": ["identity", "temperature", "platt", "isotonic"],
        "calibration_scorer": "log_loss",  # was Brier — now log_loss (primary metric)

        # ── Fallback values ─────────────────────────────────────────────────
        "fallback_points_for": 70.0,
        "fallback_points_against": 70.0,
        "fallback_seed": 8.5,
        "fallback_elo": 1500.0,

        # ── XGBoost ─────────────────────────────────────────────────────────
        "tabular_model": "xgb_or_hgb",
        "xgb_params": {
            "n_estimators": 800,
            "learning_rate": 0.03,         # was 0.02
            "max_depth": 3,                # was 4 — shallower = less overfit
            "min_child_weight": 6,         # was 5
            "subsample": 0.80,
            "colsample_bytree": 0.70,
            "reg_lambda": 6.0,             # was 2.0 — strong L2 regularisation
            "reg_alpha": 0.5,              # was 0.2
            "gamma": 0.2,                  # was 0 — prevents trivial splits
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
            "early_stopping_rounds": 50,   # NEW: early stopping enabled
        },
        "xgb_val_fraction": 0.15,          # fraction of train for early stopping eval
        "hgb_params": {
            "learning_rate": 0.03,
            "max_depth": 3,
            "max_iter": 600,
            "min_samples_leaf": 20,
            "l2_regularization": 2.0,
            "random_state": 42,
        },

        # ── Blend weights ────────────────────────────────────────────────────
        # Seed REMOVED from blend (was 0.10).
        # Seed is already a feature in XGBoost — including in blend = double-counting.
        # Elo increased: 0.10 → 0.30 (strong stable baseline).
        # XGB reduced: 0.55 → 0.34 (prevent single-model dominance).
        "blend_weights": {
            "elo":     0.30,               # was 0.10
            "poisson": 0.24,               # was 0.20
            "xgb":     0.34,               # was 0.55
            "manual":  0.12,               # was 0.05
        },

        # ── Manual model feature weights ─────────────────────────────────────
        # Hierarchically ordered by signal quality. Normalised at runtime.
        # seed_diff = 0.0 (excluded — seed is a primary XGB feature).
        "manual_feature_weights": {
            # High priority: structural efficiency signals
            "matchup_diff":              1.20,
            "season_margin_diff":        1.10,
            "quality_win_pct_diff":      1.00,
            "elo_diff":                  0.90,
            # Medium priority: form and contextual signals
            "recent3_margin_diff":       0.80,
            "recent5_margin_diff":       0.70,
            "sos_diff":                  0.60,
            "season_win_pct_diff":       0.50,
            # Low priority: noisy or small-sample signals
            "close_game_win_pct_diff":   0.30,
            "blowout_pct_diff":          0.25,
            "consistency_edge":          0.20,
            # Excluded from manual (kept at 0 to prevent accidental use)
            "seed_diff":                 0.00,
        },
        "manual_model_temperature": 8.0,   # sigmoid temperature for manual model

        # ── Feature columns ──────────────────────────────────────────────────
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
            "poisson_total_points",
            "poisson_uncertainty",
            "consistency_edge",
            "season_trajectory_diff",
            "quality_win_pct_diff",
            "blowout_pct_diff",
            "close_game_win_pct_diff",
            "elo_delta_diff",
            "elo_volatility_diff",
            "ewma_margin_diff",
        ],

        # ── Massey Ordinals ──────────────────────────────────────────────────
        "massey_system_priority": ["POM", "SAG", "MOR", "DOL", "WOL", "RTH", "COL"],
    }

    # Merge YAML over defaults (YAML takes precedence)
    merged = deepcopy(defaults)
    for key, val in yaml_cfg.items():
        if key == "data_dir" and val is None:
            continue          # resolve at runtime below
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key].update(val)
        else:
            merged[key] = val

    # ── Runtime injection ────────────────────────────────────────────────────
    # Resolve data_dir now (Kaggle vs local) — cannot be determined at import time
    # in all environments, so we inject it here.
    if not merged.get("data_dir"):
        merged["data_dir"] = _find_data_dir()

    # Ensure legacy aliases are consistent with primary keys
    merged["pred_clip_min"] = merged["pred_win_min"]
    merged["pred_clip_max"] = merged["pred_win_max"]

    return merged


# ---------------------------------------------------------------------------
# Public singleton
# ---------------------------------------------------------------------------

CONFIG: dict = _build_config()


def reload_config(overrides: dict | None = None) -> dict:
    """
    Re-build CONFIG from YAML and apply optional runtime overrides.

    Parameters
    ----------
    overrides : dict, optional
        Key-value pairs that override any YAML or default setting.

    Returns
    -------
    Updated CONFIG dict (also mutates the module-level CONFIG in-place).
    """
    global CONFIG
    CONFIG = _build_config()
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(CONFIG.get(k), dict):
                CONFIG[k].update(v)
            else:
                CONFIG[k] = v
        # Keep aliases consistent
        CONFIG["pred_clip_min"] = CONFIG.get("pred_win_min", CONFIG.get("pred_clip_min", 0.05))
        CONFIG["pred_clip_max"] = CONFIG.get("pred_win_max", CONFIG.get("pred_clip_max", 0.95))
    return CONFIG
