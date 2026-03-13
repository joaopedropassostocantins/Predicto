# src/models.py — Probability computation and ensemble blending — v3.0

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from src.utils import clip_probs, sigmoid, normalize_blend
from src.config import CONFIG


# ---------------------------------------------------------------------------
# Tabular model (XGBoost primary, HistGBT fallback)
# ---------------------------------------------------------------------------

def train_tabular_model(train_df: pd.DataFrame, cfg: dict):
    """
    Train XGBoost (or HistGradientBoosting if XGB unavailable).

    NaN values in feature columns are filled with 0 before training.
    """
    feature_cols = cfg["feature_cols"]
    # Only use columns that are actually present in the training frame
    available = [c for c in feature_cols if c in train_df.columns]

    X = train_df[available].fillna(0.0)
    y = train_df["ActualLowWin"].astype(int).values

    if HAS_XGB:
        # Remove any params that require eval_set at fit time
        params = {k: v for k, v in cfg["xgb_params"].items()
                  if k != "early_stopping_rounds"}
        model = XGBClassifier(**params)
    else:
        model = HistGradientBoostingClassifier(**cfg["hgb_params"])

    model.fit(X, y)
    return model


def predict_tabular_proba(model, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    feature_cols = cfg["feature_cols"]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0.0)
    p = model.predict_proba(X)[:, 1]
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


# ---------------------------------------------------------------------------
# Auxiliary probability signals
# ---------------------------------------------------------------------------

def compute_seed_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """sigmoid(seed_diff / 3) — simple seed-based probability."""
    p = sigmoid(df["seed_diff"].values / 3.0)
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_elo_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """sigmoid(elo_diff / 150) — Elo-based probability."""
    p = sigmoid(df["elo_diff"].values / 150.0)
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_manual_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Weighted linear combination of features, passed through sigmoid.

    Only features present in `df` are used.
    """
    score = np.zeros(len(df), dtype=float)
    for feat, w in cfg["manual_feature_weights"].items():
        if feat in df.columns:
            score += w * df[feat].fillna(0.0).values
    p = sigmoid(score / cfg["manual_temperature"])
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_poisson_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Direct Poisson win probability from the matchup distribution."""
    return clip_probs(df["poisson_win_prob"].values,
                      cfg["pred_clip_min"], cfg["pred_clip_max"])


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def blend_predictions(p_dict: dict, cfg: dict) -> np.ndarray:
    """
    Weighted average of component probabilities.

    Only keys present in both p_dict and cfg["blend_weights"] are used.
    The weights are renormalised after filtering.
    """
    w_raw = {k: v for k, v in cfg["blend_weights"].items() if k in p_dict}
    w = normalize_blend(w_raw)
    n = len(next(iter(p_dict.values())))
    pred = np.zeros(n, dtype=float)
    for key, arr in p_dict.items():
        if key in w:
            pred += w[key] * np.asarray(arr, dtype=float)
    return clip_probs(pred, cfg["pred_clip_min"], cfg["pred_clip_max"])


# ---------------------------------------------------------------------------
# Orchestration: compute all signals and final blend
# ---------------------------------------------------------------------------

def compute_all_probabilities(
    df: pd.DataFrame,
    cfg: dict,
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all probability signals and produce a final blended prediction.

    Parameters
    ----------
    df : DataFrame
        Matchup rows with full feature set.
    train_df : DataFrame, optional
        Historical matchup rows WITH ActualLowWin labels.
        If provided and large enough, XGBoost is trained on it and used
        as the primary signal.  Otherwise the auxiliary blend is used as
        the XGBoost proxy.

    Adds columns: p_poisson, p_seed, p_elo, p_manual, p_xgb, Pred.
    """
    out = df.copy()

    out["p_poisson"] = compute_poisson_probability(out, cfg)
    out["p_seed"]    = compute_seed_probability(out, cfg)
    out["p_elo"]     = compute_elo_probability(out, cfg)
    out["p_manual"]  = compute_manual_probability(out, cfg)

    if train_df is not None and len(train_df) >= 20:
        model = train_tabular_model(train_df, cfg)
        out["p_xgb"] = predict_tabular_proba(model, out, cfg)
    else:
        # No labelled training data: use auxiliary blend as XGBoost stand-in.
        # Note: this is only a fallback; real XGBoost should always be used
        # when training data is available.
        aux = {
            "poisson": out["p_poisson"].values,
            "seed":    out["p_seed"].values,
            "elo":     out["p_elo"].values,
            "manual":  out["p_manual"].values,
        }
        aux_cfg = {
            **cfg,
            "blend_weights": {
                "poisson": 0.35,
                "seed":    0.20,
                "elo":     0.25,
                "manual":  0.20,
            },
        }
        out["p_xgb"] = blend_predictions(aux, aux_cfg)

    out["Pred"] = blend_predictions(
        {
            "xgb":     out["p_xgb"].values,
            "poisson": out["p_poisson"].values,
            "seed":    out["p_seed"].values,
            "elo":     out["p_elo"].values,
            "manual":  out["p_manual"].values,
        },
        cfg,
    )

    return out
