# src/models.py

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


def train_tabular_model(train_df: pd.DataFrame, cfg: dict):
    X = train_df[cfg["feature_cols"]].copy()
    y = train_df["ActualLowWin"].astype(int).values

    if HAS_XGB:
        model = XGBClassifier(**cfg["xgb_params"])
    else:
        model = HistGradientBoostingClassifier(**cfg["hgb_params"])

    model.fit(X, y)
    return model


def predict_tabular_proba(model, df: pd.DataFrame, cfg: dict):
    X = df[cfg["feature_cols"]].copy()
    p = model.predict_proba(X)[:, 1]
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_seed_probability(df: pd.DataFrame, cfg: dict):
    p = sigmoid(df["seed_diff"].values / 3.0)
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_elo_probability(df: pd.DataFrame, cfg: dict):
    p = sigmoid(df["elo_diff"].values / 150.0)
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_manual_probability(df: pd.DataFrame, cfg: dict):
    score = np.zeros(len(df), dtype=float)
    for feat, w in cfg["manual_feature_weights"].items():
        if feat in df.columns:
            score += w * df[feat].values
    p = sigmoid(score / cfg["manual_temperature"])
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_poisson_probability(df: pd.DataFrame, cfg: dict):
    return clip_probs(df["poisson_win_prob"].values, cfg["pred_clip_min"], cfg["pred_clip_max"])


def blend_predictions(p_dict: dict, cfg: dict):
    w = normalize_blend(cfg["blend_weights"])
    pred = np.zeros(len(next(iter(p_dict.values()))), dtype=float)
    for key, arr in p_dict.items():
        pred += w[key] * arr
    return clip_probs(pred, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_all_probabilities(df: pd.DataFrame, cfg: dict, train_df: pd.DataFrame | None = None):
    out = df.copy()

    out["p_poisson"] = compute_poisson_probability(out, cfg)
    out["p_seed"] = compute_seed_probability(out, cfg)
    out["p_elo"] = compute_elo_probability(out, cfg)
    out["p_manual"] = compute_manual_probability(out, cfg)

    if train_df is not None and len(train_df) > 10:
        model = train_tabular_model(train_df, cfg)
        out["p_xgb"] = predict_tabular_proba(model, out, cfg)
    else:
        out["p_xgb"] = blend_predictions(
            {
                "xgb": out["p_manual"].values,
                "poisson": out["p_poisson"].values,
                "seed": out["p_seed"].values,
                "elo": out["p_elo"].values,
                "manual": out["p_manual"].values,
            },
            {
                **cfg,
                "blend_weights": {
                    "xgb": 0.45,
                    "poisson": 0.25,
                    "seed": 0.10,
                    "elo": 0.10,
                    "manual": 0.10,
                },
            },
        )

    out["Pred"] = blend_predictions(
        {
            "xgb": out["p_xgb"].values,
            "poisson": out["p_poisson"].values,
            "seed": out["p_seed"].values,
            "elo": out["p_elo"].values,
            "manual": out["p_manual"].values,
        },
        cfg,
    )

    return out
