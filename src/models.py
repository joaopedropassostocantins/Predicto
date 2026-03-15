# src/models.py — Probability computation and ensemble blending v4.0
#
# Changes from v3:
#   - XGBoost: early stopping enabled (val fraction = 0.15, patience = 50 rounds)
#   - XGBoost: better hyperparameters (depth=3, lambda=6, gamma=0.2)
#   - Seed removed from ensemble blend (was 0.10)
#     Justification: seed is already a feature in XGBoost.
#     Including it in the blend = double-counting (seed → XGB → blend AND seed → blend).
#     The seed signal reaches the final prediction through p_xgb. Removing from blend
#     also gives more weight to the structurally superior Elo and Poisson components.
#   - New blend: elo=0.30, poisson=0.24, xgb=0.34, manual=0.12  (total=1.00)
#   - Manual model: feature weights normalised; seed_diff weight set to 0.0
#   - compute_manual_probability: normalises weights before computing score
#   - Feature importance: returns top features from XGBoost training
#
# Blend weight rationale:
#   - Elo (0.30): strong, temporally stable signal; no overfitting risk
#   - Poisson (0.24): structural scoring model; provides independent signal
#   - XGBoost (0.34): data-driven; dominant but not overwhelming
#   - Manual (0.12): interpretable heuristics; hedges against XGB overfitting

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap as _shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from src.utils import clip_probs, sigmoid, normalize_blend
from src.config import CONFIG


# ---------------------------------------------------------------------------
# Tabular model (XGBoost primary, HistGBT fallback)
# ---------------------------------------------------------------------------

def train_tabular_model(
    train_df: pd.DataFrame,
    cfg: dict,
    return_importance: bool = False,
) -> tuple | object:
    """
    Train XGBoost (or HistGradientBoosting if XGB unavailable).

    v4 changes:
      - Early stopping enabled: 15% of train held out as eval set (val_fraction).
      - NaN values in feature columns filled with 0 before training.
      - Optional: return feature importance dict alongside model.

    Parameters
    ----------
    train_df : DataFrame   Training matchup rows with ActualLowWin label.
    cfg : dict             CONFIG dict.
    return_importance : bool
        If True, returns (model, importance_dict).

    Returns
    -------
    model or (model, importance_dict) if return_importance=True.
    """
    feature_cols = cfg["feature_cols"]
    available    = [c for c in feature_cols if c in train_df.columns]

    X = train_df[available].fillna(0.0)
    y = train_df["ActualLowWin"].astype(int).values

    if HAS_XGB:
        params = dict(cfg["xgb_params"])
        val_frac = cfg.get("xgb_val_fraction", 0.15)
        early_rounds = params.pop("early_stopping_rounds", 50)

        # Stratified split for early stopping validation
        if len(np.unique(y)) >= 2 and len(X) >= 20:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=val_frac, random_state=42, stratify=y
            )
        else:
            X_tr, X_val, y_tr, y_val = X, X, y, y

        model = XGBClassifier(
            early_stopping_rounds=early_rounds,
            **params,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        if return_importance:
            scores = model.get_booster().get_fscore()
            importance = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            return model, importance

    else:
        # HistGradientBoosting fallback (no early stopping available with this API)
        model = HistGradientBoostingClassifier(**cfg["hgb_params"])
        model.fit(X, y)

    return model


def predict_tabular_proba(model, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Predict class-1 probability and clip to safe range."""
    feature_cols = cfg["feature_cols"]
    available    = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0.0)
    p = model.predict_proba(X)[:, 1]
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def get_feature_importance(model, cfg: dict) -> dict:
    """
    Return feature importance dict from a trained model.

    For XGBoost: uses built-in fscore (gain-based importance).
    For HistGBT: uses feature_importances_ attribute.
    """
    feature_cols = cfg["feature_cols"]
    if HAS_XGB and isinstance(model, XGBClassifier):
        scores = model.get_booster().get_fscore()
        return {k: v for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)}
    else:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            available = [c for c in feature_cols
                         if hasattr(model, "feature_names_in_")
                         and c in model.feature_names_in_]
            return dict(zip(available or feature_cols, imp))
        return {}


def get_shap_importance(model, df: pd.DataFrame, cfg: dict, n_samples: int = 500) -> dict:
    """
    Compute SHAP-based feature importance (mean absolute SHAP value).

    More reliable than fscore for identifying true feature contributions.
    Requires `pip install shap`.

    Parameters
    ----------
    model   : Trained XGBoost or HistGBT model.
    df      : Feature DataFrame (same columns used in training).
    cfg     : CONFIG dict with feature_cols.
    n_samples : Max samples to use for SHAP computation (speed/memory tradeoff).

    Returns
    -------
    dict {feature_name: mean_abs_shap} sorted by importance descending.
    Empty dict if shap not installed or model not compatible.
    """
    if not HAS_SHAP:
        return {}

    feature_cols = cfg["feature_cols"]
    available    = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0.0)

    # Sample for speed
    if len(X) > n_samples:
        X = X.sample(n_samples, random_state=42)

    try:
        if HAS_XGB and isinstance(model, XGBClassifier):
            explainer  = _shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X)
        else:
            explainer  = _shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = dict(zip(available, mean_abs.tolist()))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Auxiliary probability signals
# ---------------------------------------------------------------------------

def compute_elo_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Elo-based win probability: sigmoid(elo_diff / scale).

    Scale=150 maps a 150-point Elo advantage to 73% win probability,
    consistent with the standard Elo logistic formula (1 / (1 + 10^(-d/400))).
    For reference: 400/log(10) ≈ 174; 150 is slightly tighter.
    """
    p = sigmoid(df["elo_diff"].values / 150.0)
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_manual_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Weighted linear combination of features, passed through sigmoid.

    v4 changes:
      - Weights normalised to unit sum before computing score.
        Prevents the temperature parameter from needing re-tuning when
        weights change.
      - seed_diff weight is 0.0 (see config): prevents double-counting.
      - Only features present in `df` are used (missing features silently skipped).

    v4.2 additions:
      - manual_model_enabled (bool, default True): if False, returns 0.5 for all rows.
        Useful to ablate the manual component without changing blend weights.
      - manual_contribution_cap (float, default None): if set, clips output to
        [0.5 - cap, 0.5 + cap] before returning. Prevents extreme manual predictions
        from dominating the blend even at the 12% weight level.
        Example: cap=0.45 → manual output stays in [0.05, 0.95].

    Parameters
    ----------
    cfg["manual_feature_weights"] : dict of {feature_name: weight}
    cfg["manual_model_temperature"] : float, sigmoid temperature (default 8.0)
    cfg["manual_model_enabled"] : bool (default True)
    cfg["manual_contribution_cap"] : float or None (default None)
    """
    # Check if manual model is enabled; return neutral 0.5 if disabled.
    if not cfg.get("manual_model_enabled", True):
        return np.full(len(df), 0.5, dtype=float)

    raw_weights = cfg.get("manual_feature_weights", {})

    # Normalise weights (excludes zeros to avoid diluting active signals)
    active = {k: v for k, v in raw_weights.items() if v > 0 and k in df.columns}
    total_w = sum(active.values()) if active else 1.0

    score = np.zeros(len(df), dtype=float)
    for feat, w in active.items():
        score += (w / total_w) * df[feat].fillna(0.0).values

    temperature = cfg.get("manual_model_temperature", 8.0)
    p = sigmoid(score / temperature)

    # Optional contribution cap: clip manual output to [0.5 - cap, 0.5 + cap].
    # This limits how extreme the manual component can be, preventing any single
    # heuristic from dominating even with the 12% blend weight.
    cap = cfg.get("manual_contribution_cap", None)
    if cap is not None and cap > 0:
        p = np.clip(p, 0.5 - float(cap), 0.5 + float(cap))

    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


def compute_poisson_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Direct Poisson win probability from the matchup distribution."""
    return clip_probs(
        df["poisson_win_prob"].values,
        cfg["pred_clip_min"],
        cfg["pred_clip_max"],
    )


# Kept for backward compatibility (seed as separate probability signal).
# NOT used in the v4 blend — seed is handled through XGBoost features.
def compute_seed_probability(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """sigmoid(seed_diff / 3) — simple seed-based probability (legacy; not in blend)."""
    p = sigmoid(df["seed_diff"].values / 3.0)
    return clip_probs(p, cfg["pred_clip_min"], cfg["pred_clip_max"])


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def blend_predictions(p_dict: dict, cfg: dict) -> np.ndarray:
    """
    Weighted average of component probabilities.

    Only keys present in both p_dict and cfg["blend_weights"] are used.
    Weights are renormalised after filtering to always sum to 1.
    """
    w_raw = {k: v for k, v in cfg["blend_weights"].items() if k in p_dict}
    if not w_raw:
        # Fallback: equal weight over all provided components
        w_raw = {k: 1.0 for k in p_dict}
    w    = normalize_blend(w_raw)
    n    = len(next(iter(p_dict.values())))
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
    train_df: Optional[pd.DataFrame] = None,
    return_model: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, object]:
    """
    Compute all probability signals and produce a final blended prediction.

    Signals computed:
        p_elo      : Elo-based probability
        p_poisson  : Poisson scoring model probability
        p_manual   : Weighted linear heuristic probability
        p_xgb      : XGBoost probability (trained on train_df if provided)
        Pred       : Blended final probability

    v4 changes:
        - p_seed still computed for reference but NOT included in blend.
        - XGBoost now uses early stopping internally.
        - Blend: elo=0.30, poisson=0.24, xgb=0.34, manual=0.12.

    Parameters
    ----------
    df : DataFrame       Matchup rows with full feature set.
    train_df : DataFrame, optional
        Historical matchup rows WITH ActualLowWin labels.
        If provided and >= 20 rows, XGBoost is trained on it.
    return_model : bool
        If True, returns (df, trained_model) for later inspection.
    """
    out = df.copy()

    out["p_elo"]     = compute_elo_probability(out, cfg)
    out["p_poisson"] = compute_poisson_probability(out, cfg)
    out["p_manual"]  = compute_manual_probability(out, cfg)
    out["p_seed"]    = compute_seed_probability(out, cfg)  # reference only

    trained_model = None
    if train_df is not None and len(train_df) >= 20:
        trained_model = train_tabular_model(train_df, cfg)
        out["p_xgb"]  = predict_tabular_proba(trained_model, out, cfg)
    else:
        # Fallback: auxiliary blend of elo + poisson when no training data
        aux = {
            "poisson": out["p_poisson"].values,
            "elo":     out["p_elo"].values,
            "manual":  out["p_manual"].values,
        }
        aux_cfg = {
            **cfg,
            "blend_weights": {"poisson": 0.45, "elo": 0.40, "manual": 0.15},
        }
        out["p_xgb"] = blend_predictions(aux, aux_cfg)

    # Final blend: elo + poisson + xgb + manual  (no seed in blend)
    out["Pred"] = blend_predictions(
        {
            "elo":     out["p_elo"].values,
            "poisson": out["p_poisson"].values,
            "xgb":     out["p_xgb"].values,
            "manual":  out["p_manual"].values,
        },
        cfg,
    )

    if return_model:
        return out, trained_model
    return out


# ---------------------------------------------------------------------------
# Blend sensitivity report
# ---------------------------------------------------------------------------

def blend_sensitivity_report(
    df: pd.DataFrame,
    cfg: dict,
    n_steps: int = 5,
) -> pd.DataFrame:
    """
    Compute Log Loss for a grid of blend weight combinations.

    Returns a DataFrame showing how Log Loss changes as each component
    weight is varied ±25% from its default value (others held constant).

    Requires 'ActualLowWin' column in df.
    Useful for understanding sensitivity and identifying dominant components.

    Parameters
    ----------
    df : DataFrame  Must contain p_elo, p_poisson, p_xgb, p_manual, ActualLowWin.
    cfg : dict      CONFIG dict with blend_weights.
    n_steps : int   Number of steps in each direction (total grid size ~n_steps²).
    """
    from sklearn.metrics import log_loss as sk_logloss

    if "ActualLowWin" not in df.columns:
        raise ValueError("blend_sensitivity_report requires 'ActualLowWin' column")

    y = df["ActualLowWin"].values.astype(int)
    default_w = cfg["blend_weights"].copy()
    components = [k for k in default_w if f"p_{k}" in df.columns]

    rows = []
    for comp in components:
        default = default_w[comp]
        # Vary this component's weight ±25% relative to default
        test_weights = np.linspace(
            max(default * 0.5, 0.05),
            min(default * 1.5, 0.55),
            n_steps,
        )
        for w_test in test_weights:
            # Build weight dict with this component at w_test, others renormalised
            w = dict(default_w)
            w[comp] = w_test
            # Renormalise others proportionally
            other_sum = sum(v for k, v in default_w.items() if k != comp)
            if other_sum > 0:
                scale = (1.0 - w_test) / other_sum
                for k in w:
                    if k != comp:
                        w[k] = default_w[k] * scale

            # Compute blended probability
            p_dict = {k: df[f"p_{k}"].values for k in components}
            test_cfg = {**cfg, "blend_weights": w}
            pred = blend_predictions(p_dict, test_cfg)
            ll = float(sk_logloss(y, pred))
            rows.append({
                "component": comp,
                "weight": round(w_test, 4),
                "log_loss": round(ll, 6),
                "delta_from_default": round(w_test - default, 4),
            })

    return pd.DataFrame(rows).sort_values(["component", "weight"]).reset_index(drop=True)
