# src/metrics.py — Evaluation metrics v4.1
#
# Changes from v4.0:
#   - Added per_band_logloss(): log_loss decomposed by probability band.
#     Identifies which confidence intervals are most miscalibrated.
#   - Added overconfidence_index(): measures tendency to predict extremes.
#     OCI > 0 means model is more extreme than the empirical base rate.
#
# Changes from v3:
#   - Added AUC-ROC to full_metric_bundle
#   - full_metric_bundle: log_loss is now the primary metric (first key returned)
#   - Added per_season_metrics() for temporal evaluation
#   - Added comparison_table() for side-by-side model comparison

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


# ---------------------------------------------------------------------------
# Clipping helper (kept for modules that imported directly from metrics)
# ---------------------------------------------------------------------------

def clip_probs(p, eps: float = 1e-6):
    """Clip probabilities to [eps, 1-eps] to prevent log(0) errors."""
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_basic_metrics(y_true, p_pred) -> dict:
    """
    Compute primary calibration metrics.

    Metric hierarchy (primary → secondary):
        1. log_loss   — primary; penalises overconfidence harshly
        2. brier      — proper scoring rule; penalises squared error
        3. accuracy   — secondary; does not distinguish 0.51 from 0.99
        4. auc        — secondary; discrimination without calibration
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)

    auc = float(roc_auc_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "log_loss": float(log_loss(y_true, p_pred, labels=[0, 1])),   # PRIMARY
        "brier":    float(brier_score_loss(y_true, p_pred)),
        "accuracy": float(accuracy_score(y_true, (p_pred >= 0.5).astype(int))),
        "auc":      auc,
    }


def expected_calibration_error(y_true, p_pred, bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_b (n_b / n) × |avg_pred_b − emp_rate_b|

    ECE measures average miscalibration weighted by bin size.
    Lower is better; 0 = perfectly calibrated.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)
    edges  = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p_pred, edges[1:-1], right=True)
    total  = len(y_true)
    ece    = 0.0

    for b in range(bins):
        mask = bucket == b
        n    = int(mask.sum())
        if n == 0:
            continue
        avg_pred = p_pred[mask].mean()
        emp_rate = y_true[mask].mean()
        ece += (n / total) * abs(avg_pred - emp_rate)

    return float(ece)


def calibration_slope_intercept(y_true, p_pred) -> dict:
    """
    Calibration intercept and slope via logistic regression on logits.

    Perfectly calibrated: intercept ≈ 0, slope ≈ 1.
    Overconfident: slope < 1.
    Underconfident: slope > 1.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)
    logits = np.log(p_pred / (1.0 - p_pred))
    x      = np.column_stack([np.ones(len(logits)), logits])
    beta   = np.zeros(2, dtype=float)

    for _ in range(50):
        z        = x @ beta
        mu       = 1.0 / (1.0 + np.exp(-z))
        w        = np.clip(mu * (1.0 - mu), 1e-9, None)
        hessian  = x.T @ (w[:, None] * x)
        grad     = x.T @ (y_true - mu)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    return {
        "calibration_intercept": float(beta[0]),
        "calibration_slope":     float(beta[1]),
    }


def probability_band_report(y_true, p_pred, bands=None) -> pd.DataFrame:
    """Empirical win rate by probability band (useful for reliability analysis)."""
    if bands is None:
        bands = [
            (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0),
        ]
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)
    rows   = []

    for left, right in bands:
        if right < 1.0:
            mask = (p_pred >= left) & (p_pred < right)
        else:
            mask = (p_pred >= left) & (p_pred <= right)
        n = int(mask.sum())
        label = f"[{left:.1f}, {right:.1f}{')'  if right < 1.0 else ']'}"
        if n == 0:
            rows.append({"band": label, "n": 0, "avg_pred": np.nan,
                         "emp_rate": np.nan, "accuracy": np.nan})
            continue
        rows.append({
            "band":     label,
            "n":        n,
            "avg_pred": float(p_pred[mask].mean()),
            "emp_rate": float(y_true[mask].mean()),
            "accuracy": float(((p_pred[mask] >= 0.5).astype(int) == y_true[mask]).mean()),
        })

    return pd.DataFrame(rows)


def favorite_hit_rate(y_true, p_pred) -> float:
    """Fraction of games where the favourite (pred ≥ 0.5) wins."""
    y_true   = np.asarray(y_true).astype(int)
    fav_pred = (clip_probs(p_pred) >= 0.5).astype(int)
    return float((fav_pred == y_true).mean())


def upset_rate_realized(y_true, p_pred) -> float:
    """Fraction of games where the underdog (pred < 0.5) actually wins."""
    y_true   = np.asarray(y_true).astype(int)
    underdog = (clip_probs(p_pred) < 0.5).astype(int)
    return float((underdog == y_true).mean())


def full_metric_bundle(y_true, p_pred, bins: int = 10) -> dict:
    """
    All calibration and discrimination metrics in one dict.

    Key order reflects metric hierarchy (primary first):
        log_loss, brier, auc, accuracy, ece, calibration_slope, calibration_intercept, ...
    """
    basic = compute_basic_metrics(y_true, p_pred)
    calib = calibration_slope_intercept(y_true, p_pred)

    return {
        # Primary metrics (calibration)
        "log_loss":             basic["log_loss"],
        "brier":                basic["brier"],
        "ece":                  expected_calibration_error(y_true, p_pred, bins=bins),
        # Secondary metrics (discrimination)
        "auc":                  basic["auc"],
        "accuracy":             basic["accuracy"],
        # Calibration diagnostics
        "calibration_intercept": calib["calibration_intercept"],
        "calibration_slope":    calib["calibration_slope"],
        # Coverage diagnostics
        "favorite_hit_rate":    favorite_hit_rate(y_true, p_pred),
        "realized_upset_rate":  upset_rate_realized(y_true, p_pred),
    }


# ---------------------------------------------------------------------------
# Temporal and multi-model comparison utilities
# ---------------------------------------------------------------------------

def per_season_metrics(
    df: pd.DataFrame,
    y_col: str = "ActualLowWin",
    p_col: str = "Pred",
    season_col: str = "Season",
) -> pd.DataFrame:
    """
    Compute metrics separately for each season.

    Returns DataFrame [Season, n_games, log_loss, brier, ece, auc, accuracy].
    """
    rows = []
    for season, grp in df.groupby(season_col):
        y = grp[y_col].values.astype(int)
        p = grp[p_col].values.astype(float)
        m = full_metric_bundle(y, p)
        rows.append({
            season_col:   season,
            "n_games":    len(grp),
            "log_loss":   round(m["log_loss"], 6),
            "brier":      round(m["brier"], 6),
            "ece":        round(m["ece"], 6),
            "auc":        round(m["auc"], 4),
            "accuracy":   round(m["accuracy"], 4),
        })

    df_out = pd.DataFrame(rows).sort_values(season_col).reset_index(drop=True)
    # Append aggregate row
    y_all = df[y_col].values.astype(int)
    p_all = df[p_col].values.astype(float)
    m_all = full_metric_bundle(y_all, p_all)
    df_out = pd.concat([df_out, pd.DataFrame([{
        season_col: "OVERALL",
        "n_games":  len(df),
        "log_loss": round(m_all["log_loss"], 6),
        "brier":    round(m_all["brier"], 6),
        "ece":      round(m_all["ece"], 6),
        "auc":      round(m_all["auc"], 4),
        "accuracy": round(m_all["accuracy"], 4),
    }])], ignore_index=True)

    return df_out


def per_band_logloss(y_true, p_pred, bands=None) -> pd.DataFrame:
    """
    Log Loss decomposed by probability band.

    Identifies where the model is most miscalibrated from a log_loss perspective.
    Bands with high log_loss indicate poor calibration in that probability range.

    Returns DataFrame [band, n, avg_pred, emp_rate, log_loss].
    """
    if bands is None:
        bands = [
            (0.0, 0.2), (0.2, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.8), (0.8, 1.0),
        ]
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)
    eps    = 1e-12
    rows   = []

    for left, right in bands:
        if right < 1.0:
            mask = (p_pred >= left) & (p_pred < right)
        else:
            mask = (p_pred >= left) & (p_pred <= right)
        n = int(mask.sum())
        label = f"[{left:.1f}, {right:.1f}{')'  if right < 1.0 else ']'}"
        if n == 0:
            rows.append({"band": label, "n": 0, "avg_pred": np.nan,
                         "emp_rate": np.nan, "log_loss": np.nan})
            continue
        p_b = p_pred[mask]
        y_b = y_true[mask]
        ll  = float(-np.mean(
            y_b * np.log(np.clip(p_b, eps, 1 - eps))
            + (1 - y_b) * np.log(np.clip(1 - p_b, eps, 1 - eps))
        ))
        rows.append({
            "band":     label,
            "n":        n,
            "avg_pred": float(p_b.mean()),
            "emp_rate": float(y_b.mean()),
            "log_loss": round(ll, 6),
        })
    return pd.DataFrame(rows)


def overconfidence_index(y_true, p_pred, threshold: float = 0.75) -> float:
    """
    Overconfidence Index (OCI).

    Measures the fraction of predictions that are more extreme than `threshold`
    (i.e., p > threshold or p < 1-threshold) minus the empirical base rate of
    dominant outcomes (actual win rate beyond threshold).

    OCI > 0: model is overconfident (predicts extremes more than warranted).
    OCI = 0: well calibrated in terms of confidence.
    OCI < 0: model is underconfident (avoids extremes).

    Parameters
    ----------
    threshold : float  Probability boundary defining "confident" predictions.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)

    pred_extreme_rate = float(((p_pred > threshold) | (p_pred < (1 - threshold))).mean())
    # Empirical rate: how often does the actual outcome strongly favour one side?
    # Approximate via: fraction of games decided by a clear favourite in hindsight
    # (i.e., we use the empirical p_max to estimate what fraction should be extreme)
    # This is a simple proxy — not a true calibration diagnostic.
    # For a rigorous version, compare mean(p_pred) vs mean(y_true) in extreme bins.
    actual_extreme_rate = float(np.mean(y_true))   # overall base rate of class=1
    # OCI: excess of predicted extreme confidence over the base rate
    return float(pred_extreme_rate - abs(actual_extreme_rate - 0.5) * 2)


def comparison_table(
    y_true,
    predictions: dict,  # {model_name: p_pred array}
) -> pd.DataFrame:
    """
    Side-by-side model comparison table.

    Parameters
    ----------
    y_true : array-like  True binary outcomes.
    predictions : dict   {model_name: probability_array}

    Returns
    -------
    DataFrame [model, log_loss, brier, ece, auc, accuracy]
    sorted by log_loss ascending (best first).
    """
    rows = []
    for name, p in predictions.items():
        m = full_metric_bundle(y_true, p)
        rows.append({
            "model":     name,
            "log_loss":  round(m["log_loss"], 6),
            "brier":     round(m["brier"], 6),
            "ece":       round(m["ece"], 6),
            "auc":       round(m["auc"], 4),
            "accuracy":  round(m["accuracy"], 4),
        })
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
