from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def clip_probs(p, eps: float = 1e-6):
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def compute_basic_metrics(y_true, p_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)

    return {
        "brier": float(brier_score_loss(y_true, p_pred)),
        "accuracy": float(accuracy_score(y_true, (p_pred >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_true, p_pred, labels=[0, 1])),
    }


def expected_calibration_error(y_true, p_pred, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p_pred, edges[1:-1], right=True)

    total = len(y_true)
    ece = 0.0

    for b in range(bins):
        mask = bucket == b
        n = int(mask.sum())
        if n == 0:
            continue

        avg_pred = p_pred[mask].mean()
        emp_rate = y_true[mask].mean()
        ece += (n / total) * abs(avg_pred - emp_rate)

    return float(ece)


def calibration_slope_intercept(y_true, p_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)

    logits = np.log(p_pred / (1.0 - p_pred))
    x = np.column_stack([np.ones(len(logits)), logits])

    beta = np.zeros(2, dtype=float)

    for _ in range(50):
        z = x @ beta
        mu = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)

        hessian = x.T @ (w[:, None] * x)
        grad = x.T @ (y_true - mu)

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
        "calibration_slope": float(beta[1]),
    }


def probability_band_report(y_true, p_pred, bands=None) -> pd.DataFrame:
    if bands is None:
        bands = [
            (0.0, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 0.8),
            (0.8, 0.9),
            (0.9, 1.0),
        ]

    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)

    rows = []
    for left, right in bands:
        if right < 1.0:
            mask = (p_pred >= left) & (p_pred < right)
        else:
            mask = (p_pred >= left) & (p_pred <= right)

        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "band": f"[{left:.1f}, {right:.1f}{')' if right < 1.0 else ']'}",
                    "n": 0,
                    "avg_pred": np.nan,
                    "emp_rate": np.nan,
                    "accuracy": np.nan,
                }
            )
            continue

        rows.append(
            {
                "band": f"[{left:.1f}, {right:.1f}{')' if right < 1.0 else ']'}",
                "n": n,
                "avg_pred": float(p_pred[mask].mean()),
                "emp_rate": float(y_true[mask].mean()),
                "accuracy": float(((p_pred[mask] >= 0.5).astype(int) == y_true[mask]).mean()),
            }
        )

    return pd.DataFrame(rows)


def favorite_hit_rate(y_true, p_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)
    fav_pred = (p_pred >= 0.5).astype(int)
    return float((fav_pred == y_true).mean())


def upset_rate_realized(y_true, p_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    p_pred = clip_probs(p_pred)
    underdog = (p_pred < 0.5).astype(int)
    return float((underdog == y_true).mean())


def full_metric_bundle(y_true, p_pred, bins: int = 10) -> dict:
    basic = compute_basic_metrics(y_true, p_pred)
    calib = calibration_slope_intercept(y_true, p_pred)

    return {
        **basic,
        **calib,
        "ece": expected_calibration_error(y_true, p_pred, bins=bins),
        "favorite_hit_rate": favorite_hit_rate(y_true, p_pred),
        "realized_upset_rate": upset_rate_realized(y_true, p_pred),
    }
