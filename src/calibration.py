# src/calibration.py — Probability calibration v4.0
#
# Changes from v3:
#   - choose_best_calibrator: selection criterion changed from Brier → Log Loss.
#     Rationale: log_loss is more sensitive to calibration quality, especially
#     at extremes. A 90% prediction that is wrong hurts more than a 55% miss.
#     Literature: Guo et al. (2017) show log_loss > Brier for detecting miscalibration.
#   - TemperatureCalibrator: removed candidates < 1.0.
#     Temperatures < 1.0 increase confidence (logit / 0.8 > logit), which is
#     almost never appropriate for sports prediction where overconfidence is common.
#   - temperature_candidates default now mirrors configs/default.yaml.
#   - Added: reliability_plot_data() for visual calibration audit.
#   - Added: calibration_audit_report() — one-call summary of all calibration metrics.
#
# References:
#   Platt (1999) "Probabilistic outputs for SVMs and comparisons"
#   Niculescu-Mizil & Caruana (2005) "Predicting good probabilities with supervised learning"
#   Guo et al. (2017) "On Calibration of Modern Neural Networks"

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils import clip_probs, logit, sigmoid


@dataclass
class CalibrationResult:
    method:  str
    fitted:  object
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Calibrator classes
# ---------------------------------------------------------------------------

class IdentityCalibrator:
    """No-op calibrator. Returns clipped input probabilities unchanged."""
    def fit(self, p, y):
        return self

    def predict(self, p):
        return clip_probs(p)


class TemperatureCalibrator:
    """
    Temperature scaling: p_cal = sigmoid(logit(p) / T).

    T > 1.0 → pushes probabilities toward 0.5 (reduces overconfidence).
    T = 1.0 → identity.
    T < 1.0 → NEVER used in v4 (would increase confidence, counterproductive).

    Selects optimal T from candidates via log loss minimisation on training set.
    """
    def __init__(self, temperatures=None):
        # Default: all values ≥ 1.0 (never increase confidence)
        self.temperatures    = temperatures or [1.00, 1.05, 1.08, 1.12, 1.15, 1.20, 1.25, 1.35, 1.50]
        self.best_temperature = 1.0

    def fit(self, p, y):
        p = clip_probs(p)
        y = np.asarray(y).astype(int)
        eps        = 1e-12
        best_t     = 1.0
        best_loss  = float("inf")
        base_logit = logit(p)

        for t in self.temperatures:
            p_t  = sigmoid(base_logit / t)
            loss = -np.mean(
                y * np.log(np.clip(p_t, eps, 1 - eps))
                + (1 - y) * np.log(np.clip(1 - p_t, eps, 1 - eps))
            )
            if loss < best_loss:
                best_loss = loss
                best_t    = t

        self.best_temperature = best_t
        return self

    def predict(self, p):
        p = clip_probs(p)
        return clip_probs(sigmoid(logit(p) / self.best_temperature))


class PlattCalibrator:
    """
    Platt scaling: logistic regression on logit(p).

    Learns an affine transformation A × logit(p) + B that maps raw logits
    to calibrated probabilities. More flexible than temperature scaling.
    """
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs", C=1e4)  # weak regularisation

    def fit(self, p, y):
        x = logit(clip_probs(p)).reshape(-1, 1)
        y = np.asarray(y).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p):
        x = logit(clip_probs(p)).reshape(-1, 1)
        return clip_probs(self.model.predict_proba(x)[:, 1])


class IsotonicCalibrator:
    """
    Isotonic regression calibration (non-parametric, monotone).

    Most flexible calibrator; risk of overfitting with small datasets.
    Use only when training set > ~500 samples.
    """
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p, y):
        x = clip_probs(p)
        y = np.asarray(y).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p):
        return clip_probs(self.model.predict(clip_probs(p)))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def fit_calibrator(method: str, p_train, y_train, cfg=None):
    """Fit and return a calibrator by name."""
    method = method.lower().strip()
    if method == "identity":
        calibrator = IdentityCalibrator()
    elif method == "temperature":
        # Use config temperature candidates if provided (must all be >= 1.0)
        if cfg is not None:
            temps = [t for t in cfg.get("temperature_candidates", []) if t >= 1.0]
        else:
            temps = None
        calibrator = TemperatureCalibrator(temperatures=temps or None)
    elif method == "platt":
        calibrator = PlattCalibrator()
    elif method == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return calibrator.fit(p_train, y_train)


def apply_calibrator(calibrator, p):
    """Apply a fitted calibrator and clip output to safe range."""
    return clip_probs(calibrator.predict(np.asarray(p, dtype=float)))


def choose_best_calibrator(
    p_train, y_train,
    p_valid, y_valid,
    scorer_fn,
    methods: Optional[List[str]] = None,
    cfg=None,
) -> CalibrationResult:
    """
    Fit all calibrators on p_train/y_train, evaluate on p_valid/y_valid.

    v4 CHANGE: Selects best calibrator by LOG LOSS (not Brier score).

    Primary criterion: log_loss (most sensitive to calibration quality).
    Tie-break: brier_score (second criterion).

    Parameters
    ----------
    scorer_fn : callable  (y_true, p_pred) → dict with 'log_loss', 'brier' keys.
    """
    if methods is None:
        methods = ["identity", "temperature", "platt", "isotonic"]

    best_result: Optional[CalibrationResult] = None

    for method in methods:
        try:
            calibrator = fit_calibrator(method, p_train, y_train, cfg=cfg)
            p_cal      = apply_calibrator(calibrator, p_valid)
            metrics    = scorer_fn(y_valid, p_cal)
            candidate  = CalibrationResult(method=method, fitted=calibrator, metrics=metrics)

            if best_result is None:
                best_result = candidate
            else:
                # Primary: lower log_loss wins
                if metrics.get("log_loss", 1.0) < best_result.metrics.get("log_loss", 1.0):
                    best_result = candidate
                # Tie-break: lower brier wins
                elif (metrics.get("log_loss", 1.0) == best_result.metrics.get("log_loss", 1.0)
                      and metrics.get("brier", 1.0) < best_result.metrics.get("brier", 1.0)):
                    best_result = candidate

        except Exception as e:
            print(f"  [calibration] WARNING: {method} failed — {e}. Skipping.")

    if best_result is None:
        # Ultimate fallback
        cal = IdentityCalibrator().fit(p_valid, y_valid)
        best_result = CalibrationResult(
            method="identity", fitted=cal,
            metrics=scorer_fn(y_valid, clip_probs(p_valid)),
        )

    return best_result


# ---------------------------------------------------------------------------
# Calibration analysis utilities
# ---------------------------------------------------------------------------

def calibration_table(y_true, p_pred, bins: int = 10) -> pd.DataFrame:
    """Binned calibration table: predicted vs empirical win rate."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(clip_probs(p_pred))
    edges  = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p_pred, edges[1:-1], right=True)

    rows = []
    for b in range(bins):
        mask = bucket == b
        n    = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": b, "bin_left": edges[b], "bin_right": edges[b + 1],
                "n": 0, "avg_pred": np.nan, "emp_rate": np.nan, "abs_gap": np.nan,
            })
            continue
        avg_pred = float(p_pred[mask].mean())
        emp_rate = float(y_true[mask].mean())
        rows.append({
            "bin":       b,
            "bin_left":  edges[b],
            "bin_right": edges[b + 1],
            "n":         n,
            "avg_pred":  avg_pred,
            "emp_rate":  emp_rate,
            "abs_gap":   abs(avg_pred - emp_rate),
        })

    return pd.DataFrame(rows)


def reliability_plot_data(y_true, p_pred, bins: int = 10) -> pd.DataFrame:
    """
    Return data for a reliability / calibration curve.

    Identical to calibration_table but with cleaner column names
    and a 'perfectly_calibrated' reference column.
    """
    table = calibration_table(y_true, p_pred, bins=bins)
    table = table.rename(columns={
        "avg_pred": "mean_pred",
        "emp_rate": "fraction_positives",
    })
    # Reference: perfectly calibrated line
    table["perfectly_calibrated"] = (table["bin_left"] + table["bin_right"]) / 2.0
    return table


def calibration_audit_report(
    y_true, p_pred_raw, p_pred_cal,
    bins: int = 10,
) -> dict:
    """
    One-call calibration audit: before/after calibration comparison.

    Returns dict with:
        raw_*    : metrics before calibration
        cal_*    : metrics after calibration
        delta_*  : improvement (raw - cal; positive = improvement)
        table_*  : reliability plot data for both
    """
    from src.metrics import full_metric_bundle

    raw_metrics = full_metric_bundle(y_true, p_pred_raw)
    cal_metrics = full_metric_bundle(y_true, p_pred_cal)

    return {
        "raw_log_loss":  raw_metrics["log_loss"],
        "cal_log_loss":  cal_metrics["log_loss"],
        "delta_log_loss": raw_metrics["log_loss"] - cal_metrics["log_loss"],

        "raw_brier":  raw_metrics["brier"],
        "cal_brier":  cal_metrics["brier"],
        "delta_brier": raw_metrics["brier"] - cal_metrics["brier"],

        "raw_ece":  raw_metrics["ece"],
        "cal_ece":  cal_metrics["ece"],
        "delta_ece": raw_metrics["ece"] - cal_metrics["ece"],

        "table_raw": reliability_plot_data(y_true, p_pred_raw, bins=bins),
        "table_cal": reliability_plot_data(y_true, p_pred_cal, bins=bins),
    }
