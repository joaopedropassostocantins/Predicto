# src/calibration.py — Probability calibration v4.1
#
# Changes from v4.0:
#   - IsotonicCalibrator: guarda n_min=50 adicionada.
#     Isotonic com <50 amostras é propenso a overfit severo.
#     Fallback automático para TemperatureCalibrator quando n < 50.
#   - choose_best_calibrator_multifold(): nova função para seleção multi-fold.
#     Problema v4.0: seleção com single-fold (apenas seasons[-1]) era frágil.
#     Novo: leave-one-out sobre todos os folds OOF disponíveis, resultado médio.
#     Mais robusto, especialmente quando n_seasons é pequeno.
#   - temperature_candidates extendidos até 2.0 (era máx 1.50).
#     Modelos esportivos podem ser significativamente overconfident.
#
# Changes from v3:
#   - choose_best_calibrator: selection criterion changed from Brier → Log Loss.
#   - TemperatureCalibrator: removed candidates < 1.0.
#   - Added: reliability_plot_data(), calibration_audit_report().
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

    v4.1: default range extended to 2.0 (sports models can be overconfident).
    """
    def __init__(self, temperatures=None):
        # Default: all values ≥ 1.0 (never increase confidence)
        # Extended to 2.0 in v4.1: sports models can be significantly overconfident
        self.temperatures    = temperatures or [
            1.00, 1.05, 1.08, 1.12, 1.15, 1.20, 1.25, 1.35, 1.50, 1.70, 2.00
        ]
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

    v4.1: Added n_min guard. If fewer than n_min samples provided, falls back
    to TemperatureCalibrator to avoid severe overfitting.
    """
    N_MIN = 50  # minimum samples for safe isotonic fitting

    def __init__(self):
        self.model     = IsotonicRegression(out_of_bounds="clip")
        self._fallback = None   # set to TemperatureCalibrator if n < N_MIN

    def fit(self, p, y):
        x = clip_probs(p)
        y = np.asarray(y).astype(int)
        if len(x) < self.N_MIN:
            # Too few samples: isotonic will memorise the training set.
            # Fall back to temperature scaling which is more regularised.
            self._fallback = TemperatureCalibrator()
            self._fallback.fit(x, y)
        else:
            self._fallback = None
            self.model.fit(x, y)
        return self

    def predict(self, p):
        if self._fallback is not None:
            return self._fallback.predict(p)
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

def choose_best_calibrator_multifold(
    oof_preds: list,
    scorer_fn,
    methods: Optional[List[str]] = None,
    cfg=None,
) -> "CalibrationResult":
    """
    Multi-fold leave-one-out calibrator selection from OOF predictions.

    v4.1: Replaces single-fold selection (was: use seasons[-1] as validation).
    New: for each fold i, train calibrator on all folds j≠i, evaluate on fold i.
    Final metric is the mean log_loss across all leave-one-out folds.
    More robust when n_seasons is small (5-10 typical).

    Parameters
    ----------
    oof_preds : list of (p_pred_array, y_true_array) tuples
        One entry per backtest season, ordered chronologically.
        Minimum 3 tuples required (first 2 warm-up, rest evaluated).
    scorer_fn : callable  (y_true, p_pred) → dict with 'log_loss' key.
    methods : list of str, optional  Calibrator methods to evaluate.
    cfg : dict, optional  CONFIG dict (for temperature candidates).

    Returns
    -------
    CalibrationResult with the best method and its mean metrics.
    """
    if methods is None:
        methods = ["identity", "temperature", "platt", "isotonic"]

    if len(oof_preds) < 3:
        # Not enough folds for LOO — fall back to last fold as val
        p_train = np.concatenate([t[0] for t in oof_preds[:-1]])
        y_train = np.concatenate([t[1] for t in oof_preds[:-1]])
        p_val   = oof_preds[-1][0]
        y_val   = oof_preds[-1][1]
        return choose_best_calibrator(
            p_train, y_train, p_val, y_val,
            scorer_fn=scorer_fn, methods=methods, cfg=cfg,
        )

    # Leave-one-out: for each fold, train on all others, evaluate on it
    method_scores: Dict[str, List[float]] = {m: [] for m in methods}

    for i in range(1, len(oof_preds)):   # start at 1 (fold 0 has no history)
        train_parts = [oof_preds[j] for j in range(i) if j != i]
        if not train_parts:
            continue
        p_tr = np.concatenate([t[0] for t in train_parts])
        y_tr = np.concatenate([t[1] for t in train_parts])
        p_va = oof_preds[i][0]
        y_va = oof_preds[i][1]

        for method in methods:
            try:
                cal  = fit_calibrator(method, p_tr, y_tr, cfg=cfg)
                p_c  = apply_calibrator(cal, p_va)
                m    = scorer_fn(y_va, p_c)
                method_scores[method].append(m.get("log_loss", 1.0))
            except Exception:
                method_scores[method].append(1.0)  # penalise failures

    # Select method with lowest mean log_loss across LOO folds
    best_method = min(
        methods,
        key=lambda m: float(np.mean(method_scores[m])) if method_scores[m] else 1.0,
    )

    # Fit final calibrator on ALL available OOF data
    p_all = np.concatenate([t[0] for t in oof_preds])
    y_all = np.concatenate([t[1] for t in oof_preds])
    final_cal = fit_calibrator(best_method, p_all, y_all, cfg=cfg)

    mean_scores = {
        "log_loss": float(np.mean(method_scores[best_method])) if method_scores[best_method] else 1.0,
    }
    return CalibrationResult(method=best_method, fitted=final_cal, metrics=mean_scores)


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
