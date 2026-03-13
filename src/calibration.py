# src/calibration.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils import clip_probs, logit, sigmoid


@dataclass
class CalibrationResult:
    method: str
    fitted: object
    metrics: Dict[str, float]


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def predict(self, p):
        return clip_probs(p)


class TemperatureCalibrator:
    def __init__(self, temperatures=None):
        self.temperatures = temperatures or [0.80, 1.00, 1.25, 1.50, 2.00]
        self.best_temperature = 1.0

    def fit(self, p, y):
        p = clip_probs(p)
        y = np.asarray(y).astype(int)

        best_t = 1.0
        best_loss = float("inf")
        base_logit = logit(p)

        for t in self.temperatures:
            p_t = sigmoid(base_logit / t)
            eps = 1e-12
            loss = -np.mean(y * np.log(np.clip(p_t, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p_t, eps, 1 - eps)))
            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.best_temperature = best_t
        return self

    def predict(self, p):
        p = clip_probs(p)
        return clip_probs(sigmoid(logit(p) / self.best_temperature))


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, p, y):
        x = logit(clip_probs(p)).reshape(-1, 1)
        y = np.asarray(y).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p):
        x = logit(clip_probs(p)).reshape(-1, 1)
        out = self.model.predict_proba(x)[:, 1]
        return clip_probs(out)


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p, y):
        x = clip_probs(p)
        y = np.asarray(y).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p):
        out = self.model.predict(clip_probs(p))
        return clip_probs(out)


def fit_calibrator(method: str, p_train, y_train, cfg=None):
    method = method.lower().strip()
    if method == "identity":
        calibrator = IdentityCalibrator()
    elif method == "temperature":
        temps = None if cfg is None else cfg.get("temperature_candidates")
        calibrator = TemperatureCalibrator(temperatures=temps)
    elif method == "platt":
        calibrator = PlattCalibrator()
    elif method == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return calibrator.fit(p_train, y_train)


def apply_calibrator(calibrator, p):
    return clip_probs(calibrator.predict(p))


def choose_best_calibrator(p_train, y_train, p_valid, y_valid, scorer_fn, methods: Optional[List[str]] = None, cfg=None) -> CalibrationResult:
    if methods is None:
        methods = ["identity", "temperature", "platt", "isotonic"]

    best_result = None
    for method in methods:
        calibrator = fit_calibrator(method, p_train, y_train, cfg=cfg)
        p_cal = apply_calibrator(calibrator, p_valid)
        metrics = scorer_fn(y_valid, p_cal)
        candidate = CalibrationResult(method=method, fitted=calibrator, metrics=metrics)
        if best_result is None or candidate.metrics["brier"] < best_result.metrics["brier"]:
            best_result = candidate
    return best_result


def calibration_table(y_true, p_pred, bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(clip_probs(p_pred))
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p_pred, edges[1:-1], right=True)

    rows = []
    for b in range(bins):
        mask = bucket == b
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": b,
                "bin_left": edges[b],
                "bin_right": edges[b + 1],
                "n": 0,
                "avg_pred": np.nan,
                "emp_rate": np.nan,
                "abs_gap": np.nan,
            })
            continue

        avg_pred = float(p_pred[mask].mean())
        emp_rate = float(y_true[mask].mean())
        abs_gap = abs(avg_pred - emp_rate)

        rows.append({
            "bin": b,
            "bin_left": edges[b],
            "bin_right": edges[b + 1],
            "n": n,
            "avg_pred": avg_pred,
            "emp_rate": emp_rate,
            "abs_gap": abs_gap,
        })

    return pd.DataFrame(rows)
