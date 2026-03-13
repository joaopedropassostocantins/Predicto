from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def clip_probs(p, eps: float = 1e-6):
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


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


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, p, y):
        x = np.asarray(clip_probs(p)).reshape(-1, 1)
        y = np.asarray(y).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p):
        x = np.asarray(clip_probs(p)).reshape(-1, 1)
        out = self.model.predict_proba(x)[:, 1]
        return clip_probs(out)


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p, y):
        x = np.asarray(clip_probs(p))
        y = np.asarray(y).astype(int)
        self.model.fit(x, y)
        return self

    def predict(self, p):
        x = np.asarray(clip_probs(p))
        out = self.model.predict(x)
        return clip_probs(out)


def fit_calibrator(method: str, p_train, y_train):
    method = method.lower().strip()

    if method == "identity":
        calibrator = IdentityCalibrator()
    elif method == "platt":
        calibrator = PlattCalibrator()
    elif method == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        raise ValueError(f"Método de calibração desconhecido: {method}")

    return calibrator.fit(p_train, y_train)


def apply_calibrator(calibrator, p):
    return clip_probs(calibrator.predict(p))


def choose_best_calibrator(
    p_train,
    y_train,
    p_valid,
    y_valid,
    scorer_fn,
    methods: Optional[list] = None,
) -> CalibrationResult:
    if methods is None:
        methods = ["identity", "platt", "isotonic"]

    best_result = None

    for method in methods:
        calibrator = fit_calibrator(method, p_train, y_train)
        p_cal = apply_calibrator(calibrator, p_valid)
        metrics = scorer_fn(y_valid, p_cal)

        candidate = CalibrationResult(
            method=method,
            fitted=calibrator,
            metrics=metrics,
        )

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
            rows.append(
                {
                    "bin": b,
                    "bin_left": edges[b],
                    "bin_right": edges[b + 1],
                    "n": 0,
                    "avg_pred": np.nan,
                    "emp_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue

        avg_pred = float(p_pred[mask].mean())
        emp_rate = float(y_true[mask].mean())
        abs_gap = abs(avg_pred - emp_rate)

        rows.append(
            {
                "bin": b,
                "bin_left": edges[b],
                "bin_right": edges[b + 1],
                "n": n,
                "avg_pred": avg_pred,
                "emp_rate": emp_rate,
                "abs_gap": abs_gap,
            }
        )

    return pd.DataFrame(rows)
