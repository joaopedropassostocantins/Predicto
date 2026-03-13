# src/utils.py

from __future__ import annotations
import numpy as np


def clip_probs(p, pmin: float = 1e-6, pmax: float = 1 - 1e-6):
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, pmin, pmax)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, pmin: float = 1e-6, pmax: float = 1 - 1e-6):
    p = clip_probs(p, pmin=pmin, pmax=pmax)
    return np.log(p / (1.0 - p))


def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


def normalize_blend(weights: dict) -> dict:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Blend weights must sum to positive value.")
    return {k: v / total for k, v in weights.items()}
