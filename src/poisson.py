# src/poisson.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, poisson


def poisson_rate_ci(scores, alpha=0.10):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.nan, np.nan, np.nan

    K = scores.sum()
    lam_hat = K / n

    if K <= 0:
        lower = 0.0
    else:
        lower = 0.5 * chi2.ppf(alpha / 2.0, 2.0 * K) / n

    upper = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (K + 1)) / n
    return float(lam_hat), float(lower), float(upper)


def build_window_stats(team_games: pd.DataFrame, window, alpha=0.10):
    g = team_games.sort_values("DayNum")
    use = g if window == "season" else g.tail(int(window))

    pf = use["PointsFor"].to_numpy(dtype=float)
    pa = use["PointsAgainst"].to_numpy(dtype=float)

    lam_for, lam_for_lo, lam_for_hi = poisson_rate_ci(pf, alpha=alpha)
    lam_against, lam_against_lo, lam_against_hi = poisson_rate_ci(pa, alpha=alpha)

    return {
        "lambda_for": lam_for,
        "lambda_for_ci_low": lam_for_lo,
        "lambda_for_ci_high": lam_for_hi,
        "lambda_against": lam_against,
        "lambda_against_ci_low": lam_against_lo,
        "lambda_against_ci_high": lam_against_hi,
    }


def poisson_match_distribution(lambda_low: float, lambda_high: float, max_points: int = 220):
    pts = np.arange(0, max_points + 1)
    p_low = poisson.pmf(pts, lambda_low)
    p_high = poisson.pmf(pts, lambda_high)

    joint = np.outer(p_low, p_high)

    p_low_gt = np.tril(joint, k=-1).sum()
    p_tie = np.trace(joint)
    p_high_gt = np.triu(joint, k=1).sum()

    total = p_low_gt + p_tie + p_high_gt
    if total > 0:
        p_low_gt /= total
        p_tie /= total
        p_high_gt /= total

    p_low_win = p_low_gt + 0.5 * p_tie
    p_high_win = p_high_gt + 0.5 * p_tie

    exp_low = float((pts * p_low).sum())
    exp_high = float((pts * p_high).sum())

    return {
        "p_low_win": float(p_low_win),
        "p_high_win": float(p_high_win),
        "p_tie": float(p_tie),
        "exp_low": exp_low,
        "exp_high": exp_high,
        "expected_margin": exp_low - exp_high,
    }


def add_poisson_matchup_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    weights = cfg["poisson_blend_weights"]
    total = sum(weights.values())
    w3 = weights["recent3"] / total
    w5 = weights["recent5"] / total
    ws = weights["season"] / total

    out["poisson_lambda_low"] = (
        w3 * ((out["recent3_lambda_for_low"] + out["recent3_lambda_against_high"]) / 2.0)
        + w5 * ((out["recent5_lambda_for_low"] + out["recent5_lambda_against_high"]) / 2.0)
        + ws * ((out["season_lambda_for_low"] + out["season_lambda_against_high"]) / 2.0)
    )

    out["poisson_lambda_high"] = (
        w3 * ((out["recent3_lambda_for_high"] + out["recent3_lambda_against_low"]) / 2.0)
        + w5 * ((out["recent5_lambda_for_high"] + out["recent5_lambda_against_low"]) / 2.0)
        + ws * ((out["season_lambda_for_high"] + out["season_lambda_against_low"]) / 2.0)
    )

    pois = out.apply(
        lambda r: poisson_match_distribution(
            float(r["poisson_lambda_low"]),
            float(r["poisson_lambda_high"]),
            max_points=cfg["max_points_poisson"],
        ),
        axis=1,
    )

    pois_df = pd.DataFrame(list(pois))
    out["poisson_win_prob"] = pois_df["p_low_win"].values
    out["poisson_expected_margin"] = pois_df["expected_margin"].values
    out["poisson_win_prob_centered"] = out["poisson_win_prob"] - 0.5

    return out
