# src/poisson.py — Poisson scoring model with shrinkage toward league mean

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, poisson


# ---------------------------------------------------------------------------
# Rate estimation with CI and optional Bayesian shrinkage
# ---------------------------------------------------------------------------

def poisson_rate_ci(scores, alpha: float = 0.10):
    """
    Exact Poisson confidence interval using the chi-squared method.

    Returns (lambda_hat, lower_ci, upper_ci).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.nan, np.nan, np.nan

    K = float(scores.sum())
    lam_hat = K / n

    lower = (0.5 * chi2.ppf(alpha / 2.0, 2.0 * K) / n) if K > 0 else 0.0
    upper = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (K + 1)) / n

    return float(lam_hat), float(lower), float(upper)


def build_window_stats(
    team_games: pd.DataFrame,
    window,
    alpha: float = 0.10,
    shrinkage: float = 0.0,
    league_avg_for: float | None = None,
    league_avg_against: float | None = None,
) -> dict:
    """
    Estimate Poisson scoring and conceding rates for a given window.

    Parameters
    ----------
    window : int or "season"
        Number of most recent games to use, or "season" for all games.
    shrinkage : float
        Bayesian shrinkage weight toward league average.
        0.0 → use raw MLE; 1.0 → use league average.
        Applied as: lambda_shrunk = (1 - shrinkage) * lambda_mle
                                    + shrinkage * league_avg
    league_avg_for, league_avg_against : float, optional
        Required when shrinkage > 0.  If None and shrinkage > 0, shrinkage
        is silently ignored.
    """
    g = team_games.sort_values("DayNum")
    use = g if window == "season" else g.tail(int(window))

    pf = use["PointsFor"].to_numpy(dtype=float)
    pa = use["PointsAgainst"].to_numpy(dtype=float)

    lam_for,     lam_for_lo,     lam_for_hi     = poisson_rate_ci(pf, alpha=alpha)
    lam_against, lam_against_lo, lam_against_hi = poisson_rate_ci(pa, alpha=alpha)

    # Apply shrinkage when league averages are provided
    if shrinkage > 0.0 and league_avg_for is not None and league_avg_against is not None:
        w = float(shrinkage)
        lam_for     = (1.0 - w) * lam_for     + w * league_avg_for
        lam_against = (1.0 - w) * lam_against + w * league_avg_against
        # Shrink CI bounds proportionally (preserve relative width)
        lam_for_lo  = (1.0 - w) * lam_for_lo  + w * league_avg_for
        lam_for_hi  = (1.0 - w) * lam_for_hi  + w * league_avg_for
        lam_against_lo = (1.0 - w) * lam_against_lo + w * league_avg_against
        lam_against_hi = (1.0 - w) * lam_against_hi + w * league_avg_against

    return {
        "lambda_for":          float(lam_for)    if not np.isnan(lam_for)     else 70.0,
        "lambda_for_ci_low":   float(lam_for_lo) if not np.isnan(lam_for_lo)  else 65.0,
        "lambda_for_ci_high":  float(lam_for_hi) if not np.isnan(lam_for_hi)  else 75.0,
        "lambda_against":          float(lam_against)    if not np.isnan(lam_against)     else 70.0,
        "lambda_against_ci_low":   float(lam_against_lo) if not np.isnan(lam_against_lo)  else 65.0,
        "lambda_against_ci_high":  float(lam_against_hi) if not np.isnan(lam_against_hi)  else 75.0,
    }


# ---------------------------------------------------------------------------
# Match-level joint Poisson distribution
# ---------------------------------------------------------------------------

def poisson_match_distribution(
    lambda_low: float,
    lambda_high: float,
    max_points: int = 220,
) -> dict:
    """
    Compute the joint Poisson PMF for two independent scoring processes.

    Returns P(Low wins), P(tie), P(High wins), expected margin, and
    expected scores.  Ties are split 50/50.
    """
    # Guard against invalid lambdas
    lambda_low  = max(float(lambda_low),  0.5)
    lambda_high = max(float(lambda_high), 0.5)

    pts   = np.arange(0, max_points + 1)
    p_low  = poisson.pmf(pts, lambda_low)
    p_high = poisson.pmf(pts, lambda_high)

    joint    = np.outer(p_low, p_high)
    p_low_gt = float(np.tril(joint, k=-1).sum())
    p_tie    = float(np.trace(joint))
    p_high_gt= float(np.triu(joint, k=1).sum())

    total = p_low_gt + p_tie + p_high_gt
    if total > 0:
        p_low_gt  /= total
        p_tie     /= total
        p_high_gt /= total

    p_low_win  = p_low_gt  + 0.5 * p_tie
    p_high_win = p_high_gt + 0.5 * p_tie

    exp_low  = float((pts * p_low).sum())
    exp_high = float((pts * p_high).sum())

    return {
        "p_low_win":       float(p_low_win),
        "p_high_win":      float(p_high_win),
        "p_tie":           float(p_tie),
        "exp_low":         exp_low,
        "exp_high":        exp_high,
        "expected_margin": exp_low - exp_high,
    }


# ---------------------------------------------------------------------------
# Blend Poisson lambdas across windows and compute matchup features
# ---------------------------------------------------------------------------

def add_poisson_matchup_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Blend Poisson lambdas from multiple windows and compute win probabilities.

    Expected columns in `df` (for both _low and _high suffixes):
        recent3_lambda_for, recent3_lambda_against,
        recent5_lambda_for, recent5_lambda_against,
        season_lambda_for,  season_lambda_against.

    Writes to df:
        poisson_lambda_low, poisson_lambda_high,
        poisson_win_prob, poisson_expected_margin, poisson_win_prob_centered.
    """
    out = df.copy()

    weights = cfg["poisson_blend_weights"]
    total = sum(weights.values())
    w3 = weights["recent3"] / total
    w5 = weights["recent5"] / total
    ws = weights["season"]  / total

    # Blended expected scoring rate for each side:
    # λ_team = blend of (team's own scoring lambda + opponent's conceding lambda) / 2
    out["poisson_lambda_low"] = (
        w3 * (out["recent3_lambda_for_low"] + out["recent3_lambda_against_high"]) / 2.0
        + w5 * (out["recent5_lambda_for_low"] + out["recent5_lambda_against_high"]) / 2.0
        + ws * (out["season_lambda_for_low"]  + out["season_lambda_against_high"])  / 2.0
    )
    out["poisson_lambda_high"] = (
        w3 * (out["recent3_lambda_for_high"] + out["recent3_lambda_against_low"]) / 2.0
        + w5 * (out["recent5_lambda_for_high"] + out["recent5_lambda_against_low"]) / 2.0
        + ws * (out["season_lambda_for_high"]  + out["season_lambda_against_low"])  / 2.0
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
    out["poisson_win_prob"]          = pois_df["p_low_win"].values
    out["poisson_expected_margin"]   = pois_df["expected_margin"].values
    out["poisson_win_prob_centered"] = out["poisson_win_prob"] - 0.5

    return out
