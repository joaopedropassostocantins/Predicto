# src/features.py

import numpy as np
import pandas as pd
from scipy.stats import chi2


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
        lower = 0.5 * chi2.ppf(alpha / 2, 2 * K) / n

    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (K + 1)) / n
    return float(lam_hat), float(lower), float(upper)


def build_team_features(games_long: pd.DataFrame, season: int, recent_n: int, alpha: float) -> pd.DataFrame:
    season_games = games_long[games_long["Season"] == season].copy()

    rows = []
    for team_id, grp in season_games.groupby("TeamID"):
        grp = grp.sort_values("DayNum")
        recent = grp.tail(recent_n)

        pf_recent = recent["PointsFor"].values
        pa_recent = recent["PointsAgainst"].values
        margin_recent = recent["Margin"].values

        pf_all = grp["PointsFor"].values
        pa_all = grp["PointsAgainst"].values
        margin_all = grp["Margin"].values
        win_all = grp["Win"].values

        lam_for, lam_for_low, lam_for_high = poisson_rate_ci(pf_recent, alpha=alpha)
        lam_against, lam_against_low, lam_against_high = poisson_rate_ci(pa_recent, alpha=alpha)

        rows.append(
            {
                "Season": season,
                "TeamID": int(team_id),
                "recent_avg_for": float(np.mean(pf_recent)) if len(pf_recent) else np.nan,
                "recent_avg_against": float(np.mean(pa_recent)) if len(pa_recent) else np.nan,
                "recent_avg_margin": float(np.mean(margin_recent)) if len(margin_recent) else np.nan,
                "recent_std_for": float(np.std(pf_recent, ddof=0)) if len(pf_recent) else np.nan,
                "season_win_pct": float(np.mean(win_all)) if len(win_all) else np.nan,
                "season_avg_margin": float(np.mean(margin_all)) if len(margin_all) else np.nan,
                "lambda_for": lam_for,
                "lambda_for_ci_low": lam_for_low,
                "lambda_for_ci_high": lam_for_high,
                "lambda_against": lam_against,
                "lambda_against_ci_low": lam_against_low,
                "lambda_against_ci_high": lam_against_high,
            }
        )

    return pd.DataFrame(rows)


def attach_team_features(matchups, team_features_m, team_features_w, seeds_m, seeds_w, cfg):
    team_features_all = pd.concat(
        [team_features_m.assign(Gender="M"), team_features_w.assign(Gender="W")],
        ignore_index=True,
    )
    seeds_all = pd.concat([seeds_m.assign(Gender="M"), seeds_w.assign(Gender="W")], ignore_index=True)

    low = team_features_all.rename(
        columns={
            "TeamID": "TeamIDLow",
            "recent_avg_for": "low_recent_avg_for",
            "recent_avg_against": "low_recent_avg_against",
            "recent_avg_margin": "low_recent_avg_margin",
            "recent_std_for": "low_recent_std_for",
            "season_win_pct": "low_season_win_pct",
            "season_avg_margin": "low_season_avg_margin",
            "lambda_for": "low_lambda_for",
            "lambda_for_ci_low": "low_lambda_for_ci_low",
            "lambda_for_ci_high": "low_lambda_for_ci_high",
            "lambda_against": "low_lambda_against",
            "lambda_against_ci_low": "low_lambda_against_ci_low",
            "lambda_against_ci_high": "low_lambda_against_ci_high",
        }
    )

    high = team_features_all.rename(
        columns={
            "TeamID": "TeamIDHigh",
            "recent_avg_for": "high_recent_avg_for",
            "recent_avg_against": "high_recent_avg_against",
            "recent_avg_margin": "high_recent_avg_margin",
            "recent_std_for": "high_recent_std_for",
            "season_win_pct": "high_season_win_pct",
            "season_avg_margin": "high_season_avg_margin",
            "lambda_for": "high_lambda_for",
            "lambda_for_ci_low": "high_lambda_for_ci_low",
            "lambda_for_ci_high": "high_lambda_for_ci_high",
            "lambda_against": "high_lambda_against",
            "lambda_against_ci_low": "high_lambda_against_ci_low",
            "lambda_against_ci_high": "high_lambda_against_ci_high",
        }
    )

    seeds_low = seeds_all.rename(columns={"TeamID": "TeamIDLow", "SeedNum": "low_seed"})
    seeds_high = seeds_all.rename(columns={"TeamID": "TeamIDHigh", "SeedNum": "high_seed"})

    out = (
        matchups.merge(
            low[
                [
                    "Gender",
                    "Season",
                    "TeamIDLow",
                    "low_recent_avg_for",
                    "low_recent_avg_against",
                    "low_recent_avg_margin",
                    "low_recent_std_for",
                    "low_season_win_pct",
                    "low_season_avg_margin",
                    "low_lambda_for",
                    "low_lambda_for_ci_low",
                    "low_lambda_for_ci_high",
                    "low_lambda_against",
                    "low_lambda_against_ci_low",
                    "low_lambda_against_ci_high",
                ]
            ],
            on=["Gender", "Season", "TeamIDLow"],
            how="left",
        )
        .merge(
            high[
                [
                    "Gender",
                    "Season",
                    "TeamIDHigh",
                    "high_recent_avg_for",
                    "high_recent_avg_against",
                    "high_recent_avg_margin",
                    "high_recent_std_for",
                    "high_season_win_pct",
                    "high_season_avg_margin",
                    "high_lambda_for",
                    "high_lambda_for_ci_low",
                    "high_lambda_for_ci_high",
                    "high_lambda_against",
                    "high_lambda_against_ci_low",
                    "high_lambda_against_ci_high",
                ]
            ],
            on=["Gender", "Season", "TeamIDHigh"],
            how="left",
        )
        .merge(seeds_low[["Gender", "Season", "TeamIDLow", "low_seed"]], on=["Gender", "Season", "TeamIDLow"], how="left")
        .merge(
            seeds_high[["Gender", "Season", "TeamIDHigh", "high_seed"]],
            on=["Gender", "Season", "TeamIDHigh"],
            how="left",
        )
    )

    fill_values = {
        "low_recent_avg_for": cfg["fallback_points_for"],
        "high_recent_avg_for": cfg["fallback_points_for"],
        "low_recent_avg_against": cfg["fallback_points_against"],
        "high_recent_avg_against": cfg["fallback_points_against"],
        "low_recent_avg_margin": 0.0,
        "high_recent_avg_margin": 0.0,
        "low_recent_std_for": 12.0,
        "high_recent_std_for": 12.0,
        "low_season_win_pct": 0.5,
        "high_season_win_pct": 0.5,
        "low_season_avg_margin": 0.0,
        "high_season_avg_margin": 0.0,
        "low_lambda_for": cfg["fallback_points_for"],
        "high_lambda_for": cfg["fallback_points_for"],
        "low_lambda_against": cfg["fallback_points_against"],
        "high_lambda_against": cfg["fallback_points_against"],
        "low_lambda_for_ci_low": cfg["fallback_points_for"] - 5,
        "high_lambda_for_ci_low": cfg["fallback_points_for"] - 5,
        "low_lambda_for_ci_high": cfg["fallback_points_for"] + 5,
        "high_lambda_for_ci_high": cfg["fallback_points_for"] + 5,
        "low_lambda_against_ci_low": cfg["fallback_points_against"] - 5,
        "high_lambda_against_ci_low": cfg["fallback_points_against"] - 5,
        "low_lambda_against_ci_high": cfg["fallback_points_against"] + 5,
        "high_lambda_against_ci_high": cfg["fallback_points_against"] + 5,
        "low_seed": cfg["fallback_seed"],
        "high_seed": cfg["fallback_seed"],
    }

    for col, val in fill_values.items():
        out[col] = out[col].fillna(val)

    return out


def make_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["recent_offense_diff"] = out["low_recent_avg_for"] - out["high_recent_avg_for"]
    out["recent_defense_diff"] = out["high_recent_avg_against"] - out["low_recent_avg_against"]
    out["recent_net_rating_diff"] = out["low_recent_avg_margin"] - out["high_recent_avg_margin"]
    out["season_win_pct_diff"] = out["low_season_win_pct"] - out["high_season_win_pct"]
    out["season_avg_margin_diff"] = out["low_season_avg_margin"] - out["high_season_avg_margin"]
    out["seed_diff"] = out["high_seed"] - out["low_seed"]
    out["low_expected_points_matchup"] = (out["low_lambda_for"] + out["high_lambda_against"]) / 2.0
    out["high_expected_points_matchup"] = (out["high_lambda_for"] + out["low_lambda_against"]) / 2.0
    out["matchup_attack_vs_defense_diff"] = out["low_expected_points_matchup"] - out["high_expected_points_matchup"]
    out["consistency_diff"] = out["high_recent_std_for"] - out["low_recent_std_for"]
    out["low_ci_width"] = out["low_lambda_for_ci_high"] - out["low_lambda_for_ci_low"]
    out["high_ci_width"] = out["high_lambda_for_ci_high"] - out["high_lambda_for_ci_low"]
    out["ci_width_diff"] = out["high_ci_width"] - out["low_ci_width"]
    return out
