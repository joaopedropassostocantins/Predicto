# src/features.py

from __future__ import annotations

import numpy as np
import pandas as pd

from src.poisson import build_window_stats, add_poisson_matchup_features
from src.ratings import calculate_elo_from_games, get_latest_elo


def build_team_features(games_df: pd.DataFrame, season: int, recent_games_window: int, alpha_ci: float) -> pd.DataFrame:
    season_df = games_df[games_df["Season"] == season].copy()

    season_stats = (
        season_df.groupby("TeamID")
        .agg(
            season_points_for=("PointsFor", "mean"),
            season_points_against=("PointsAgainst", "mean"),
            season_margin=("Margin", "mean"),
            season_wins=("Win", "sum"),
            season_games_played=("TeamID", "count"),
            season_points_for_std=("PointsFor", "std"),
            season_margin_std=("Margin", "std"),
        )
        .reset_index()
    )
    season_stats["season_win_pct"] = season_stats["season_wins"] / season_stats["season_games_played"]

    for w in [3, 5]:
        recent_form = (
            season_df.groupby("TeamID", group_keys=False)
            .apply(lambda x: x.sort_values("DayNum").tail(w))
            .groupby("TeamID")
            .agg(
                **{
                    f"recent{w}_points_for": ("PointsFor", "mean"),
                    f"recent{w}_points_against": ("PointsAgainst", "mean"),
                    f"recent{w}_margin": ("Margin", "mean"),
                }
            )
            .reset_index()
        )
        season_stats = season_stats.merge(recent_form, on="TeamID", how="left")
        season_stats[f"recent{w}_points_for"] = season_stats[f"recent{w}_points_for"].fillna(season_stats["season_points_for"])
        season_stats[f"recent{w}_points_against"] = season_stats[f"recent{w}_points_against"].fillna(season_stats["season_points_against"])
        season_stats[f"recent{w}_margin"] = season_stats[f"recent{w}_margin"].fillna(season_stats["season_margin"])

    # strength of schedule proxy
    opp_strength = (
        season_df.groupby("OppTeamID")
        .agg(
            opp_avg_margin=("Margin", "mean"),
            opp_win_rate=("Win", "mean"),
        )
        .reset_index()
        .rename(columns={"OppTeamID": "TeamID"})
    )

    sos = (
        season_df.merge(
            opp_strength.rename(
                columns={
                    "TeamID": "OppTeamID",
                    "opp_avg_margin": "opp_margin_proxy",
                    "opp_win_rate": "opp_win_proxy",
                }
            ),
            on="OppTeamID",
            how="left",
        )
        .groupby("TeamID")
        .agg(
            sos_margin=("opp_margin_proxy", "mean"),
            sos_win_rate=("opp_win_proxy", "mean"),
        )
        .reset_index()
    )
    season_stats = season_stats.merge(sos, on="TeamID", how="left")

    season_stats["quality_proxy"] = (
        0.55 * season_stats["season_margin"].fillna(0.0)
        + 25.0 * season_stats["season_win_pct"].fillna(0.5)
        + 0.25 * season_stats["sos_margin"].fillna(0.0)
    )
    season_stats["rank_proxy"] = season_stats["quality_proxy"].rank(ascending=False, method="average")

    # Elo
    base_games = (
        season_df[season_df["Win"] == 1][["Season", "DayNum", "TeamID", "OppTeamID"]]
        .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    elo_history, _ = calculate_elo_from_games(base_games)
    latest_elo = get_latest_elo(elo_history)
    season_stats = season_stats.merge(latest_elo[["Season", "TeamID", "Elo"]], on=["Season", "TeamID"], how="left")
    season_stats["Elo"] = season_stats["Elo"].fillna(1500.0)

    # Poisson windows
    poisson_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        row = {"Season": season, "TeamID": team_id}
        for window in [3, 5, "season"]:
            stats = build_window_stats(grp, window, alpha=alpha_ci)
            prefix = f"recent{window}" if window != "season" else "season"
            row[f"{prefix}_lambda_for"] = stats["lambda_for"]
            row[f"{prefix}_lambda_for_ci_low"] = stats["lambda_for_ci_low"]
            row[f"{prefix}_lambda_for_ci_high"] = stats["lambda_for_ci_high"]
            row[f"{prefix}_lambda_against"] = stats["lambda_against"]
            row[f"{prefix}_lambda_against_ci_low"] = stats["lambda_against_ci_low"]
            row[f"{prefix}_lambda_against_ci_high"] = stats["lambda_against_ci_high"]
        poisson_rows.append(row)

    poisson_df = pd.DataFrame(poisson_rows)
    season_stats = season_stats.merge(poisson_df, on=["Season", "TeamID"], how="left")

    return season_stats


def attach_team_features(eval_df: pd.DataFrame, m_features: pd.DataFrame, w_features: pd.DataFrame, seeds_m: pd.DataFrame, seeds_w: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = eval_df.copy()

    all_features = pd.concat([m_features, w_features], ignore_index=True)
    all_seeds = pd.concat([seeds_m, seeds_w], ignore_index=True)

    out = out.merge(
        all_seeds.rename(columns={"TeamID": "TeamIDLow", "SeedNum": "SeedLow"})[["Season", "TeamIDLow", "SeedLow"]],
        on=["Season", "TeamIDLow"],
        how="left",
    )
    out = out.merge(
        all_seeds.rename(columns={"TeamID": "TeamIDHigh", "SeedNum": "SeedHigh"})[["Season", "TeamIDHigh", "SeedHigh"]],
        on=["Season", "TeamIDHigh"],
        how="left",
    )

    out["SeedLow"] = out["SeedLow"].fillna(cfg["fallback_seed"])
    out["SeedHigh"] = out["SeedHigh"].fillna(cfg["fallback_seed"])

    feature_cols = [c for c in all_features.columns if c not in ["Season", "TeamID"]]

    low_map = {"TeamID": "TeamIDLow"}
    high_map = {"TeamID": "TeamIDHigh"}
    low_map.update({c: f"{c}_low" for c in feature_cols})
    high_map.update({c: f"{c}_high" for c in feature_cols})

    out = out.merge(all_features.rename(columns=low_map), on=["Season", "TeamIDLow"], how="left")
    out = out.merge(all_features.rename(columns=high_map), on=["Season", "TeamIDHigh"], how="left")

    fallback_numeric = {
        "season_points_for": cfg["fallback_points_for"],
        "season_points_against": cfg["fallback_points_against"],
        "season_margin": 0.0,
        "season_win_pct": 0.5,
        "season_games_played": 0.0,
        "recent3_points_for": cfg["fallback_points_for"],
        "recent3_points_against": cfg["fallback_points_against"],
        "recent3_margin": 0.0,
        "recent5_points_for": cfg["fallback_points_for"],
        "recent5_points_against": cfg["fallback_points_against"],
        "recent5_margin": 0.0,
        "Elo": cfg["fallback_elo"],
        "sos_margin": 0.0,
        "sos_win_rate": 0.5,
        "quality_proxy": 0.0,
        "rank_proxy": 0.0,
        "season_points_for_std": 12.0,
        "season_margin_std": 12.0,
        "recent3_lambda_for": cfg["fallback_points_for"],
        "recent3_lambda_against": cfg["fallback_points_against"],
        "recent5_lambda_for": cfg["fallback_points_for"],
        "recent5_lambda_against": cfg["fallback_points_against"],
        "season_lambda_for": cfg["fallback_points_for"],
        "season_lambda_against": cfg["fallback_points_against"],
    }

    for base_col, fill_val in fallback_numeric.items():
        low_col = f"{base_col}_low"
        high_col = f"{base_col}_high"
        if low_col in out.columns:
            out[low_col] = out[low_col].fillna(fill_val)
        if high_col in out.columns:
            out[high_col] = out[high_col].fillna(fill_val)

    return out


def make_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["seed_diff"] = out["SeedHigh"] - out["SeedLow"]
    out["elo_diff"] = out["Elo_low"] - out["Elo_high"]

    out["season_points_for_diff"] = out["season_points_for_low"] - out["season_points_for_high"]
    out["season_points_against_diff"] = out["season_points_against_low"] - out["season_points_against_high"]
    out["season_margin_diff"] = out["season_margin_low"] - out["season_margin_high"]
    out["season_win_pct_diff"] = out["season_win_pct_low"] - out["season_win_pct_high"]

    out["recent3_points_for_diff"] = out["recent3_points_for_low"] - out["recent3_points_for_high"]
    out["recent3_points_against_diff"] = out["recent3_points_against_low"] - out["recent3_points_against_high"]
    out["recent3_margin_diff"] = out["recent3_margin_low"] - out["recent3_margin_high"]

    out["recent5_points_for_diff"] = out["recent5_points_for_low"] - out["recent5_points_for_high"]
    out["recent5_points_against_diff"] = out["recent5_points_against_low"] - out["recent5_points_against_high"]
    out["recent5_margin_diff"] = out["recent5_margin_low"] - out["recent5_margin_high"]

    out["offense_vs_defense_low"] = out["season_points_for_low"] - out["season_points_against_high"]
    out["offense_vs_defense_high"] = out["season_points_for_high"] - out["season_points_against_low"]
    out["matchup_diff"] = out["offense_vs_defense_low"] - out["offense_vs_defense_high"]

    out["sos_diff"] = out["sos_margin_low"] - out["sos_margin_high"]
    out["quality_diff"] = out["quality_proxy_low"] - out["quality_proxy_high"]
    out["rank_diff_signed"] = out["rank_proxy_high"] - out["rank_proxy_low"]
    out["consistency_edge"] = out["season_points_for_std_high"] - out["season_points_for_std_low"]

    out = add_poisson_matchup_features(out, cfg={
        "poisson_blend_weights": {"recent3": 0.40, "recent5": 0.30, "season": 0.30},
        "max_points_poisson": 220,
    })

    return out
