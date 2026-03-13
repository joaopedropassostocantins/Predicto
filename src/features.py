
import pandas as pd
import numpy as np
from src.ratings import calculate_elo, get_latest_elo

def build_team_features(games_df: pd.DataFrame, season: int, recent_games_window: int, alpha_ci: float) -> pd.DataFrame:
    season_df = games_df[games_df["Season"] == season].copy()
    
    # Calculate season-long averages
    season_stats = season_df.groupby("TeamID").agg(
        season_points_for=("PointsFor", "mean"),
        season_points_against=("PointsAgainst", "mean"),
        season_margin=("Margin", "mean"),
        season_wins=("Win", "sum"),
        season_games_played=("TeamID", "count"),
        season_points_for_std=("PointsFor", "std"),
        season_margin_std=("Margin", "std"),
    ).reset_index()
    season_stats["season_win_pct"] = season_stats["season_wins"] / season_stats["season_games_played"]

    # Calculate recent form for multiple windows
    windows = [3, 5]
    for w in windows:
        recent_stats = season_df.groupby("TeamID").apply(
            lambda x: x.tail(w)
        ).reset_index(drop=True)

        recent_form = recent_stats.groupby("TeamID").agg(**{
            f"recent{w}_points_for": ("PointsFor", "mean"),
            f"recent{w}_points_against": ("PointsAgainst", "mean"),
            f"recent{w}_margin": ("Margin", "mean"),
        }).reset_index()
        
        season_stats = season_stats.merge(recent_form, on="TeamID", how="left")
        
        # Fill NaNs for teams with less than w games
        season_stats[f"recent{w}_points_for"] = season_stats[f"recent{w}_points_for"].fillna(season_stats["season_points_for"])
        season_stats[f"recent{w}_points_against"] = season_stats[f"recent{w}_points_against"].fillna(season_stats["season_points_against"])
        season_stats[f"recent{w}_margin"] = season_stats[f"recent{w}_margin"].fillna(season_stats["season_margin"])

    # Add Elo
    elo_history, _ = calculate_elo(season_df)
    latest_elo = get_latest_elo(elo_history)
    season_stats = season_stats.merge(latest_elo[["TeamID", "Elo"]], on="TeamID", how="left")
    season_stats["Elo"] = season_stats["Elo"].fillna(1500)

    return season_stats

def attach_team_features(eval_df: pd.DataFrame, m_features: pd.DataFrame, w_features: pd.DataFrame, seeds_m: pd.DataFrame, seeds_w: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = eval_df.copy()

    # Attach seeds
    m_seeds_season = seeds_m[seeds_m["Season"] == out["Season"].iloc[0]][["TeamID", "SeedNum"]].rename(columns={"SeedNum": "SeedM"})
    w_seeds_season = seeds_w[seeds_w["Season"] == out["Season"].iloc[0]][["TeamID", "SeedNum"]].rename(columns={"SeedNum": "SeedW"})
    all_seeds = pd.concat([m_seeds_season, w_seeds_season])

    out = out.merge(all_seeds.rename(columns={"TeamID": "TeamIDLow", "SeedM": "SeedLow", "SeedW": "SeedLow"}), on=["TeamIDLow"], how="left")
    out = out.merge(all_seeds.rename(columns={"TeamID": "TeamIDHigh", "SeedM": "SeedHigh", "SeedW": "SeedHigh"}), on=["TeamIDHigh"], how="left")
    
    out["SeedLow"] = out["SeedLow"].fillna(cfg["fallback_seed"])
    out["SeedHigh"] = out["SeedHigh"].fillna(cfg["fallback_seed"])

    # Attach team features
    all_features = pd.concat([m_features, w_features])

    for col in all_features.columns:
        if col == "TeamID":
            continue
        out = out.merge(all_features[["TeamID", col]].rename(columns={
            "TeamID": "TeamIDLow", col: f"{col}_low"
        }), on="TeamIDLow", how="left")
        out = out.merge(all_features[["TeamID", col]].rename(columns={
            "TeamID": "TeamIDHigh", col: f"{col}_high"
        }), on="TeamIDHigh", how="left")

    # Calculate differences
    out["seed_diff"] = out["SeedLow"] - out["SeedHigh"]
    out["elo_diff"] = out["Elo_low"] - out["Elo_high"]
    
    # Season differences
    out["season_points_for_diff"] = out["season_points_for_low"] - out["season_points_for_high"]
    out["season_points_against_diff"] = out["season_points_against_low"] - out["season_points_against_high"]
    out["season_margin_diff"] = out["season_margin_low"] - out["season_margin_high"]
    out["season_win_pct_diff"] = out["season_win_pct_low"] - out["season_win_pct_high"]
    
    # Recent differences
    for w in [3, 5]:
        out[f"recent{w}_points_for_diff"] = out[f"recent{w}_points_for_low"] - out[f"recent{w}_points_for_high"]
        out[f"recent{w}_points_against_diff"] = out[f"recent{w}_points_against_low"] - out[f"recent{w}_points_against_high"]
        out[f"recent{w}_margin_diff"] = out[f"recent{w}_margin_low"] - out[f"recent{w}_margin_high"]

    return out

def make_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add interaction features
    df["offense_vs_defense_low"] = df["season_points_for_low"] - df["season_points_against_high"]
    df["offense_vs_defense_high"] = df["season_points_for_high"] - df["season_points_against_low"]
    df["matchup_diff"] = df["offense_vs_defense_low"] - df["offense_vs_defense_high"]
    
    return df
