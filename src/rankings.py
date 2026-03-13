# src/rankings.py

from __future__ import annotations

import os
import pandas as pd


def load_massey_ordinals(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "MMasseyOrdinals.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"])

    df = pd.read_csv(path)

    # Menor rank = melhor
    return df[["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]].copy()


def build_daily_massey_feature(
    massey_df: pd.DataFrame,
    system_priority=None,
) -> pd.DataFrame:
    if massey_df.empty:
        return pd.DataFrame(columns=["Season", "DayNum", "TeamID", "massey_rank"])

    if system_priority is None:
        system_priority = ["POM", "SAG", "MOR", "DOL", "WOL", "RTH", "COL"]

    temp = massey_df.copy()
    priority_map = {name: i for i, name in enumerate(system_priority)}
    temp["system_priority"] = temp["SystemName"].map(priority_map).fillna(len(system_priority) + 1)

    temp = temp.sort_values(
        ["Season", "RankingDayNum", "TeamID", "system_priority", "OrdinalRank"]
    )

    best = temp.groupby(["Season", "RankingDayNum", "TeamID"], as_index=False).first()
    best = best.rename(columns={"RankingDayNum": "DayNum", "OrdinalRank": "massey_rank"})

    return best[["Season", "DayNum", "TeamID", "massey_rank"]].copy()


def attach_pre_game_massey_rank(
    games_df: pd.DataFrame,
    daily_rank_df: pd.DataFrame,
    team_col: str,
    day_col: str = "DayNum",
) -> pd.Series:
    if daily_rank_df.empty:
        return pd.Series([pd.NA] * len(games_df), index=games_df.index, dtype="float")

    left = games_df[["Season", day_col, team_col]].copy()
    left = left.rename(columns={day_col: "game_day", team_col: "TeamID"})
    left["_row_id"] = range(len(left))

    right = daily_rank_df.rename(columns={"DayNum": "rank_day"}).copy()

    merged = left.merge(right, on=["Season", "TeamID"], how="left")
    merged = merged[merged["rank_day"] < merged["game_day"]].copy()

    if merged.empty:
        return pd.Series([pd.NA] * len(games_df), index=games_df.index, dtype="float")

    merged = merged.sort_values(["_row_id", "rank_day"])
    latest = merged.groupby("_row_id", as_index=False).tail(1)

    out = pd.Series([pd.NA] * len(games_df), index=games_df.index, dtype="float")
    out.loc[latest["_row_id"].values] = latest["massey_rank"].values
    return out
