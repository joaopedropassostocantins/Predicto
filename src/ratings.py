# src/ratings.py

from __future__ import annotations

import pandas as pd
from src.config import CONFIG


def calculate_elo_from_games(games_df: pd.DataFrame, k_factor=None, initial_rating=None):
    if k_factor is None:
        k_factor = CONFIG["elo_k_factor"]
    if initial_rating is None:
        initial_rating = CONFIG["elo_initial_rating"]

    ratings = {}
    history_rows = []

    games = (
        games_df[["Season", "DayNum", "WTeamID", "LTeamID"]]
        .drop_duplicates()
        .sort_values(["Season", "DayNum", "WTeamID", "LTeamID"])
        .reset_index(drop=True)
    )

    for row in games.itertuples(index=False):
        season = row.Season
        winner = row.WTeamID
        loser = row.LTeamID

        if (season, winner) not in ratings:
            ratings[(season, winner)] = float(initial_rating)
        if (season, loser) not in ratings:
            ratings[(season, loser)] = float(initial_rating)

        rw = ratings[(season, winner)]
        rl = ratings[(season, loser)]

        ew = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        el = 1.0 - ew

        rw_new = rw + k_factor * (1.0 - ew)
        rl_new = rl + k_factor * (0.0 - el)

        ratings[(season, winner)] = rw_new
        ratings[(season, loser)] = rl_new

        history_rows.append(
            {"Season": season, "DayNum": row.DayNum, "TeamID": winner, "Elo": rw_new}
        )
        history_rows.append(
            {"Season": season, "DayNum": row.DayNum, "TeamID": loser, "Elo": rl_new}
        )

    elo_history = pd.DataFrame(history_rows)
    return elo_history, ratings


def get_latest_elo(elo_history_df: pd.DataFrame) -> pd.DataFrame:
    return (
        elo_history_df.sort_values(["Season", "TeamID", "DayNum"])
        .groupby(["Season", "TeamID"], as_index=False)
        .tail(1)[["Season", "TeamID", "Elo"]]
        .reset_index(drop=True)
    )
