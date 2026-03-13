# src/ratings.py — Elo rating system with cross-season carryover + margin weighting

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _elo_expected(r_a: float, r_b: float) -> float:
    """Expected win probability for team A vs team B."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def _margin_factor(margin: float, margin_cap: float) -> float:
    """
    Log-based margin multiplier for the K-factor update.

    Clamps the raw margin at `margin_cap`, then maps [0, cap] → [0, 1]
    via log(margin+1) / log(cap+1).  A blowout win gets a K-factor close
    to 1; a 1-point win gets a K-factor close to 0.
    """
    m = min(abs(float(margin)), margin_cap)
    return np.log(m + 1.0) / np.log(margin_cap + 1.0)


# ---------------------------------------------------------------------------
# Core single-season Elo calculation
# ---------------------------------------------------------------------------

def calculate_elo_from_games(
    games_df: pd.DataFrame,
    k_factor: float | None = None,
    initial_rating: float | None = None,
    use_margin: bool | None = None,
    margin_cap: float | None = None,
    initial_ratings: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute Elo ratings from a games frame.

    Parameters
    ----------
    games_df : DataFrame
        Must contain [Season, DayNum, WTeamID, LTeamID].
        Optional: [Margin] – winning margin (positive).  Required when
        use_margin=True.
    initial_ratings : dict, optional
        {team_id: starting_elo}.  Teams not present start at initial_rating.

    Returns
    -------
    elo_history : DataFrame  [Season, DayNum, TeamID, Elo]
    end_ratings : dict       {(season, team_id): final_elo}
    """
    if k_factor is None:
        k_factor = CONFIG["elo_k_factor"]
    if initial_rating is None:
        initial_rating = CONFIG["elo_initial_rating"]
    if use_margin is None:
        use_margin = CONFIG.get("elo_use_margin", False)
    if margin_cap is None:
        margin_cap = CONFIG.get("elo_margin_cap", 30.0)
    if initial_ratings is None:
        initial_ratings = {}

    req_cols = ["Season", "DayNum", "WTeamID", "LTeamID"]
    extra = ["Margin"] if (use_margin and "Margin" in games_df.columns) else []
    cols = req_cols + extra

    games = (
        games_df[cols]
        .drop_duplicates(subset=req_cols)
        .sort_values(["Season", "DayNum", "WTeamID", "LTeamID"])
        .reset_index(drop=True)
    )

    has_margin = "Margin" in games.columns
    ratings: dict = {}
    history_rows: list[dict] = []

    for row in games.itertuples(index=False):
        season = row.Season
        winner = row.WTeamID
        loser = row.LTeamID

        key_w = (season, winner)
        key_l = (season, loser)

        if key_w not in ratings:
            ratings[key_w] = float(initial_ratings.get(winner, initial_rating))
        if key_l not in ratings:
            ratings[key_l] = float(initial_ratings.get(loser, initial_rating))

        rw = ratings[key_w]
        rl = ratings[key_l]
        ew = _elo_expected(rw, rl)

        if use_margin and has_margin:
            mf = _margin_factor(float(row.Margin), margin_cap)
        else:
            mf = 1.0

        ratings[key_w] = rw + k_factor * mf * (1.0 - ew)
        ratings[key_l] = rl + k_factor * mf * (0.0 - (1.0 - ew))

        history_rows.append({"Season": season, "DayNum": row.DayNum,
                              "TeamID": winner, "Elo": ratings[key_w]})
        history_rows.append({"Season": season, "DayNum": row.DayNum,
                              "TeamID": loser,  "Elo": ratings[key_l]})

    elo_history = (
        pd.DataFrame(history_rows)
        if history_rows
        else pd.DataFrame(columns=["Season", "DayNum", "TeamID", "Elo"])
    )
    return elo_history, ratings


def get_latest_elo(elo_history_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (Season, TeamID) — the final Elo for that season."""
    if len(elo_history_df) == 0:
        return pd.DataFrame(columns=["Season", "TeamID", "Elo"])
    return (
        elo_history_df
        .sort_values(["Season", "TeamID", "DayNum"])
        .groupby(["Season", "TeamID"], as_index=False)
        .last()[["Season", "TeamID", "Elo"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Cross-season Elo with carryover
# ---------------------------------------------------------------------------

def precompute_starting_elo(
    games_all: pd.DataFrame,
    k_factor: float | None = None,
    initial_rating: float | None = None,
    carry_factor: float | None = None,
    use_margin: bool | None = None,
    margin_cap: float | None = None,
) -> dict:
    """
    Process all seasons in chronological order, carrying Elo between seasons.

    At the start of season S, each team's starting Elo is:
        start_elo(S) = carry_factor * end_elo(S-1)
                       + (1 - carry_factor) * initial_rating

    Teams not seen in the previous season start at initial_rating.

    Parameters
    ----------
    games_all : DataFrame
        Regular-season games for ALL seasons.  Must have
        [Season, DayNum, WTeamID, LTeamID].  Optionally [Margin].

    Returns
    -------
    starting_elo : dict  {(season, team_id): starting_elo_for_that_season}
        Useful as the `initial_ratings` argument to build_team_features.
    """
    if k_factor is None:
        k_factor = CONFIG["elo_k_factor"]
    if initial_rating is None:
        initial_rating = CONFIG["elo_initial_rating"]
    if carry_factor is None:
        carry_factor = CONFIG.get("elo_carry_factor", 0.75)
    if use_margin is None:
        use_margin = CONFIG.get("elo_use_margin", False)
    if margin_cap is None:
        margin_cap = CONFIG.get("elo_margin_cap", 30.0)

    seasons = sorted(games_all["Season"].unique())

    prev_end: dict = {}          # team_id → end-of-previous-season Elo
    starting_elo: dict = {}      # (season, team_id) → starting Elo

    for season in seasons:
        season_games = games_all[games_all["Season"] == season].copy()

        # Build per-team starting ratings for this season
        season_start: dict = {
            team: carry_factor * elo + (1.0 - carry_factor) * initial_rating
            for team, elo in prev_end.items()
        }

        # Record starting Elo for each known team
        all_teams = set(season_games["WTeamID"]).union(set(season_games["LTeamID"]))
        for team in all_teams:
            starting_elo[(season, team)] = season_start.get(team, initial_rating)

        _, end_ratings = calculate_elo_from_games(
            season_games,
            k_factor=k_factor,
            initial_rating=initial_rating,
            use_margin=use_margin,
            margin_cap=margin_cap,
            initial_ratings=season_start,
        )

        # Update prev_end with this season's final ratings
        for (s, team), elo in end_ratings.items():
            if s == season:
                prev_end[team] = elo

    return starting_elo
