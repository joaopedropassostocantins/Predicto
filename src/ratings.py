# src/ratings.py — Elo rating system v4.0
#
# Changes from v3:
#   - margin_factor: fixed baseline issue. Old formula log(m+1)/log(cap+1)
#     gave 0.20 for 1-pt wins; new formula (log(m+1)+1)/(log(cap+1)+1)
#     gives ~0.45, ensuring close wins still update meaningfully.
#   - Default k_factor: 25 → 20  (within literature range 16-22)
#   - Default carry_factor: 0.75 → 0.82  (more cross-season memory)
#   - Default margin_cap: 30 → 15  (NCAA basketball practical cap)
#   - New: compute_elo_season_features() returns per-team Elo-derived signals:
#       elo_end, elo_delta (momentum), elo_volatility (stability)
#   - All functions remain backward-compatible with existing call sites.
#
# References:
#   Elo (1978) "The Rating of Chessplayers"
#   FiveThirtyEight NBA/NCAAB Elo methodology
#   Hvattum & Arntzen (2010) "Using ELO ratings for match result prediction"

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _elo_expected(r_a: float, r_b: float) -> float:
    """Expected win probability for team A vs team B (standard logistic formula)."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def _margin_factor(margin: float, margin_cap: float) -> float:
    """
    Log-based margin multiplier for the K-factor update.

    FIXED in v4: adds +1 baseline so close wins still generate meaningful Elo updates.

    Old formula: log(m+1) / log(cap+1)
      • For m=1, cap=30: 0.20  ← too low; 1-pt wins barely mattered
      • For m=30 (cap):  1.00

    New formula: (log(m+1) + 1) / (log(cap+1) + 1)
      • For m=1,  cap=15: (log(2)+1)/(log(16)+1) ≈ 0.449  ← meaningful update
      • For m=5,  cap=15: (log(6)+1)/(log(16)+1) ≈ 0.740
      • For m=10, cap=15: (log(11)+1)/(log(16)+1) ≈ 0.901
      • For m=15 (cap):  1.000

    Rationale: A 1-point win in a tournament game carries real predictive signal
    for team quality. Downweighting it to 20% of K loses information. The new
    formula maps [0, cap] → [base ≈ 0.45, 1.0] while still saturating at cap.

    Parameters
    ----------
    margin : float   Absolute point difference (positive).
    margin_cap : float   Maximum margin considered (beyond this, factor = 1.0).
    """
    m = min(abs(float(margin)), margin_cap)
    return (np.log(m + 1.0) + 1.0) / (np.log(margin_cap + 1.0) + 1.0)


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
    elo_history : DataFrame  [Season, DayNum, TeamID, Elo, EloUpdate]
        EloUpdate column records the signed Elo change each game (useful for
        downstream momentum/volatility features).
    end_ratings : dict       {(season, team_id): final_elo}
    """
    if k_factor is None:
        k_factor = CONFIG["elo_k_factor"]
    if initial_rating is None:
        initial_rating = CONFIG["elo_initial_rating"]
    if use_margin is None:
        use_margin = CONFIG.get("elo_use_margin", True)
    if margin_cap is None:
        margin_cap = CONFIG.get("elo_margin_cap", 15.0)
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
        loser  = row.LTeamID

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

        delta_w = k_factor * mf * (1.0 - ew)
        delta_l = k_factor * mf * (0.0 - (1.0 - ew))

        ratings[key_w] = rw + delta_w
        ratings[key_l] = rl + delta_l

        # Record history with signed Elo update (needed for momentum features)
        history_rows.append({
            "Season":    season,
            "DayNum":    row.DayNum,
            "TeamID":    winner,
            "Elo":       ratings[key_w],
            "EloUpdate": delta_w,
        })
        history_rows.append({
            "Season":    season,
            "DayNum":    row.DayNum,
            "TeamID":    loser,
            "Elo":       ratings[key_l],
            "EloUpdate": delta_l,
        })

    elo_history = (
        pd.DataFrame(history_rows)
        if history_rows
        else pd.DataFrame(columns=["Season", "DayNum", "TeamID", "Elo", "EloUpdate"])
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
# NEW: Compute Elo-derived features per team per season
# ---------------------------------------------------------------------------

def compute_elo_season_features(
    elo_history_df: pd.DataFrame,
    n_delta: int = 5,
) -> pd.DataFrame:
    """
    Compute Elo momentum and volatility features from the history DataFrame.

    Parameters
    ----------
    elo_history_df : DataFrame  [Season, DayNum, TeamID, Elo, EloUpdate]
    n_delta : int
        Number of most recent games for elo_delta calculation.

    Returns
    -------
    DataFrame [Season, TeamID, elo_end, elo_delta, elo_volatility]
        elo_end        : final Elo rating for the season
        elo_delta      : Elo gained/lost over the last n_delta games (momentum)
        elo_volatility : std of per-game Elo updates (consistency of performance)
    """
    if len(elo_history_df) == 0:
        return pd.DataFrame(columns=["Season", "TeamID", "elo_end", "elo_delta", "elo_volatility"])

    rows = []
    for (season, team_id), grp in elo_history_df.groupby(["Season", "TeamID"]):
        grp_sorted = grp.sort_values("DayNum")
        elo_end = float(grp_sorted["Elo"].iloc[-1])

        # Momentum: Elo change over last n_delta games
        updates = grp_sorted["EloUpdate"].values
        if len(updates) >= n_delta:
            elo_delta = float(updates[-n_delta:].sum())
        else:
            elo_delta = float(updates.sum())

        # Volatility: std of per-game Elo updates (high std = inconsistent team)
        elo_volatility = float(np.std(updates)) if len(updates) > 1 else 0.0

        rows.append({
            "Season":        season,
            "TeamID":        team_id,
            "elo_end":       elo_end,
            "elo_delta":     elo_delta,
            "elo_volatility": elo_volatility,
        })

    return pd.DataFrame(rows)


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
    """
    if k_factor is None:
        k_factor = CONFIG["elo_k_factor"]
    if initial_rating is None:
        initial_rating = CONFIG["elo_initial_rating"]
    if carry_factor is None:
        carry_factor = CONFIG.get("elo_carry_factor", 0.82)
    if use_margin is None:
        use_margin = CONFIG.get("elo_use_margin", True)
    if margin_cap is None:
        margin_cap = CONFIG.get("elo_margin_cap", 15.0)

    seasons = sorted(games_all["Season"].unique())

    prev_end: dict = {}       # team_id → end-of-previous-season Elo
    starting_elo: dict = {}   # (season, team_id) → starting Elo

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


# ---------------------------------------------------------------------------
# Convenience: compute full Elo history + season features for one season
# ---------------------------------------------------------------------------

def compute_season_elo(
    season_games_long: pd.DataFrame,
    season: int,
    starting_elo: dict | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Compute Elo history + derived features (delta, volatility) for one season.

    Parameters
    ----------
    season_games_long : DataFrame
        Long-format regular-season games (Win==1 rows only used for Elo).
    starting_elo : dict, optional
        {(season, team_id): starting_elo} from precompute_starting_elo.
    cfg : dict, optional  CONFIG dict.

    Returns
    -------
    DataFrame [Season, TeamID, elo_end, elo_delta, elo_volatility]
    """
    from src.config import CONFIG as _CFG
    if cfg is None:
        cfg = _CFG

    season_init: dict = {}
    if starting_elo is not None:
        season_df = season_games_long[season_games_long["Season"] == season]
        all_teams = set(season_df["TeamID"]).union(set(season_df["OppTeamID"]))
        for team in all_teams:
            s_elo = starting_elo.get((season, team))
            if s_elo is not None:
                season_init[team] = s_elo

    # Build win-perspective games for Elo
    season_wins = (
        season_games_long[
            (season_games_long["Season"] == season) &
            (season_games_long["Win"] == 1)
        ][["Season", "DayNum", "TeamID", "OppTeamID", "Margin"]]
        .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    elo_history, _ = calculate_elo_from_games(
        season_wins,
        k_factor=cfg.get("elo_k_factor", 20.0),
        initial_rating=cfg.get("elo_initial_rating", 1500.0),
        use_margin=cfg.get("elo_use_margin", True),
        margin_cap=cfg.get("elo_margin_cap", 15.0),
        initial_ratings=season_init,
    )

    return compute_elo_season_features(elo_history)
