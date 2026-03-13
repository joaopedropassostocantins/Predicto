# src/features.py — Feature engineering for March ML Mania 2026 — v3.0

from __future__ import annotations

import numpy as np
import pandas as pd

from src.poisson import build_window_stats, add_poisson_matchup_features
from src.ratings import calculate_elo_from_games, get_latest_elo


# ---------------------------------------------------------------------------
# Team-level feature building (one season at a time)
# ---------------------------------------------------------------------------

def build_team_features(
    games_df: pd.DataFrame,
    season: int,
    recent_games_window: int,
    alpha_ci: float,
    starting_elo: dict | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Build a team-level feature frame for `season`.

    Parameters
    ----------
    games_df : DataFrame
        Long-format regular season games (both win and loss perspective).
        Must have: Season, DayNum, TeamID, OppTeamID, PointsFor,
                   PointsAgainst, Margin, Win.
    starting_elo : dict, optional
        {team_id: starting_elo} for cross-season Elo carryover.
        If None, all teams start at elo_initial_rating (1500).
    cfg : dict, optional
        CONFIG dict.  Used for fallback values and Poisson settings.
    """
    from src.config import CONFIG as _CFG
    if cfg is None:
        cfg = _CFG

    season_df = games_df[games_df["Season"] == season].copy()

    # ── 1. Season aggregates ───────────────────────────────────────────────
    season_stats = (
        season_df.groupby("TeamID")
        .agg(
            season_points_for     = ("PointsFor",  "mean"),
            season_points_against = ("PointsAgainst", "mean"),
            season_margin         = ("Margin",     "mean"),
            season_wins           = ("Win",         "sum"),
            season_games_played   = ("TeamID",      "count"),
            season_points_for_std = ("PointsFor",  "std"),
            season_margin_std     = ("Margin",      "std"),
        )
        .reset_index()
    )
    season_stats["Season"] = season
    season_stats["season_win_pct"] = (
        season_stats["season_wins"] / season_stats["season_games_played"]
    )

    # Fill std NaN for single-game seasons
    season_stats["season_points_for_std"] = season_stats["season_points_for_std"].fillna(12.0)
    season_stats["season_margin_std"] = season_stats["season_margin_std"].fillna(12.0)

    # ── 2. Recent form (last 3 and last 5 games) ───────────────────────────
    # Sort once, then use groupby.tail(w) — works in all pandas versions.
    season_sorted = season_df.sort_values(["TeamID", "DayNum"])
    for w in [3, 5]:
        last_w = season_sorted.groupby("TeamID", sort=False).tail(w)
        recent_form = (
            last_w.groupby("TeamID")
            .agg(
                **{
                    f"recent{w}_points_for":     ("PointsFor",     "mean"),
                    f"recent{w}_points_against": ("PointsAgainst", "mean"),
                    f"recent{w}_margin":         ("Margin",        "mean"),
                }
            )
            .reset_index()
        )
        season_stats = season_stats.merge(recent_form, on="TeamID", how="left")
        season_stats[f"recent{w}_points_for"] = season_stats[f"recent{w}_points_for"].fillna(
            season_stats["season_points_for"]
        )
        season_stats[f"recent{w}_points_against"] = season_stats[f"recent{w}_points_against"].fillna(
            season_stats["season_points_against"]
        )
        season_stats[f"recent{w}_margin"] = season_stats[f"recent{w}_margin"].fillna(
            season_stats["season_margin"]
        )

    # ── 3. Season trajectory: late-season improvement ─────────────────────
    # Compare average margin of first 8 games vs last 8 games.
    # Positive value → improving into the tournament.
    n_traj = 8
    traj_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        g = grp.sort_values("DayNum")
        if len(g) < n_traj:
            traj = 0.0
        else:
            early = g.head(n_traj)["Margin"].mean()
            late  = g.tail(n_traj)["Margin"].mean()
            traj  = float(late - early)
        traj_rows.append({"TeamID": team_id, "season_trajectory": traj})
    traj_df = pd.DataFrame(traj_rows)
    season_stats = season_stats.merge(traj_df, on="TeamID", how="left")
    season_stats["season_trajectory"] = season_stats["season_trajectory"].fillna(0.0)

    # ── 4. Blowout rate (won by >15) ───────────────────────────────────────
    season_df["_blowout"] = (season_df["Margin"] > 15).astype(float)
    blowout_df = (
        season_df.groupby("TeamID")["_blowout"]
        .mean()
        .reset_index()
        .rename(columns={"_blowout": "blowout_pct"})
    )
    season_df = season_df.drop(columns=["_blowout"])
    season_stats = season_stats.merge(blowout_df, on="TeamID", how="left")
    season_stats["blowout_pct"] = season_stats["blowout_pct"].fillna(0.0)

    # ── 5. Close-game win rate (games decided by <5 pts) ──────────────────
    close_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        close = grp[grp["Margin"].abs() < 5]
        wpct  = float(close["Win"].mean()) if len(close) > 0 else 0.5
        close_rows.append({"TeamID": team_id, "close_game_win_pct": wpct})
    close_df = pd.DataFrame(close_rows)
    season_stats = season_stats.merge(close_df, on="TeamID", how="left")
    season_stats["close_game_win_pct"] = season_stats["close_game_win_pct"].fillna(0.5)

    # ── 6. Strength of schedule proxy ─────────────────────────────────────
    opp_strength = (
        season_df.groupby("OppTeamID")
        .agg(
            opp_avg_margin = ("Margin", "mean"),
            opp_win_rate   = ("Win",    "mean"),
        )
        .reset_index()
        .rename(columns={"OppTeamID": "TeamID"})
    )
    sos = (
        season_df.merge(
            opp_strength.rename(columns={
                "TeamID":        "OppTeamID",
                "opp_avg_margin": "opp_margin_proxy",
                "opp_win_rate":   "opp_win_proxy",
            }),
            on="OppTeamID",
            how="left",
        )
        .groupby("TeamID")
        .agg(
            sos_margin   = ("opp_margin_proxy", "mean"),
            sos_win_rate = ("opp_win_proxy",    "mean"),
        )
        .reset_index()
    )
    season_stats = season_stats.merge(sos, on="TeamID", how="left")
    season_stats["sos_margin"]   = season_stats["sos_margin"].fillna(0.0)
    season_stats["sos_win_rate"] = season_stats["sos_win_rate"].fillna(0.5)

    # ── 7. Quality win percentage ──────────────────────────────────────────
    # Win rate against teams with above-median season margin.
    median_margin = season_stats["season_margin"].median()
    quality_teams = set(
        season_stats.loc[season_stats["season_margin"] > median_margin, "TeamID"]
    )
    qwin_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        vs_q = grp[grp["OppTeamID"].isin(quality_teams)]
        wpct = float(vs_q["Win"].mean()) if len(vs_q) > 0 else 0.5
        qwin_rows.append({"TeamID": team_id, "quality_win_pct": wpct})
    qwin_df = pd.DataFrame(qwin_rows)
    season_stats = season_stats.merge(qwin_df, on="TeamID", how="left")
    season_stats["quality_win_pct"] = season_stats["quality_win_pct"].fillna(0.5)

    # ── 8. Quality proxy and rank proxy ───────────────────────────────────
    season_stats["quality_proxy"] = (
        0.55 * season_stats["season_margin"].fillna(0.0)
        + 25.0 * season_stats["season_win_pct"].fillna(0.5)
        + 0.25 * season_stats["sos_margin"].fillna(0.0)
    )
    season_stats["rank_proxy"] = season_stats["quality_proxy"].rank(
        ascending=False, method="average"
    )

    # ── 9. Elo (cross-season aware) ────────────────────────────────────────
    base_games = (
        season_df[season_df["Win"] == 1][["Season", "DayNum", "TeamID", "OppTeamID", "Margin"]]
        .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Resolve per-team starting Elo from the precomputed cross-season dict
    season_init: dict = {}
    if starting_elo is not None:
        for team_id in season_stats["TeamID"]:
            s_elo = starting_elo.get((season, team_id))
            if s_elo is not None:
                season_init[team_id] = s_elo

    use_margin = cfg.get("elo_use_margin", False)
    margin_cap = cfg.get("elo_margin_cap", 30.0)
    k_factor   = cfg.get("elo_k_factor", 25.0)
    init_elo   = cfg.get("elo_initial_rating", 1500.0)

    elo_history, _ = calculate_elo_from_games(
        base_games,
        k_factor=k_factor,
        initial_rating=init_elo,
        use_margin=use_margin,
        margin_cap=margin_cap,
        initial_ratings=season_init,
    )
    latest_elo = get_latest_elo(elo_history)
    season_stats = season_stats.merge(
        latest_elo[["Season", "TeamID", "Elo"]], on=["Season", "TeamID"], how="left"
    )
    season_stats["Elo"] = season_stats["Elo"].fillna(float(init_elo))

    # ── 10. Poisson lambda windows ────────────────────────────────────────
    poisson_shrinkage = cfg.get("poisson_shrinkage", 0.0)
    alpha_ci_val      = alpha_ci

    # Compute league-average lambdas for shrinkage
    league_avg_for     = float(season_df["PointsFor"].mean())
    league_avg_against = float(season_df["PointsAgainst"].mean())

    poisson_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        row = {"Season": season, "TeamID": team_id}
        for window in [3, 5, "season"]:
            stats = build_window_stats(
                grp, window, alpha=alpha_ci_val,
                shrinkage=poisson_shrinkage,
                league_avg_for=league_avg_for,
                league_avg_against=league_avg_against,
            )
            prefix = f"recent{window}" if window != "season" else "season"
            row[f"{prefix}_lambda_for"]          = stats["lambda_for"]
            row[f"{prefix}_lambda_for_ci_low"]   = stats["lambda_for_ci_low"]
            row[f"{prefix}_lambda_for_ci_high"]  = stats["lambda_for_ci_high"]
            row[f"{prefix}_lambda_against"]      = stats["lambda_against"]
            row[f"{prefix}_lambda_against_ci_low"]  = stats["lambda_against_ci_low"]
            row[f"{prefix}_lambda_against_ci_high"] = stats["lambda_against_ci_high"]
        poisson_rows.append(row)

    poisson_df = pd.DataFrame(poisson_rows)
    season_stats = season_stats.merge(poisson_df, on=["Season", "TeamID"], how="left")

    return season_stats


# ---------------------------------------------------------------------------
# Attach team features to matchup rows
# ---------------------------------------------------------------------------

def attach_team_features(
    eval_df: pd.DataFrame,
    m_features: pd.DataFrame,
    w_features: pd.DataFrame,
    seeds_m: pd.DataFrame,
    seeds_w: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    out = eval_df.copy()

    all_features = pd.concat([m_features, w_features], ignore_index=True)
    all_seeds    = pd.concat([seeds_m, seeds_w],        ignore_index=True)

    # Seeds
    out = out.merge(
        all_seeds.rename(columns={"TeamID": "TeamIDLow",  "SeedNum": "SeedLow"})
                 [["Season", "TeamIDLow", "SeedLow"]],
        on=["Season", "TeamIDLow"], how="left",
    )
    out = out.merge(
        all_seeds.rename(columns={"TeamID": "TeamIDHigh", "SeedNum": "SeedHigh"})
                 [["Season", "TeamIDHigh", "SeedHigh"]],
        on=["Season", "TeamIDHigh"], how="left",
    )
    out["SeedLow"]  = out["SeedLow"].fillna(cfg["fallback_seed"])
    out["SeedHigh"] = out["SeedHigh"].fillna(cfg["fallback_seed"])

    # Team features (low / high)
    feature_cols = [c for c in all_features.columns if c not in ["Season", "TeamID"]]
    low_map  = {"TeamID": "TeamIDLow"}
    high_map = {"TeamID": "TeamIDHigh"}
    low_map.update( {c: f"{c}_low"  for c in feature_cols})
    high_map.update({c: f"{c}_high" for c in feature_cols})

    out = out.merge(all_features.rename(columns=low_map),  on=["Season", "TeamIDLow"],  how="left")
    out = out.merge(all_features.rename(columns=high_map), on=["Season", "TeamIDHigh"], how="left")

    # Fallback numeric values
    fallback_numeric = {
        "season_points_for":     cfg["fallback_points_for"],
        "season_points_against": cfg["fallback_points_against"],
        "season_margin":         0.0,
        "season_win_pct":        0.5,
        "season_games_played":   0.0,
        "recent3_points_for":    cfg["fallback_points_for"],
        "recent3_points_against":cfg["fallback_points_against"],
        "recent3_margin":        0.0,
        "recent5_points_for":    cfg["fallback_points_for"],
        "recent5_points_against":cfg["fallback_points_against"],
        "recent5_margin":        0.0,
        "Elo":                   cfg["fallback_elo"],
        "sos_margin":            0.0,
        "sos_win_rate":          0.5,
        "quality_proxy":         0.0,
        "rank_proxy":            0.0,
        "season_points_for_std": 12.0,
        "season_margin_std":     12.0,
        "season_trajectory":     0.0,
        "blowout_pct":           0.0,
        "close_game_win_pct":    0.5,
        "quality_win_pct":       0.5,
        "recent3_lambda_for":    cfg["fallback_points_for"],
        "recent3_lambda_against":cfg["fallback_points_against"],
        "recent5_lambda_for":    cfg["fallback_points_for"],
        "recent5_lambda_against":cfg["fallback_points_against"],
        "season_lambda_for":     cfg["fallback_points_for"],
        "season_lambda_against": cfg["fallback_points_against"],
    }
    for base_col, fill_val in fallback_numeric.items():
        for suffix in ("_low", "_high"):
            col = f"{base_col}{suffix}"
            if col in out.columns:
                out[col] = out[col].fillna(fill_val)

    return out


# ---------------------------------------------------------------------------
# Matchup-level feature differences
# ---------------------------------------------------------------------------

def make_matchup_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Create matchup-level difference features.

    All differences are oriented as (Low − High) so that a positive value
    means the Low-ID team has the advantage.  Model target is ActualLowWin=1.

    Parameters
    ----------
    cfg : dict, optional
        CONFIG dict.  Used for Poisson blend weights and max_points_poisson.
        Falls back to CONFIG defaults if None.
    """
    from src.config import CONFIG as _CFG
    if cfg is None:
        cfg = _CFG

    out = df.copy()

    # Seed: positive → Low has better (lower) seed
    out["seed_diff"] = out["SeedHigh"] - out["SeedLow"]

    # Elo: positive → Low has higher Elo
    out["elo_diff"] = out["Elo_low"] - out["Elo_high"]

    # Season stats
    out["season_points_for_diff"]     = out["season_points_for_low"]     - out["season_points_for_high"]
    out["season_points_against_diff"] = out["season_points_against_low"] - out["season_points_against_high"]
    out["season_margin_diff"]         = out["season_margin_low"]         - out["season_margin_high"]
    out["season_win_pct_diff"]        = out["season_win_pct_low"]        - out["season_win_pct_high"]

    # Recent 3
    out["recent3_points_for_diff"]    = out["recent3_points_for_low"]    - out["recent3_points_for_high"]
    out["recent3_points_against_diff"]= out["recent3_points_against_low"]- out["recent3_points_against_high"]
    out["recent3_margin_diff"]        = out["recent3_margin_low"]        - out["recent3_margin_high"]

    # Recent 5
    out["recent5_points_for_diff"]    = out["recent5_points_for_low"]    - out["recent5_points_for_high"]
    out["recent5_points_against_diff"]= out["recent5_points_against_low"]- out["recent5_points_against_high"]
    out["recent5_margin_diff"]        = out["recent5_margin_low"]        - out["recent5_margin_high"]

    # Offense vs opponent defense
    out["offense_vs_defense_low"]  = out["season_points_for_low"]  - out["season_points_against_high"]
    out["offense_vs_defense_high"] = out["season_points_for_high"] - out["season_points_against_low"]
    out["matchup_diff"] = out["offense_vs_defense_low"] - out["offense_vs_defense_high"]

    # Strength signals
    out["sos_diff"]         = out["sos_margin_low"]       - out["sos_margin_high"]
    out["quality_diff"]     = out["quality_proxy_low"]    - out["quality_proxy_high"]
    out["rank_diff_signed"] = out["rank_proxy_high"]      - out["rank_proxy_low"]  # high rank_proxy → worse

    # Consistency: high scoring variance in opponent → harder to predict → small edge
    out["consistency_edge"] = (
        out["season_points_for_std_high"] - out["season_points_for_std_low"]
    )

    # ── v3 new features ────────────────────────────────────────────────────
    out["season_trajectory_diff"]  = out["season_trajectory_low"]   - out["season_trajectory_high"]
    out["quality_win_pct_diff"]    = out["quality_win_pct_low"]      - out["quality_win_pct_high"]
    out["blowout_pct_diff"]        = out["blowout_pct_low"]          - out["blowout_pct_high"]
    out["close_game_win_pct_diff"] = out["close_game_win_pct_low"]   - out["close_game_win_pct_high"]

    # ── Poisson matchup features ───────────────────────────────────────────
    poisson_cfg = {
        "poisson_blend_weights": cfg.get(
            "poisson_blend_weights", {"recent3": 0.35, "recent5": 0.30, "season": 0.35}
        ),
        "max_points_poisson": cfg.get("max_points_poisson", 220),
    }
    out = add_poisson_matchup_features(out, cfg=poisson_cfg)

    return out
