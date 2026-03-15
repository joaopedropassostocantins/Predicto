# src/features.py — Feature engineering v4.1
#
# Changes from v4.0:
#   - Added offensive/defensive efficiency proxy features.
#     off_eff ≈ points_for / games (already captured via season_points_for)
#     net_eff ≈ margin (already captured)
#     NEW: off_eff_diff = season_points_for_low - season_points_against_high
#          (directly captures: how much team A scores vs how much B allows)
#     NEW: def_eff_diff = season_points_against_high - season_points_for_low
#          (mirror — captures defensive matchup strength)
#     These are now added as matchup-level features in make_matchup_features().
#   - Feature added to feature_cols in config.py: off_eff_diff, def_eff_diff,
#     net_eff_diff (= off_eff_diff - def_eff_diff = matchup_diff, but explicit)
#
# Changes from v3:
#   - Added EWMA (alpha=0.20) for point margin
#   - Added elo_delta and elo_volatility
#   - league_avg_score injected for multiplicative Poisson
#   - Better SoS (two-pass), quality_proxy normalised by n_teams
#   - Adaptive Poisson shrinkage k/(k+n)
#   - All features strictly pre-game (no post-game leakage)
#
# Feature causal ordering:
#   1. Season aggregates (all games up to pre-tournament)
#   2. Recent form (last 3, last 5 games)
#   3. Trajectory (late vs early season)
#   4. Poisson lambdas (attack/defense rates, windowed)
#   5. Elo rating (cross-season carry-over)
#   6. Elo momentum/volatility (from Elo history)
#   7. EWMA margin (recency-weighted form)
#   8. Strength of schedule, quality wins (opponent-based)
#   9. Offensive/defensive efficiency proxy (NEW v4.1)

from __future__ import annotations

import numpy as np
import pandas as pd

from src.poisson import build_window_stats, add_poisson_matchup_features
from src.ratings import (
    calculate_elo_from_games,
    get_latest_elo,
    compute_elo_season_features,
)


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

    All features are computed using only regular-season data available before
    the tournament — no future information is used.

    Parameters
    ----------
    games_df : DataFrame
        Long-format regular season games (both win and loss perspective).
        Must have: Season, DayNum, TeamID, OppTeamID, PointsFor,
                   PointsAgainst, Margin, Win.
    starting_elo : dict, optional
        {(season, team_id): starting_elo} for cross-season Elo carryover.
    cfg : dict, optional  CONFIG dict.
    """
    from src.config import CONFIG as _CFG
    if cfg is None:
        cfg = _CFG

    season_df = games_df[games_df["Season"] == season].copy()

    # ── 1. Season aggregates ───────────────────────────────────────────────
    season_stats = (
        season_df.groupby("TeamID")
        .agg(
            season_points_for     = ("PointsFor",     "mean"),
            season_points_against = ("PointsAgainst", "mean"),
            season_margin         = ("Margin",        "mean"),
            season_wins           = ("Win",            "sum"),
            season_games_played   = ("TeamID",         "count"),
            season_points_for_std = ("PointsFor",     "std"),
            season_margin_std     = ("Margin",         "std"),
        )
        .reset_index()
    )
    season_stats["Season"] = season
    season_stats["season_win_pct"] = (
        season_stats["season_wins"] / season_stats["season_games_played"]
    )
    # Fill std NaN for single-game scenarios
    season_stats["season_points_for_std"] = season_stats["season_points_for_std"].fillna(12.0)
    season_stats["season_margin_std"]     = season_stats["season_margin_std"].fillna(12.0)

    # League averages (needed for Poisson shrinkage and multiplicative model)
    league_avg_for     = float(season_df["PointsFor"].mean())
    league_avg_against = float(season_df["PointsAgainst"].mean())
    # Store for injection into matchup frame
    season_stats["league_avg_score"] = league_avg_for

    # ── 2. Recent form (last 3 and last 5 games) ───────────────────────────
    season_sorted = season_df.sort_values(["TeamID", "DayNum"])
    for w in [3, 5]:
        last_w = season_sorted.groupby("TeamID", sort=False).tail(w)
        recent_form = (
            last_w.groupby("TeamID")
            .agg(**{
                f"recent{w}_points_for":     ("PointsFor",     "mean"),
                f"recent{w}_points_against": ("PointsAgainst", "mean"),
                f"recent{w}_margin":         ("Margin",        "mean"),
            })
            .reset_index()
        )
        season_stats = season_stats.merge(recent_form, on="TeamID", how="left")
        # Fill with season averages when fewer than w games available
        season_stats[f"recent{w}_points_for"] = (
            season_stats[f"recent{w}_points_for"].fillna(season_stats["season_points_for"])
        )
        season_stats[f"recent{w}_points_against"] = (
            season_stats[f"recent{w}_points_against"].fillna(season_stats["season_points_against"])
        )
        season_stats[f"recent{w}_margin"] = (
            season_stats[f"recent{w}_margin"].fillna(season_stats["season_margin"])
        )

    # ── 3. EWMA margin (NEW v4) ────────────────────────────────────────────
    # Exponentially weighted moving average with alpha=0.20 (half-life ≈ 3 games).
    # Unlike fixed windows, EWMA naturally handles varying game counts.
    ewma_alpha = 0.20
    ewma_rows  = []
    for team_id, grp in season_df.groupby("TeamID"):
        g = grp.sort_values("DayNum")
        margins = g["Margin"].values.astype(float)
        if len(margins) == 0:
            ewma_val = 0.0
        elif len(margins) == 1:
            ewma_val = float(margins[0])
        else:
            # Manual EWMA: older observations downweighted by (1-alpha)^k
            ewma_val = float(margins[0])
            for m in margins[1:]:
                ewma_val = ewma_alpha * m + (1.0 - ewma_alpha) * ewma_val
        ewma_rows.append({"TeamID": team_id, "ewma_margin": ewma_val})
    ewma_df = pd.DataFrame(ewma_rows) if ewma_rows else pd.DataFrame(columns=["TeamID", "ewma_margin"])
    season_stats = season_stats.merge(ewma_df, on="TeamID", how="left")
    season_stats["ewma_margin"] = season_stats["ewma_margin"].fillna(0.0)

    # ── 4. Season trajectory: late vs early season improvement ─────────────
    n_traj = 8
    traj_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        g = grp.sort_values("DayNum")
        if len(g) < n_traj:
            traj = 0.0
        else:
            early = float(g.head(n_traj)["Margin"].mean())
            late  = float(g.tail(n_traj)["Margin"].mean())
            traj  = late - early
        traj_rows.append({"TeamID": team_id, "season_trajectory": traj})
    traj_df = pd.DataFrame(traj_rows) if traj_rows else pd.DataFrame(columns=["TeamID", "season_trajectory"])
    season_stats = season_stats.merge(traj_df, on="TeamID", how="left")
    season_stats["season_trajectory"] = season_stats["season_trajectory"].fillna(0.0)

    # ── 5. Blowout rate (won by >15) ───────────────────────────────────────
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

    # ── 6. Close-game win rate (decided by <5 pts) ─────────────────────────
    close_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        close = grp[grp["Margin"].abs() < 5]
        wpct  = float(close["Win"].mean()) if len(close) > 0 else 0.5
        close_rows.append({"TeamID": team_id, "close_game_win_pct": wpct})
    close_df = pd.DataFrame(close_rows) if close_rows else pd.DataFrame(columns=["TeamID", "close_game_win_pct"])
    season_stats = season_stats.merge(close_df, on="TeamID", how="left")
    season_stats["close_game_win_pct"] = season_stats["close_game_win_pct"].fillna(0.5)

    # ── 7. Strength of Schedule (two-pass to avoid circularity) ───────────
    # Pass 1: compute each team's overall quality proxy (margin + win_pct)
    # Pass 2: SoS = average quality of opponents faced
    # This avoids using the team's own SoS in computing their opponents' SoS.
    basic_quality = (
        season_stats[["TeamID", "season_margin", "season_win_pct"]]
        .set_index("TeamID")
    )
    basic_quality["basic_q"] = (
        0.6 * basic_quality["season_margin"].fillna(0.0)
        + 20.0 * basic_quality["season_win_pct"].fillna(0.5)
    )

    sos_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        opp_ids = grp["OppTeamID"].values
        opp_quality = basic_quality.loc[
            basic_quality.index.isin(opp_ids), "basic_q"
        ].mean()
        sos_rows.append({
            "TeamID":     team_id,
            "sos_margin": float(opp_quality) if not np.isnan(opp_quality) else 0.0,
        })
    sos_df = pd.DataFrame(sos_rows) if sos_rows else pd.DataFrame(columns=["TeamID", "sos_margin"])
    # Also compute opponent win rate
    opp_win_df = (
        season_df.merge(
            season_stats[["TeamID", "season_win_pct"]].rename(
                columns={"TeamID": "OppTeamID", "season_win_pct": "opp_win_rate"}
            ),
            on="OppTeamID", how="left",
        )
        .groupby("TeamID")["opp_win_rate"]
        .mean()
        .reset_index()
        .rename(columns={"opp_win_rate": "sos_win_rate"})
    )
    season_stats = season_stats.merge(sos_df, on="TeamID", how="left")
    season_stats = season_stats.merge(opp_win_df, on="TeamID", how="left")
    season_stats["sos_margin"]   = season_stats["sos_margin"].fillna(0.0)
    season_stats["sos_win_rate"] = season_stats["sos_win_rate"].fillna(0.5)

    # ── 8. Quality win percentage ──────────────────────────────────────────
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

    # ── 9. Quality proxy and rank proxy ────────────────────────────────────
    # quality_proxy: composite strength signal (used for rank_proxy and SoS)
    # Normalised by n_teams for cross-season comparability.
    n_teams = max(len(season_stats), 1)
    season_stats["quality_proxy"] = (
        0.55 * season_stats["season_margin"].fillna(0.0)
        + 25.0 * season_stats["season_win_pct"].fillna(0.5)
        + 0.25 * season_stats["sos_margin"].fillna(0.0)
    )
    season_stats["rank_proxy"] = (
        season_stats["quality_proxy"].rank(ascending=False, method="average") / n_teams
    )

    # ── 10. Elo (cross-season aware) with momentum/volatility ──────────────
    base_games = (
        season_df[season_df["Win"] == 1][["Season", "DayNum", "TeamID", "OppTeamID", "Margin"]]
        .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    season_init: dict = {}
    if starting_elo is not None:
        for team_id in season_stats["TeamID"]:
            s_elo = starting_elo.get((season, team_id))
            if s_elo is not None:
                season_init[team_id] = s_elo

    k_factor   = cfg.get("elo_k_factor", 20.0)
    init_elo   = cfg.get("elo_initial_rating", 1500.0)
    use_margin = cfg.get("elo_use_margin", True)
    margin_cap = cfg.get("elo_margin_cap", 15.0)

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
        latest_elo[["Season", "TeamID", "Elo"]],
        on=["Season", "TeamID"],
        how="left",
    )
    season_stats["Elo"] = season_stats["Elo"].fillna(float(init_elo))

    # Elo momentum + volatility (NEW v4)
    elo_features = compute_elo_season_features(elo_history, n_delta=5)
    if len(elo_features) > 0:
        season_stats = season_stats.merge(
            elo_features[["Season", "TeamID", "elo_delta", "elo_volatility"]],
            on=["Season", "TeamID"],
            how="left",
        )
    else:
        season_stats["elo_delta"]     = 0.0
        season_stats["elo_volatility"] = 0.0
    season_stats["elo_delta"]      = season_stats["elo_delta"].fillna(0.0)
    season_stats["elo_volatility"] = season_stats["elo_volatility"].fillna(0.0)

    # ── 11. Poisson lambda windows (adaptive shrinkage) ────────────────────
    poisson_shrinkage_k = cfg.get("poisson_shrinkage_k", 8.0)
    alpha_ci_val        = alpha_ci

    poisson_rows = []
    for team_id, grp in season_df.groupby("TeamID"):
        row = {"Season": season, "TeamID": team_id}
        for window in [3, 5, "season"]:
            stats = build_window_stats(
                grp,
                window,
                alpha=alpha_ci_val,
                shrinkage_k=poisson_shrinkage_k,
                league_avg_for=league_avg_for,
                league_avg_against=league_avg_against,
            )
            prefix = f"recent{window}" if window != "season" else "season"
            row[f"{prefix}_lambda_for"]              = stats["lambda_for"]
            row[f"{prefix}_lambda_for_ci_low"]       = stats["lambda_for_ci_low"]
            row[f"{prefix}_lambda_for_ci_high"]      = stats["lambda_for_ci_high"]
            row[f"{prefix}_lambda_against"]          = stats["lambda_against"]
            row[f"{prefix}_lambda_against_ci_low"]   = stats["lambda_against_ci_low"]
            row[f"{prefix}_lambda_against_ci_high"]  = stats["lambda_against_ci_high"]
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
    low_map  = {"TeamID": "TeamIDLow",  **{c: f"{c}_low"  for c in feature_cols}}
    high_map = {"TeamID": "TeamIDHigh", **{c: f"{c}_high" for c in feature_cols}}

    out = out.merge(all_features.rename(columns=low_map),  on=["Season", "TeamIDLow"],  how="left")
    out = out.merge(all_features.rename(columns=high_map), on=["Season", "TeamIDHigh"], how="left")

    # Fallback numeric values for missing teams
    fallback_numeric = {
        "season_points_for":      cfg["fallback_points_for"],
        "season_points_against":  cfg["fallback_points_against"],
        "season_margin":          0.0,
        "season_win_pct":         0.5,
        "season_games_played":    0.0,
        "recent3_points_for":     cfg["fallback_points_for"],
        "recent3_points_against": cfg["fallback_points_against"],
        "recent3_margin":         0.0,
        "recent5_points_for":     cfg["fallback_points_for"],
        "recent5_points_against": cfg["fallback_points_against"],
        "recent5_margin":         0.0,
        "ewma_margin":            0.0,
        "Elo":                    cfg["fallback_elo"],
        "elo_delta":              0.0,
        "elo_volatility":         0.0,
        "sos_margin":             0.0,
        "sos_win_rate":           0.5,
        "quality_proxy":          0.0,
        "rank_proxy":             0.5,
        "season_points_for_std":  12.0,
        "season_margin_std":      12.0,
        "season_trajectory":      0.0,
        "blowout_pct":            0.0,
        "close_game_win_pct":     0.5,
        "quality_win_pct":        0.5,
        "league_avg_score":       70.0,
        "recent3_lambda_for":     cfg["fallback_points_for"],
        "recent3_lambda_against": cfg["fallback_points_against"],
        "recent5_lambda_for":     cfg["fallback_points_for"],
        "recent5_lambda_against": cfg["fallback_points_against"],
        "season_lambda_for":      cfg["fallback_points_for"],
        "season_lambda_against":  cfg["fallback_points_against"],
    }
    for base_col, fill_val in fallback_numeric.items():
        for suffix in ("_low", "_high"):
            col = f"{base_col}{suffix}"
            if col in out.columns:
                out[col] = out[col].fillna(fill_val)

    # Inject league_avg_score as matchup-level scalar (average of both sides)
    if "league_avg_score_low" in out.columns and "league_avg_score_high" in out.columns:
        out["league_avg_score"] = (
            out["league_avg_score_low"].fillna(70.0)
            + out["league_avg_score_high"].fillna(70.0)
        ) / 2.0
    else:
        out["league_avg_score"] = 70.0

    return out


# ---------------------------------------------------------------------------
# Matchup-level feature differences
# ---------------------------------------------------------------------------

def make_matchup_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Create matchup-level difference features.

    All differences are oriented as (Low − High): positive value means
    the Low-ID team has the advantage.  Model target is ActualLowWin=1.

    v4 additions:
        elo_delta_diff, elo_volatility_diff  (Elo momentum and stability)
        ewma_margin_diff                     (EWMA recency-weighted margin)
    """
    from src.config import CONFIG as _CFG
    if cfg is None:
        cfg = _CFG

    out = df.copy()

    # Seed: positive → Low has better (lower) seed number
    out["seed_diff"] = out["SeedHigh"] - out["SeedLow"]

    # Elo: positive → Low has higher Elo rating
    out["elo_diff"] = out["Elo_low"] - out["Elo_high"]

    # NEW v4: Elo momentum and volatility
    out["elo_delta_diff"]     = out["elo_delta_low"]     - out["elo_delta_high"]
    out["elo_volatility_diff"] = out["elo_volatility_low"] - out["elo_volatility_high"]

    # Season stats
    out["season_points_for_diff"]     = out["season_points_for_low"]     - out["season_points_for_high"]
    out["season_points_against_diff"] = out["season_points_against_low"] - out["season_points_against_high"]
    out["season_margin_diff"]         = out["season_margin_low"]         - out["season_margin_high"]
    out["season_win_pct_diff"]        = out["season_win_pct_low"]        - out["season_win_pct_high"]

    # Recent 3
    out["recent3_points_for_diff"]     = out["recent3_points_for_low"]     - out["recent3_points_for_high"]
    out["recent3_points_against_diff"] = out["recent3_points_against_low"] - out["recent3_points_against_high"]
    out["recent3_margin_diff"]         = out["recent3_margin_low"]         - out["recent3_margin_high"]

    # Recent 5
    out["recent5_points_for_diff"]     = out["recent5_points_for_low"]     - out["recent5_points_for_high"]
    out["recent5_points_against_diff"] = out["recent5_points_against_low"] - out["recent5_points_against_high"]
    out["recent5_margin_diff"]         = out["recent5_margin_low"]         - out["recent5_margin_high"]

    # NEW v4: EWMA margin difference
    out["ewma_margin_diff"] = out["ewma_margin_low"] - out["ewma_margin_high"]

    # Offense vs opponent defense (matchup interaction)
    out["offense_vs_defense_low"]  = out["season_points_for_low"]  - out["season_points_against_high"]
    out["offense_vs_defense_high"] = out["season_points_for_high"] - out["season_points_against_low"]
    out["matchup_diff"] = out["offense_vs_defense_low"] - out["offense_vs_defense_high"]

    # Strength signals
    out["sos_diff"]         = out["sos_margin_low"]    - out["sos_margin_high"]
    out["quality_diff"]     = out["quality_proxy_low"] - out["quality_proxy_high"]
    # rank_proxy: lower value = better rank; high rank_proxy team has higher rank number = worse
    out["rank_diff_signed"] = out["rank_proxy_high"]   - out["rank_proxy_low"]

    # Consistency: scoring variance difference
    out["consistency_edge"] = (
        out["season_points_for_std_high"] - out["season_points_for_std_low"]
    )

    # v3 features (preserved)
    out["season_trajectory_diff"]  = out["season_trajectory_low"]   - out["season_trajectory_high"]
    out["quality_win_pct_diff"]    = out["quality_win_pct_low"]      - out["quality_win_pct_high"]
    out["blowout_pct_diff"]        = out["blowout_pct_low"]          - out["blowout_pct_high"]
    out["close_game_win_pct_diff"] = out["close_game_win_pct_low"]   - out["close_game_win_pct_high"]

    # NEW v4.1: Offensive/defensive efficiency proxy features
    # off_eff_diff: how much Low scores vs how much High allows (direct matchup score)
    # def_eff_diff: how much High scores vs how much Low allows (reverse)
    # net_eff_diff: overall efficiency advantage (= matchup_diff, explicit formulation)
    #
    # These differ from matchup_diff in that they're interpreted independently:
    # off_eff_diff > 0 means Low has a scoring advantage against High's defense
    # def_eff_diff < 0 means Low has a defensive advantage against High's offense
    if "season_points_for_low" in out.columns and "season_points_against_high" in out.columns:
        out["off_eff_diff"] = (
            out["season_points_for_low"] - out["season_points_against_high"]
        )
        out["def_eff_diff"] = (
            out["season_points_against_low"] - out["season_points_for_high"]
        )
        # net: positive = Low has overall efficiency advantage
        out["net_eff_diff"] = out["off_eff_diff"] - out["def_eff_diff"]

        # Recent form efficiency (last 5 games)
        if "recent5_points_for_low" in out.columns:
            out["recent_off_eff_diff"] = (
                out["recent5_points_for_low"] - out["recent5_points_against_high"]
            )
            out["recent_def_eff_diff"] = (
                out["recent5_points_against_low"] - out["recent5_points_for_high"]
            )

    # Poisson matchup features (includes multiplicative lambda + new v4 features)
    poisson_cfg = {
        "poisson_blend_weights": cfg.get(
            "poisson_blend_weights", {"recent3": 0.40, "recent5": 0.35, "season": 0.25}
        ),
        "max_points_poisson": cfg.get("max_points_poisson", 155),
    }
    out = add_poisson_matchup_features(out, cfg=poisson_cfg)

    return out
