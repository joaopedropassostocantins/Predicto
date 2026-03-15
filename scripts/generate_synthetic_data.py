#!/usr/bin/env python3
"""
scripts/generate_synthetic_data.py
====================================
Gera dados sintéticos realistas no formato do Kaggle March Machine Learning Mania
para validação do pipeline sem acesso aos dados reais.

Seasons geradas: 2015-2025
- Men: TeamIDs 1101-1464 (364 times, ~353 com jogos na temporada)
- Women: TeamIDs 3101-3400 (300 times, ~250 com jogos na temporada)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEASONS = list(range(2015, 2027))  # 2015 to 2026 inclusive

# ── Team pools ──────────────────────────────────────────────────────────────
M_TEAMS = list(range(1101, 1465))   # 364 teams
W_TEAMS = list(range(3101, 3401))   # 300 teams

# ── True team strengths (latent) ─────────────────────────────────────────────
# Higher = better; draws from normal(70, 8) for scoring rate
M_STRENGTH = {t: np.random.normal(70, 8) for t in M_TEAMS}
W_STRENGTH = {t: np.random.normal(62, 7) for t in W_TEAMS}


def simulate_game(t1: int, t2: int, strengths: dict, loc: str = "N"):
    """Simulate a game, return (score1, score2) as ints."""
    s1 = max(strengths[t1] * (1.03 if loc == "H" else 0.97 if loc == "A" else 1.0), 1)
    s2 = max(strengths[t2] * (1.03 if loc == "A" else 0.97 if loc == "H" else 1.0), 1)
    pts1 = int(np.random.normal(s1, 8))
    pts2 = int(np.random.normal(s2, 8))
    pts1 = max(pts1, 30)
    pts2 = max(pts2, 30)
    if pts1 == pts2:
        pts1 += 1
    return pts1, pts2


def add_detailed_cols(df: pd.DataFrame, is_winner: bool, score_col: str, opp_score_col: str) -> dict:
    """Simulate box-score stats proportional to score."""
    n = len(df)
    scores = df[score_col].values
    suffix = "W" if is_winner else "L"
    return {
        f"{suffix}FGM":  np.clip((scores * 0.38 + np.random.randn(n) * 2).astype(int), 10, 45),
        f"{suffix}FGA":  np.clip((scores * 0.72 + np.random.randn(n) * 3).astype(int), 20, 75),
        f"{suffix}FGM3": np.clip((scores * 0.10 + np.random.randn(n) * 1).astype(int), 0, 15),
        f"{suffix}FGA3": np.clip((scores * 0.25 + np.random.randn(n) * 2).astype(int), 3, 30),
        f"{suffix}FTM":  np.clip((scores * 0.18 + np.random.randn(n) * 1.5).astype(int), 2, 28),
        f"{suffix}FTA":  np.clip((scores * 0.24 + np.random.randn(n) * 2).astype(int), 3, 35),
        f"{suffix}OR":   np.clip(np.random.randint(4, 15, n), 0, 20),
        f"{suffix}DR":   np.clip(np.random.randint(15, 30, n), 0, 40),
        f"{suffix}Ast":  np.clip(np.random.randint(8, 22, n), 0, 30),
        f"{suffix}TO":   np.clip(np.random.randint(8, 18, n), 0, 25),
        f"{suffix}Stl":  np.clip(np.random.randint(3, 10, n), 0, 15),
        f"{suffix}Blk":  np.clip(np.random.randint(1, 6, n), 0, 12),
        f"{suffix}PF":   np.clip(np.random.randint(12, 25, n), 5, 35),
    }


def build_regular_season(teams: list, strengths: dict, seasons: list,
                          n_games_per_team: int = 28) -> pd.DataFrame:
    """Build regular season results."""
    rows = []
    team_arr = np.array(teams)
    for season in seasons:
        # Each team plays ~28 games against random opponents
        games_played = {t: set() for t in teams}
        for _ in range(n_games_per_team * len(teams) // 2):
            t1, t2 = np.random.choice(team_arr, 2, replace=False)
            if t2 in games_played[t1]:
                continue
            games_played[t1].add(t2)
            games_played[t2].add(t1)
            day = np.random.randint(20, 132)
            locs = ["H", "A", "N", "N", "N"]
            loc = random.choice(locs)
            s1, s2 = simulate_game(t1, t2, strengths, loc)
            wteam, wscore, lteam, lscore, wloc = (
                (t1, s1, t2, s2, loc) if s1 > s2 else (t2, s2, t1, s1, {"H": "A", "A": "H", "N": "N"}[loc])
            )
            rows.append({
                "Season": season, "DayNum": day,
                "WTeamID": wteam, "WScore": wscore,
                "LTeamID": lteam, "LScore": lscore,
                "WLoc": wloc,
            })

    df = pd.DataFrame(rows).sort_values(["Season", "DayNum"]).reset_index(drop=True)

    # Add detailed columns
    w_stats = add_detailed_cols(df, True, "WScore", "LScore")
    l_stats = add_detailed_cols(df, False, "LScore", "WScore")
    for k, v in {**w_stats, **l_stats}.items():
        df[k] = v

    return df


def build_tourney_seeds(teams: list, seasons: list, n_tourney: int = 64) -> pd.DataFrame:
    """Assign seeds 1-16 across 4 regions for top teams."""
    rows = []
    team_arr = np.array(teams)
    for season in seasons:
        selected = np.random.choice(team_arr, n_tourney, replace=False)
        regions = ["W", "X", "Y", "Z"]
        for i, team in enumerate(selected):
            region = regions[i % 4]
            seed_num = (i // 4) + 1
            rows.append({
                "Season": season,
                "Seed": f"{region}{seed_num:02d}",
                "TeamID": team,
            })
    return pd.DataFrame(rows)


def build_tourney_detailed(seeds_df: pd.DataFrame, strengths: dict, seasons: list) -> pd.DataFrame:
    """Simulate NCAA tournament bracket results."""
    rows = []
    for season in seasons:
        season_seeds = seeds_df[seeds_df["Season"] == season].copy()
        teams = season_seeds["TeamID"].tolist()
        # Simple elimination bracket simulation (64 teams → 63 games)
        alive = teams[:]
        day = 134
        while len(alive) > 1:
            next_round = []
            random.shuffle(alive)
            for i in range(0, len(alive) - 1, 2):
                t1, t2 = alive[i], alive[i + 1]
                s1, s2 = simulate_game(t1, t2, strengths, "N")
                wteam, wscore, lteam, lscore = (t1, s1, t2, s2) if s1 > s2 else (t2, s2, t1, s1)
                row = {
                    "Season": season, "DayNum": day,
                    "WTeamID": wteam, "WScore": wscore,
                    "LTeamID": lteam, "LScore": lscore,
                    "WLoc": "N",
                }
                # Add box score stubs
                for prefix in ["W", "L"]:
                    score = wscore if prefix == "W" else lscore
                    for col, val in add_detailed_cols(pd.DataFrame([{"score": score, "opp": 0}]),
                                                      prefix == "W", "score", "opp").items():
                        row[col] = int(val[0])
                rows.append(row)
                next_round.append(wteam)
            if len(alive) % 2 == 1:
                next_round.append(alive[-1])
            alive = next_round
            day += 2

    return pd.DataFrame(rows).sort_values(["Season", "DayNum"]).reset_index(drop=True)


def build_sample_submission(m_seeds: pd.DataFrame, w_seeds: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Build sample submission for target season."""
    rows = []
    for gender, seeds_df in [("M", m_seeds), ("W", w_seeds)]:
        s = seeds_df[seeds_df["Season"] == target_season]
        teams = sorted(s["TeamID"].tolist())
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                rows.append({
                    "ID": f"{target_season}_{teams[i]}_{teams[j]}",
                    "Pred": 0.5,
                })
    return pd.DataFrame(rows)


def main():
    print(f"Generating synthetic NCAA data for seasons {SEASONS[0]}-{SEASONS[-1]} ...")
    print(f"Output directory: {DATA_DIR}")

    # ── Men's data ──────────────────────────────────────────────────────────
    print("  Building M regular season ... ", end="", flush=True)
    m_regular = build_regular_season(M_TEAMS, M_STRENGTH, SEASONS, n_games_per_team=28)
    m_regular.to_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv", index=False)
    print(f"done ({len(m_regular)} rows)")

    print("  Building M tourney seeds ... ", end="", flush=True)
    m_seeds = build_tourney_seeds(M_TEAMS, SEASONS, n_tourney=64)
    m_seeds.to_csv(DATA_DIR / "MNCAATourneySeeds.csv", index=False)
    print(f"done ({len(m_seeds)} rows)")

    print("  Building M tourney results ... ", end="", flush=True)
    m_tourney = build_tourney_detailed(m_seeds, M_STRENGTH, SEASONS)
    m_tourney.to_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv", index=False)
    print(f"done ({len(m_tourney)} rows)")

    # ── Women's data ─────────────────────────────────────────────────────────
    print("  Building W regular season ... ", end="", flush=True)
    w_regular = build_regular_season(W_TEAMS, W_STRENGTH, SEASONS, n_games_per_team=28)
    w_regular.to_csv(DATA_DIR / "WRegularSeasonDetailedResults.csv", index=False)
    print(f"done ({len(w_regular)} rows)")

    print("  Building W tourney seeds ... ", end="", flush=True)
    w_seeds = build_tourney_seeds(W_TEAMS, SEASONS, n_tourney=64)
    w_seeds.to_csv(DATA_DIR / "WNCAATourneySeeds.csv", index=False)
    print(f"done ({len(w_seeds)} rows)")

    print("  Building W tourney results ... ", end="", flush=True)
    w_tourney = build_tourney_detailed(w_seeds, W_STRENGTH, SEASONS)
    w_tourney.to_csv(DATA_DIR / "WNCAATourneyDetailedResults.csv", index=False)
    print(f"done ({len(w_tourney)} rows)")

    # ── Sample submission (target = 2026, use 2025 teams as proxy) ──────────
    print("  Building sample_submission.csv ... ", end="", flush=True)
    # Use 2026 seeds directly
    sub = build_sample_submission(m_seeds, w_seeds, 2026)
    sub.to_csv(DATA_DIR / "sample_submission.csv", index=False)
    print(f"done ({len(sub)} rows)")

    print()
    print("All files written:")
    for f in sorted(DATA_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:50s} {size_kb:>6} KB")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
