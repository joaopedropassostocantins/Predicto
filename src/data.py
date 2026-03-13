# src/data.py
# SUBSTITUA O ARQUIVO INTEIRO POR ESTE CONTEÚDO LIMPO

import os
import re
import pandas as pd


def parse_seed(seed_str: str):
    m = re.search(r"(\d+)", str(seed_str))
    return int(m.group(1)) if m else None


def load_regular_season_detailed(data_dir: str, prefix: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{prefix}RegularSeasonDetailedResults.csv")
    df = pd.read_csv(path)

    winners = df[["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc"]].copy()
    winners.columns = ["Season", "DayNum", "TeamID", "PointsFor", "OppTeamID", "PointsAgainst", "GameLocRaw"]
    winners["Win"] = 1

    losers = df[["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "WLoc"]].copy()
    losers.columns = ["Season", "DayNum", "TeamID", "PointsFor", "OppTeamID", "PointsAgainst", "GameLocRaw"]
    losers["Win"] = 0

    winners["GameLoc"] = winners["GameLocRaw"].map({"H": "H", "A": "A", "N": "N"})
    losers["GameLoc"] = losers["GameLocRaw"].map({"H": "A", "A": "H", "N": "N"})

    out = pd.concat([winners, losers], ignore_index=True)
    out["Gender"] = prefix
    out["Margin"] = out["PointsFor"] - out["PointsAgainst"]
    return out.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)


def load_tourney_detailed(data_dir: str, prefix: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{prefix}NCAATourneyDetailedResults.csv")
    df = pd.read_csv(path)
    df["Gender"] = prefix
    return df


def load_seeds(data_dir: str, prefix: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{prefix}NCAATourneySeeds.csv")
    df = pd.read_csv(path)
    df["SeedNum"] = df["Seed"].apply(parse_seed)
    return df[["Season", "TeamID", "SeedNum"]].copy()


def load_sample_submission(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "SampleSubmissionStage2.csv"))


def parse_submission_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["ID"].str.split("_", expand=True)
    out["Season"] = parts[0].astype(int)
    out["TeamIDLow"] = parts[1].astype(int)
    out["TeamIDHigh"] = parts[2].astype(int)
    out["Gender"] = (out["TeamIDLow"] < 2000).map({True: "M", False: "W"})
    out["LowHome"] = 0
    out["HighHome"] = 0
    return out


def prepare_eval_games(tourney_df: pd.DataFrame, season: int, gender: str) -> pd.DataFrame:
    df = tourney_df[tourney_df["Season"] == season].copy()
    df["TeamIDLow"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["TeamIDHigh"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["ActualLowWin"] = (df["WTeamID"] == df["TeamIDLow"]).astype(int)
    df["ID"] = (
        df["Season"].astype(str)
        + "_"
        + df["TeamIDLow"].astype(str)
        + "_"
        + df["TeamIDHigh"].astype(str)
    )
    df["Gender"] = gender
    df["LowHome"] = 0
    df["HighHome"] = 0
    return df[["ID", "Season", "Gender", "TeamIDLow", "TeamIDHigh", "ActualLowWin", "LowHome", "HighHome"]]
