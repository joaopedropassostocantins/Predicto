# src/evaluate.py

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.config import CONFIG
from src.data import (
    load_regular_season_detailed,
    load_tourney_detailed,
    load_seeds,
    prepare_eval_games,
)
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.model import compute_all_probabilities


def build_prediction_frame(season: int, eval_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_seeds = load_seeds(cfg["data_dir"], "M")
    w_seeds = load_seeds(cfg["data_dir"], "W")

    m_features = build_team_features(m_regular, season, cfg["recent_games_window"], cfg["alpha_ci"])
    w_features = build_team_features(w_regular, season, cfg["recent_games_window"], cfg["alpha_ci"])

    pred_df = attach_team_features(eval_df, m_features, w_features, m_seeds, w_seeds, cfg)
    pred_df = make_matchup_features(pred_df)
    pred_df = compute_all_probabilities(pred_df, cfg)
    return pred_df


def evaluate_season(season: int, cfg: dict):
    m_tourney = load_tourney_detailed(cfg["data_dir"], "M")
    w_tourney = load_tourney_detailed(cfg["data_dir"], "W")

    eval_m = prepare_eval_games(m_tourney, season, "M")
    eval_w = prepare_eval_games(w_tourney, season, "W")
    eval_df = pd.concat([eval_m, eval_w], ignore_index=True)

    pred_df = build_prediction_frame(season, eval_df, cfg)

    brier = brier_score_loss(pred_df["ActualLowWin"], pred_df["Pred"])
    acc = ((pred_df["Pred"] >= 0.5).astype(int) == pred_df["ActualLowWin"]).mean()
    ll = log_loss(pred_df["ActualLowWin"], pred_df["Pred"], labels=[0, 1])

    return pred_df, brier, acc, ll


def main():
    seasons = [2022, 2023, 2024, 2025]
    rows = []
    frames = []

    for season in seasons:
        pred_df, brier, acc, ll = evaluate_season(season, CONFIG)
        frames.append(pred_df)
        rows.append(
            {
                "Season": season,
                "BrierScore": brier,
                "Accuracy": acc,
                "LogLoss": ll,
                "Games": len(pred_df),
            }
        )

    summary = pd.DataFrame(rows)
    all_preds = pd.concat(frames, ignore_index=True)

    summary.to_csv("/kaggle/working/eval_summary.csv", index=False)
    all_preds.to_csv("/kaggle/working/eval_predictions.csv", index=False)

    print(summary)
    print("\nArquivos salvos:")
    print("/kaggle/working/eval_summary.csv")
    print("/kaggle/working/eval_predictions.csv")


if __name__ == "__main__":
    main()
