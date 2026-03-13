import pandas as pd

from src.config import CONFIG
from src.data import load_regular_season_detailed, load_sample_submission, load_seeds, parse_submission_ids
from src.features import attach_team_features, build_team_features, make_matchup_features
from src.model import compute_all_probabilities


def main():
    data_dir = CONFIG["data_dir"]
    target_season = CONFIG["target_season"]

    sub = load_sample_submission(data_dir)
    sub = parse_submission_ids(sub)

    m_regular = load_regular_season_detailed(data_dir, "M")
    w_regular = load_regular_season_detailed(data_dir, "W")
    m_seeds = load_seeds(data_dir, "M")
    w_seeds = load_seeds(data_dir, "W")

    m_features = build_team_features(m_regular, target_season, CONFIG["recent_games_window"], CONFIG["alpha_ci"])
    w_features = build_team_features(w_regular, target_season, CONFIG["recent_games_window"], CONFIG["alpha_ci"])

    pred_df = attach_team_features(sub, m_features, w_features, m_seeds, w_seeds, CONFIG)
    pred_df = make_matchup_features(pred_df)
    pred_df = compute_all_probabilities(pred_df, CONFIG)

    submission = pred_df[["ID", "Pred"]].copy()
    submission.to_csv("/kaggle/working/submission.csv", index=False)

    cols_to_show = [c for c in ["ID", "p_manual", "p_poisson", "p_seed", "p_rank", "Pred"] if c in pred_df.columns]
    print(pred_df[cols_to_show].head(10))
    print("\nArquivo salvo em /kaggle/working/submission.csv")


if __name__ == "__main__":
    main()
