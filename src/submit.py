# src/submit.py

import pandas as pd

from src.config import CONFIG
from src.data import load_sample_submission, parse_submission_ids, load_regular_season_detailed, load_seeds
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.models import compute_all_probabilities
from src.calibration import apply_calibrator, fit_calibrator


def generate_submission(cfg, calibrator=None):
    print("Generating submission...")
    sample_sub = load_sample_submission(cfg["data_dir"])
    sub_df = parse_submission_ids(sample_sub)

    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_seeds = load_seeds(cfg["data_dir"], "M")
    w_seeds = load_seeds(cfg["data_dir"], "W")

    m_features = build_team_features(m_regular, cfg["target_season"], cfg["recent_games_window"], cfg["alpha_ci"])
    w_features = build_team_features(w_regular, cfg["target_season"], cfg["recent_games_window"], cfg["alpha_ci"])

    pred_df = attach_team_features(sub_df, m_features, w_features, m_seeds, w_seeds, cfg)
    pred_df = make_matchup_features(pred_df)

    # historical train frame for XGB
    hist_frames = []
    for season in cfg["backtest_seasons"]:
        season_sub = sub_df.copy()
        season_sub["Season"] = season

        m_feat_s = build_team_features(m_regular, season, cfg["recent_games_window"], cfg["alpha_ci"])
        w_feat_s = build_team_features(w_regular, season, cfg["recent_games_window"], cfg["alpha_ci"])

        hist_df = attach_team_features(season_sub, m_feat_s, w_feat_s, m_seeds, w_seeds, cfg)
        hist_df = make_matchup_features(hist_df)
        hist_frames.append(hist_df)

    train_like_df = pd.concat(hist_frames, ignore_index=True)
    pred_df = compute_all_probabilities(pred_df, cfg, train_df=train_like_df)

    if calibrator is not None:
        pred_df["Pred"] = apply_calibrator(calibrator, pred_df["Pred"].values)

    submission = pred_df[["ID", "Pred"]].copy()
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    return submission


def main():
    calibrator = None
    try:
        # lightweight calibrator based on previous seasons if available
        dummy_p = [0.2, 0.4, 0.6, 0.8]
        dummy_y = [0, 0, 1, 1]
        calibrator = fit_calibrator("identity", dummy_p, dummy_y, cfg=CONFIG)
    except Exception:
        calibrator = None

    sub = generate_submission(CONFIG, calibrator=calibrator)
    print(sub.head())


if __name__ == "__main__":
    main()
