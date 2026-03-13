
import pandas as pd
import os
from src.data import load_sample_submission, parse_submission_ids, load_regular_season_detailed, load_seeds
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.models import compute_all_probabilities
from src.calibration import apply_calibrator, fit_calibrator

def generate_submission(cfg, train_df, calibrator=None):
    print("Generating submission...")
    sample_sub = load_sample_submission(cfg["data_dir"])
    sub_df = parse_submission_ids(sample_sub)
    
    # Load data for the target season
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_seeds = load_seeds(cfg["data_dir"], "M")
    w_seeds = load_seeds(cfg["data_dir"], "W")
    
    # Build features for target season
    m_features = build_team_features(m_regular, cfg["target_season"], cfg["recent_games_window"], cfg["alpha_ci"])
    w_features = build_team_features(w_regular, cfg["target_season"], cfg["recent_games_window"], cfg["alpha_ci"])
    
    # Attach features to submission frame
    pred_df = attach_team_features(sub_df, m_features, w_features, m_seeds, w_seeds, cfg)
    pred_df = make_matchup_features(pred_df)
    
    # Compute probabilities using the trained models
    pred_df = compute_all_probabilities(pred_df, cfg, train_df=train_df)
    
    # Apply calibration if provided
    if calibrator:
        pred_df["Pred"] = apply_calibrator(calibrator, pred_df["Pred"].values)
        
    # Prepare final submission
    submission = pred_df[["ID", "Pred"]].copy()
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    return submission
