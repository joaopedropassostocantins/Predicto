
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest import evaluate_single_season
from src.calibration import choose_best_calibrator, apply_calibrator
from src.config import CONFIG
from src.models import compute_all_probabilities
from src.metrics import full_metric_bundle
from src.data import load_sample_submission, parse_submission_ids, load_regular_season_detailed, load_seeds
from src.features import build_team_features, attach_team_features, make_matchup_features

def main():
    print("Starting final training and calibration...")
    
    # 1. Load and prepare data for all seasons
    seasons = CONFIG["backtest_seasons"]
    raw_frames = {}
    for s in seasons:
        print(f"Processing season {s}...")
        raw_frames[s] = evaluate_single_season(s, CONFIG)
    
    # 2. Train calibrator using rolling approach on historical data
    # We'll use the last season as validation for the calibrator
    train_seasons = seasons[:-1]
    valid_season = seasons[-1]
    
    print(f"Training base models on seasons {train_seasons}...")
    train_df = pd.concat([raw_frames[s] for s in train_seasons], ignore_index=True)
    # Base probabilities for training calibrator
    train_df = compute_all_probabilities(train_df, CONFIG, train_df=None)
    
    print(f"Validating calibrator on season {valid_season}...")
    valid_df = raw_frames[valid_season].copy()
    valid_df = compute_all_probabilities(valid_df, CONFIG, train_df=train_df)
    
    best_cal = choose_best_calibrator(
        p_train=train_df["Pred"].values,
        y_train=train_df["ActualLowWin"].values,
        p_valid=valid_df["Pred"].values,
        y_valid=valid_df["ActualLowWin"].values,
        scorer_fn=lambda y, p: full_metric_bundle(y, p),
        methods=CONFIG["calibration_methods"],
        cfg=CONFIG,
    )
    
    print(f"Best calibrator found: {best_cal.method}")
    print(f"Validation Brier Score: {best_cal.metrics['brier']:.4f}")
    
    # 3. Generate Final Submission for Target Season
    print(f"Generating submission for target season {CONFIG['target_season']}...")
    sample_sub = load_sample_submission(CONFIG["data_dir"])
    sub_df = parse_submission_ids(sample_sub)
    
    m_regular = load_regular_season_detailed(CONFIG["data_dir"], "M")
    w_regular = load_regular_season_detailed(CONFIG["data_dir"], "W")
    m_seeds = load_seeds(CONFIG["data_dir"], "M")
    w_seeds = load_seeds(CONFIG["data_dir"], "W")
    
    m_features = build_team_features(m_regular, CONFIG["target_season"], CONFIG["recent_games_window"], CONFIG["alpha_ci"])
    w_features = build_team_features(w_regular, CONFIG["target_season"], CONFIG["recent_games_window"], CONFIG["alpha_ci"])
    
    pred_df = attach_team_features(sub_df, m_features, w_features, m_seeds, w_seeds, CONFIG)
    pred_df = make_matchup_features(pred_df)
    
    # Final model uses all backtest seasons for training
    final_train_df = pd.concat([raw_frames[s] for s in seasons], ignore_index=True)
    pred_df = compute_all_probabilities(pred_df, CONFIG, train_df=final_train_df)
    
    # Apply the best calibrator
    pred_df["Pred"] = apply_calibrator(best_cal.fitted, pred_df["Pred"].values)
    
    submission = pred_df[["ID", "Pred"]].copy()
    submission.to_csv("final_submission.csv", index=False)
    print("Final submission saved to final_submission.csv")
    
    # Save the summary of performance
    summary_data = {
        "Best Calibrator": [best_cal.method],
        "Validation Brier": [best_cal.metrics['brier']],
        "Validation Accuracy": [best_cal.metrics['accuracy']],
        "Seasons Trained": [str(seasons)]
    }
    pd.DataFrame(summary_data).to_csv("training_summary.csv", index=False)

if __name__ == "__main__":
    main()
