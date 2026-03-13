
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
from src.data import load_regular_season_detailed, load_seeds
from src.features import build_team_features, attach_team_features, make_matchup_features

def main():
    print("Generating 2024 submission with calibrated model...")
    
    # 1. Prepare historical data for training
    seasons = CONFIG["backtest_seasons"]
    raw_frames = {}
    for s in seasons:
        print(f"Processing season {s}...")
        raw_frames[s] = evaluate_single_season(s, CONFIG)
    
    # 2. Calibrate
    train_seasons = seasons[:-1]
    valid_season = seasons[-1]
    
    train_df = pd.concat([raw_frames[s] for s in train_seasons], ignore_index=True)
    train_df = compute_all_probabilities(train_df, CONFIG, train_df=None)
    
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
    print(f"Best calibrator: {best_cal.method}")

    # 3. Create all possible matchups for 2024
    print("Creating 2024 matchups...")
    m_seeds = load_seeds(CONFIG["data_dir"], "M")
    w_seeds = load_seeds(CONFIG["data_dir"], "W")
    
    m_teams_2024 = m_seeds[m_seeds["Season"] == 2024]["TeamID"].unique()
    w_teams_2024 = w_seeds[w_seeds["Season"] == 2024]["TeamID"].unique()
    
    matchups = []
    # Men's matchups
    for i, t1 in enumerate(sorted(m_teams_2024)):
        for t2 in sorted(m_teams_2024)[i+1:]:
            matchups.append({"Season": 2024, "TeamIDLow": t1, "TeamIDHigh": t2, "Gender": "M", "ID": f"2024_{t1}_{t2}"})
            
    # Women's matchups
    for i, t1 in enumerate(sorted(w_teams_2024)):
        for t2 in sorted(w_teams_2024)[i+1:]:
            matchups.append({"Season": 2024, "TeamIDLow": t1, "TeamIDHigh": t2, "Gender": "W", "ID": f"2024_{t1}_{t2}"})
            
    sub_df = pd.DataFrame(matchups)
    sub_df["LowHome"] = 0
    sub_df["HighHome"] = 0
    
    # 4. Feature Engineering for 2024
    m_regular = load_regular_season_detailed(CONFIG["data_dir"], "M")
    w_regular = load_regular_season_detailed(CONFIG["data_dir"], "W")
    
    m_features = build_team_features(m_regular, 2024, CONFIG["recent_games_window"], CONFIG["alpha_ci"])
    w_features = build_team_features(w_regular, 2024, CONFIG["recent_games_window"], CONFIG["alpha_ci"])
    
    pred_df = attach_team_features(sub_df, m_features, w_features, m_seeds, w_seeds, CONFIG)
    pred_df = make_matchup_features(pred_df)
    
    # 5. Final Prediction
    final_train_df = pd.concat([raw_frames[s] for s in seasons], ignore_index=True)
    pred_df = compute_all_probabilities(pred_df, CONFIG, train_df=final_train_df)
    pred_df["Pred"] = apply_calibrator(best_cal.fitted, pred_df["Pred"].values)
    
    # 6. Save
    pred_df[["ID", "Pred"]].to_csv("submission_2024_calibrated.csv", index=False)
    print("Submission saved to submission_2024_calibrated.csv")

if __name__ == "__main__":
    main()
