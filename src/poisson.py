
import numpy as np
import pandas as pd
from scipy.stats import poisson
from src.config import CONFIG
from src.utils import clip_probs, sigmoid

def poisson_match_win_probability(lambda_a: float, lambda_b: float, max_points: int = CONFIG["max_points_poisson"]) -> float:
    pts = np.arange(0, max_points + 1)
    pa = poisson.pmf(pts, lambda_a)
    pb = poisson.pmf(pts, lambda_b)

    joint = np.outer(pa, pb)
    p_a_gt_b = np.tril(joint, k=-1).sum()
    p_tie = np.trace(joint)
    p_b_gt_a = np.triu(joint, k=1).sum()

    total = p_a_gt_b + p_tie + p_b_gt_a
    if total > 0:
        p_a_gt_b /= total
        p_tie /= total
        p_b_gt_a /= total

    return float(p_a_gt_b + 0.5 * p_tie)

def calculate_lambda(points_for_avg, points_against_avg, league_avg_points):
    # Basic lambda calculation: (Team A Offense * Team B Defense) / League Average
    # Here we simplify: lambda = (points_for_avg + points_against_avg_opp) / 2
    # A more robust version would use offensive and defensive ratings.
    return (points_for_avg + points_against_avg) / 2.0

def compute_poisson_features(df, cfg):
    out = df.copy()
    
    windows = cfg.get("poisson_windows", [3, 5, "season"])
    
    for window in windows:
        suffix = f"_{window}" if window != "season" else "_season"
        p_list = []
        
        for row in out.itertuples(index=False):
            # Extract points for and against based on window
            if window == "season":
                pf_low = getattr(row, "season_points_for_low")
                pa_low = getattr(row, "season_points_against_low")
                pf_high = getattr(row, "season_points_for_high")
                pa_high = getattr(row, "season_points_against_high")
            else:
                # Assuming these features exist in the dataframe
                pf_low = getattr(row, f"recent{window}_points_for_low", getattr(row, "season_points_for_low"))
                pa_low = getattr(row, f"recent{window}_points_against_low", getattr(row, "season_points_against_low"))
                pf_high = getattr(row, f"recent{window}_points_for_high", getattr(row, "season_points_for_high"))
                pa_high = getattr(row, f"recent{window}_points_against_high", getattr(row, "season_points_against_high"))
            
            lambda_low = calculate_lambda(pf_low, pa_high, 70.0)
            lambda_high = calculate_lambda(pf_high, pa_low, 70.0)
            
            p = poisson_match_win_probability(lambda_low, lambda_high, cfg["max_points_poisson"])
            p_list.append(p)
            
        out[f"p_poisson{suffix}"] = p_list
        
    # Weighted blend of Poisson probabilities
    weights = cfg.get("poisson_blend_weights", {"recent3": 0.4, "recent5": 0.3, "season": 0.3})
    out["p_poisson_blend"] = (
        weights.get("recent3", 0) * out.get("p_poisson_3", out["p_poisson_season"]) +
        weights.get("recent5", 0) * out.get("p_poisson_5", out["p_poisson_season"]) +
        weights.get("season", 0) * out["p_poisson_season"]
    )
    
    return out
