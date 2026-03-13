
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.config import CONFIG
from src.utils import sigmoid, clip_probs
from src.poisson import compute_poisson_features

class TabularModel:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        if model_type == "xgboost":
            self.model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        elif model_type == "lightgbm":
            self.model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
        elif model_type == "catboost":
            self.model = CatBoostClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

def compute_manual_score(df, cfg):
    # Refactored manual score using key features
    out = df.copy()
    out["score_raw"] = (
        0.3 * out["seed_diff"] +
        0.4 * out["elo_diff"] / 100.0 +
        0.2 * out["recent3_margin_diff"] +
        0.1 * out["season_win_pct_diff"]
    )
    out["p_manual"] = sigmoid(out["score_raw"] / 5.0)
    return out

def compute_all_probabilities(df, cfg, train_df=None):
    out = df.copy()
    
    # 1. Poisson Baseline
    out = compute_poisson_features(out, cfg)
    
    # 2. Manual Score
    out = compute_manual_score(out, cfg)
    
    # 3. Tabular Models (if train_df is provided)
    feature_cols = [
        "seed_diff", "elo_diff", "season_win_pct_diff", "season_margin_diff",
        "recent3_margin_diff", "recent5_margin_diff", "matchup_diff",
        "p_poisson_blend", "p_poisson_season"
    ]
    
    if train_df is not None:
        X_train = train_df[feature_cols]
        y_train = train_df["ActualLowWin"]
        X_test = out[feature_cols]
        
        # XGBoost
        xgb = TabularModel("xgboost").fit(X_train, y_train)
        out["p_xgboost"] = xgb.predict_proba(X_test)
        
        # LightGBM
        lgb = TabularModel("lightgbm").fit(X_train, y_train)
        out["p_lightgbm"] = lgb.predict_proba(X_test)
        
        # CatBoost
        cat = TabularModel("catboost").fit(X_train, y_train)
        out["p_catboost"] = cat.predict_proba(X_test)
        
        # 4. Final Blend
        out["Pred"] = (
            0.4 * out["p_xgboost"] +
            0.2 * out["p_lightgbm"] +
            0.2 * out["p_catboost"] +
            0.1 * out["p_poisson_blend"] +
            0.1 * out["p_manual"]
        )
    else:
        # Fallback to Poisson and Manual blend if no training data
        out["Pred"] = 0.7 * out["p_poisson_blend"] + 0.3 * out["p_manual"]
        
    out["Pred"] = clip_probs(out["Pred"])
    return out
