"""
scripts/calibration_server.py
FastAPI server that receives config parameters from the calibration panel,
runs a rolling backtest, and returns metrics as JSON.

Start with:
    uvicorn scripts.calibration_server:app --reload --port 8787
"""

from __future__ import annotations

import copy
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Predicto imports ─────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG as BASE_CONFIG
from src.backtest import rolling_backtest

app = FastAPI(title="Predicto Calibration Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema (mirrors panel DEFAULT shape) ──────────────────────────────

class BlendWeights(BaseModel):
    xgb:     float = 0.55
    poisson: float = 0.20
    seed:    float = 0.10
    elo:     float = 0.10
    manual:  float = 0.05

class EloParams(BaseModel):
    k_factor:       float = 25.0
    initial_rating: float = 1500.0
    carry_factor:   float = 0.75
    use_margin:     bool  = True
    margin_cap:     float = 30.0

class PoissonParams(BaseModel):
    shrinkage: float = 0.20
    alpha_ci:  float = 0.10
    max_points: int  = 220
    w_recent3: float = 0.35
    w_recent5: float = 0.30
    w_season:  float = 0.35

class XGBParams(BaseModel):
    n_estimators:     int   = 700
    learning_rate:    float = 0.02
    max_depth:        int   = 4
    subsample:        float = 0.80
    colsample_bytree: float = 0.80
    min_child_weight: int   = 5
    reg_lambda:       float = 2.0
    reg_alpha:        float = 0.2

class ManualWeights(BaseModel):
    seed_diff:                 float = 0.90
    elo_diff:                  float = 1.10
    season_margin_diff:        float = 1.20
    season_win_pct_diff:       float = 0.80
    recent3_margin_diff:       float = 1.25
    recent5_margin_diff:       float = 1.00
    matchup_diff:              float = 1.10
    poisson_win_prob_centered: float = 2.20
    quality_diff:              float = 1.00
    sos_diff:                  float = 0.70
    rank_diff_signed:          float = 0.60
    consistency_edge:          float = 0.35

class ClipParams(BaseModel):
    pred_clip_min:      float = 0.025
    pred_clip_max:      float = 0.975
    manual_temperature: float = 10.0

class RunBacktestRequest(BaseModel):
    blend:          BlendWeights  = BlendWeights()
    elo:            EloParams     = EloParams()
    poisson:        PoissonParams = PoissonParams()
    xgb:            XGBParams     = XGBParams()
    manual_weights: ManualWeights = ManualWeights()
    clip:           ClipParams    = ClipParams()
    # Optional: limit seasons for faster preview runs
    seasons:        Optional[List[int]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_config(req: RunBacktestRequest) -> dict:
    """Build a full CONFIG dict by overriding BASE_CONFIG with panel values."""
    cfg = copy.deepcopy(BASE_CONFIG)

    # Blend weights (raw; renormalised internally by model)
    total = req.blend.xgb + req.blend.poisson + req.blend.seed + req.blend.elo + req.blend.manual
    cfg["blend_weights"] = {
        "xgb":     req.blend.xgb / total,
        "poisson": req.blend.poisson / total,
        "seed":    req.blend.seed / total,
        "elo":     req.blend.elo / total,
        "manual":  req.blend.manual / total,
    }

    # Elo
    cfg["elo_k_factor"]       = req.elo.k_factor
    cfg["elo_initial_rating"] = req.elo.initial_rating
    cfg["elo_carry_factor"]   = req.elo.carry_factor
    cfg["elo_use_margin"]     = req.elo.use_margin
    cfg["elo_margin_cap"]     = req.elo.margin_cap

    # Poisson
    total_w = req.poisson.w_recent3 + req.poisson.w_recent5 + req.poisson.w_season
    cfg["poisson_blend_weights"] = {
        "recent3": req.poisson.w_recent3 / total_w,
        "recent5": req.poisson.w_recent5 / total_w,
        "season":  req.poisson.w_season  / total_w,
    }
    cfg["poisson_shrinkage"]  = req.poisson.shrinkage
    cfg["alpha_ci"]           = req.poisson.alpha_ci
    cfg["max_points_poisson"] = req.poisson.max_points

    # XGBoost
    cfg["xgb_params"].update({
        "n_estimators":     req.xgb.n_estimators,
        "learning_rate":    req.xgb.learning_rate,
        "max_depth":        req.xgb.max_depth,
        "subsample":        req.xgb.subsample,
        "colsample_bytree": req.xgb.colsample_bytree,
        "min_child_weight": req.xgb.min_child_weight,
        "reg_lambda":       req.xgb.reg_lambda,
        "reg_alpha":        req.xgb.reg_alpha,
    })

    # Manual weights
    cfg["manual_feature_weights"] = req.manual_weights.model_dump()

    # Clipping + temperature
    cfg["pred_clip_min"]      = req.clip.pred_clip_min
    cfg["pred_clip_max"]      = req.clip.pred_clip_max
    cfg["manual_temperature"] = req.clip.manual_temperature

    return cfg


def _summary_to_json(results: dict) -> dict:
    """Convert backtest results DataFrames to JSON-serialisable dicts."""
    summary_df = results["summary"]
    rows: List[Dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        rows.append({
            "season":     int(row["Season"]),
            "games":      int(row["Games"]),
            "brier":      round(float(row["brier"]), 4),
            "accuracy":   round(float(row["accuracy"]), 4),
            "log_loss":   round(float(row["log_loss"]), 4),
            "auc":        round(float(row.get("auc", 0)), 4),
            "calibrator": str(row["Calibrator"]),
        })

    # Aggregated mean
    means = summary_df[["brier", "accuracy", "log_loss"]].mean()
    auc_col = "auc" if "auc" in summary_df.columns else None
    agg: Dict[str, Any] = {
        "brier":    round(float(means["brier"]), 4),
        "accuracy": round(float(means["accuracy"]), 4),
        "log_loss": round(float(means["log_loss"]), 4),
    }
    if auc_col:
        agg["auc"] = round(float(summary_df[auc_col].mean()), 4)

    # Calibration table
    cal_table: List[Dict[str, Any]] = []
    for _, row in results["calibration_table"].iterrows():
        cal_table.append({k: round(float(v), 4) for k, v in row.items()})

    return {"seasons": rows, "aggregate": agg, "calibration_table": cal_table}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-backtest")
def run_backtest(req: RunBacktestRequest):
    try:
        cfg = _merge_config(req)
        seasons = req.seasons or cfg["backtest_seasons"]
        results = rolling_backtest(seasons=seasons, cfg=cfg, calibrate=True)
        return {"ok": True, "results": _summary_to_json(results)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
