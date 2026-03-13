from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

from src.calibration import apply_calibrator, choose_best_calibrator, calibration_table
from src.config import CONFIG
from src.data import (
    load_regular_season_detailed,
    load_tourney_detailed,
    load_seeds,
    prepare_eval_games,
)
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.metrics import full_metric_bundle, probability_band_report
from src.model import compute_all_probabilities


def build_prediction_frame_asof(
    season: int,
    eval_df: pd.DataFrame,
    games_long_m: pd.DataFrame,
    games_long_w: pd.DataFrame,
    seeds_m: pd.DataFrame,
    seeds_w: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    m_features = build_team_features(games_long_m, season, cfg["recent_games_window"], cfg["alpha_ci"])
    w_features = build_team_features(games_long_w, season, cfg["recent_games_window"], cfg["alpha_ci"])

    pred_df = attach_team_features(eval_df, m_features, w_features, seeds_m, seeds_w, cfg)
    pred_df = make_matchup_features(pred_df)
    pred_df = compute_all_probabilities(pred_df, cfg)
    return pred_df


def evaluate_single_season(
    season: int,
    cfg: dict,
    genders: Tuple[str, ...] = ("M", "W"),
) -> pd.DataFrame:
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_tourney = load_tourney_detailed(cfg["data_dir"], "M")
    w_tourney = load_tourney_detailed(cfg["data_dir"], "W")
    m_seeds = load_seeds(cfg["data_dir"], "M")
    w_seeds = load_seeds(cfg["data_dir"], "W")

    eval_frames = []
    if "M" in genders:
        eval_frames.append(prepare_eval_games(m_tourney, season, "M"))
    if "W" in genders:
        eval_frames.append(prepare_eval_games(w_tourney, season, "W"))

    eval_df = pd.concat(eval_frames, ignore_index=True)

    pred_df = build_prediction_frame_asof(
        season=season,
        eval_df=eval_df,
        games_long_m=m_regular,
        games_long_w=w_regular,
        seeds_m=m_seeds,
        seeds_w=w_seeds,
        cfg=cfg,
    )
    return pred_df


def rolling_backtest(
    seasons: List[int],
    cfg: dict,
    genders: Tuple[str, ...] = ("M", "W"),
    calibrate: bool = True,
    calibrator_methods=None,
) -> Dict[str, pd.DataFrame]:
    all_frames = []
    summary_rows = []

    raw_frames_by_season = {}
    for season in seasons:
        pred_df = evaluate_single_season(season=season, cfg=cfg, genders=genders)
        raw_frames_by_season[season] = pred_df.copy()

    for i, season in enumerate(seasons):
        fold_df = raw_frames_by_season[season].copy()
        fold_df["PredRaw"] = fold_df["Pred"]

        if calibrate and i >= 1:
            train_df = pd.concat([raw_frames_by_season[s] for s in seasons[:i]], ignore_index=True)

            best_cal = choose_best_calibrator(
                p_train=train_df["Pred"].values,
                y_train=train_df["ActualLowWin"].values,
                p_valid=fold_df["Pred"].values,
                y_valid=fold_df["ActualLowWin"].values,
                scorer_fn=lambda y, p: full_metric_bundle(y, p),
                methods=calibrator_methods or ["identity", "platt", "isotonic"],
            )
            fold_df["Pred"] = apply_calibrator(best_cal.fitted, fold_df["Pred"].values)
            cal_method = best_cal.method
        else:
            cal_method = "none"

        metrics = full_metric_bundle(fold_df["ActualLowWin"].values, fold_df["Pred"].values)
        metrics["Season"] = season
        metrics["Games"] = len(fold_df)
        metrics["Calibrator"] = cal_method

        all_frames.append(fold_df)
        summary_rows.append(metrics)

    all_preds = pd.concat(all_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    calib_table = calibration_table(all_preds["ActualLowWin"].values, all_preds["Pred"].values, bins=10)
    band_report = probability_band_report(all_preds["ActualLowWin"].values, all_preds["Pred"].values)

    return {
        "summary": summary,
        "predictions": all_preds,
        "calibration_table": calib_table,
        "probability_bands": band_report,
    }


def save_backtest_outputs(results: Dict[str, pd.DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for name, df in results.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
