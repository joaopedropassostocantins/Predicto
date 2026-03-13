# src/backtest.py — Rolling temporal backtest — v3.0

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.calibration import apply_calibrator, choose_best_calibrator, calibration_table
from src.data import load_regular_season_detailed, load_tourney_detailed, load_seeds, prepare_eval_games
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.metrics import full_metric_bundle, probability_band_report
from src.models import compute_all_probabilities
from src.ratings import precompute_starting_elo


# ---------------------------------------------------------------------------
# Build feature frame for a single season (no training yet)
# ---------------------------------------------------------------------------

def build_prediction_frame_asof(
    season: int,
    eval_df: pd.DataFrame,
    games_long_m: pd.DataFrame,
    games_long_w: pd.DataFrame,
    seeds_m: pd.DataFrame,
    seeds_w: pd.DataFrame,
    cfg: dict,
    starting_elo_m: dict | None = None,
    starting_elo_w: dict | None = None,
) -> pd.DataFrame:
    """
    Build a matchup feature frame for `season` using only regular-season data
    from that same season.  No future information is used.

    Parameters
    ----------
    starting_elo_m / starting_elo_w : dict, optional
        {(season, team_id): starting_elo} from cross-season Elo computation.
    """
    m_features = build_team_features(
        games_long_m, season,
        cfg["recent_games_window"], cfg["alpha_ci"],
        starting_elo=starting_elo_m, cfg=cfg,
    )
    w_features = build_team_features(
        games_long_w, season,
        cfg["recent_games_window"], cfg["alpha_ci"],
        starting_elo=starting_elo_w, cfg=cfg,
    )

    pred_df = attach_team_features(eval_df, m_features, w_features, seeds_m, seeds_w, cfg)
    pred_df = make_matchup_features(pred_df, cfg=cfg)
    return pred_df


def evaluate_single_season(
    season: int,
    cfg: dict,
    genders: Tuple[str, ...] = ("M", "W"),
    starting_elo_m: dict | None = None,
    starting_elo_w: dict | None = None,
) -> pd.DataFrame:
    """
    Load data and build feature frame for one season's tournament games.
    Returns features + ActualLowWin labels (no probabilities yet).
    """
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_tourney = load_tourney_detailed(cfg["data_dir"], "M")
    w_tourney = load_tourney_detailed(cfg["data_dir"], "W")
    m_seeds   = load_seeds(cfg["data_dir"], "M")
    w_seeds   = load_seeds(cfg["data_dir"], "W")

    eval_frames = []
    if "M" in genders:
        eval_frames.append(prepare_eval_games(m_tourney, season, "M"))
    if "W" in genders:
        eval_frames.append(prepare_eval_games(w_tourney, season, "W"))

    eval_df = pd.concat(eval_frames, ignore_index=True)

    return build_prediction_frame_asof(
        season=season,
        eval_df=eval_df,
        games_long_m=m_regular,
        games_long_w=w_regular,
        seeds_m=m_seeds,
        seeds_w=w_seeds,
        cfg=cfg,
        starting_elo_m=starting_elo_m,
        starting_elo_w=starting_elo_w,
    )


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def rolling_backtest(
    seasons: List[int],
    cfg: dict,
    genders: Tuple[str, ...] = ("M", "W"),
    calibrate: bool = True,
    calibrator_methods: list | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Rolling temporal backtest.

    For each season i in `seasons`:
    - Train XGBoost on tournament games from seasons[0 .. i-1].
    - Evaluate on season i.
    - Calibrator selection uses out-of-sample XGB predictions from
      earlier folds so that calibration itself has no temporal leakage.

    Cross-season Elo is pre-computed once using all regular season data
    (no future regular season data leaks into tournament predictions for
    season i because Elo is capped at end-of-season i regular games).
    """
    if calibrator_methods is None:
        calibrator_methods = cfg.get("calibration_methods",
                                     ["identity", "temperature", "platt", "isotonic"])

    # ── Pre-load data once ────────────────────────────────────────────────
    print("Loading data...")
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")

    # ── Pre-compute cross-season Elo (uses only regular season data) ──────
    print("Pre-computing cross-season Elo...")
    starting_elo_m = precompute_starting_elo(
        _build_elo_games(m_regular),
        k_factor=cfg["elo_k_factor"],
        initial_rating=cfg["elo_initial_rating"],
        carry_factor=cfg["elo_carry_factor"],
        use_margin=cfg.get("elo_use_margin", False),
        margin_cap=cfg.get("elo_margin_cap", 30.0),
    )
    starting_elo_w = precompute_starting_elo(
        _build_elo_games(w_regular),
        k_factor=cfg["elo_k_factor"],
        initial_rating=cfg["elo_initial_rating"],
        carry_factor=cfg["elo_carry_factor"],
        use_margin=cfg.get("elo_use_margin", False),
        margin_cap=cfg.get("elo_margin_cap", 30.0),
    )

    # ── Build feature frames for all seasons ─────────────────────────────
    print("Building feature frames...")
    raw_frames: Dict[int, pd.DataFrame] = {}
    for season in seasons:
        print(f"  Season {season}...", end=" ", flush=True)
        raw_frames[season] = _load_tourney_features(
            season, cfg, genders,
            m_regular, w_regular, starting_elo_m, starting_elo_w,
        )
        print(f"{len(raw_frames[season])} games")

    # ── Rolling evaluation ────────────────────────────────────────────────
    all_frames   = []
    summary_rows = []

    for i, season in enumerate(seasons):
        fold_df = raw_frames[season].copy()

        # Train XGBoost on all previous seasons
        if i >= 1:
            train_df = pd.concat(
                [raw_frames[s] for s in seasons[:i]], ignore_index=True
            )
            fold_df = compute_all_probabilities(fold_df, cfg, train_df=train_df)
        else:
            fold_df = compute_all_probabilities(fold_df, cfg, train_df=None)

        fold_df["PredRaw"] = fold_df["Pred"].copy()

        # ── Calibrator selection ──────────────────────────────────────────
        # For correct calibration, we need out-of-sample XGB predictions
        # for the calibrator fitting set.  We generate these by running the
        # same rolling approach on the earlier folds.
        cal_method = "none"
        if calibrate and i >= 2:
            # Build calibrator training set: rolling OOF predictions for
            # all seasons before the validation season.
            cal_oof_preds = []
            for j in range(i - 1):
                cal_fold = raw_frames[seasons[j]].copy()
                if j >= 1:
                    cal_hist = pd.concat(
                        [raw_frames[s] for s in seasons[:j]], ignore_index=True
                    )
                    cal_fold = compute_all_probabilities(cal_fold, cfg, train_df=cal_hist)
                else:
                    cal_fold = compute_all_probabilities(cal_fold, cfg, train_df=None)
                cal_oof_preds.append(cal_fold)

            cal_train_df = pd.concat(cal_oof_preds, ignore_index=True)

            # Validation season: XGB trained on seasons[0..i-2], predict on
            # seasons[i-1].  This is the held-out set for calibrator selection.
            cal_val_raw = raw_frames[seasons[i - 1]].copy()
            cal_val_hist = pd.concat(
                [raw_frames[s] for s in seasons[:i - 1]], ignore_index=True
            )
            cal_val_df = compute_all_probabilities(
                cal_val_raw, cfg, train_df=cal_val_hist
            )

            best_cal = choose_best_calibrator(
                p_train=cal_train_df["Pred"].values,
                y_train=cal_train_df["ActualLowWin"].values,
                p_valid=cal_val_df["Pred"].values,
                y_valid=cal_val_df["ActualLowWin"].values,
                scorer_fn=lambda y, p: full_metric_bundle(y, p),
                methods=calibrator_methods,
                cfg=cfg,
            )
            fold_df["Pred"] = apply_calibrator(best_cal.fitted, fold_df["Pred"].values)
            cal_method = best_cal.method

        metrics             = full_metric_bundle(
            fold_df["ActualLowWin"].values, fold_df["Pred"].values
        )
        metrics["Season"]     = season
        metrics["Games"]      = len(fold_df)
        metrics["Calibrator"] = cal_method

        all_frames.append(fold_df)
        summary_rows.append(metrics)

        print(
            f"  → Season {season}: Brier={metrics['brier']:.4f}  "
            f"Acc={metrics['accuracy']:.3f}  "
            f"LogLoss={metrics['log_loss']:.4f}  "
            f"Cal={cal_method}"
        )

    all_preds = pd.concat(all_frames, ignore_index=True)
    summary   = pd.DataFrame(summary_rows)

    calib_table  = calibration_table(
        all_preds["ActualLowWin"].values, all_preds["Pred"].values, bins=10
    )
    band_report = probability_band_report(
        all_preds["ActualLowWin"].values, all_preds["Pred"].values
    )

    return {
        "summary":           summary,
        "predictions":       all_preds,
        "calibration_table": calib_table,
        "probability_bands": band_report,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_elo_games(regular_df: pd.DataFrame) -> pd.DataFrame:
    """Extract (Season, DayNum, WTeamID, LTeamID, Margin) for Elo computation."""
    wins = regular_df[regular_df["Win"] == 1].copy()
    return (
        wins[["Season", "DayNum", "TeamID", "OppTeamID", "Margin"]]
        .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
        .drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID"])
        .reset_index(drop=True)
    )


def _load_tourney_features(
    season: int,
    cfg: dict,
    genders: tuple,
    m_regular: pd.DataFrame,
    w_regular: pd.DataFrame,
    starting_elo_m: dict,
    starting_elo_w: dict,
) -> pd.DataFrame:
    """Load tourney eval frame and attach features for one season."""
    m_tourney = load_tourney_detailed(cfg["data_dir"], "M")
    w_tourney = load_tourney_detailed(cfg["data_dir"], "W")
    m_seeds   = load_seeds(cfg["data_dir"], "M")
    w_seeds   = load_seeds(cfg["data_dir"], "W")

    eval_frames = []
    if "M" in genders:
        eval_frames.append(prepare_eval_games(m_tourney, season, "M"))
    if "W" in genders:
        eval_frames.append(prepare_eval_games(w_tourney, season, "W"))

    eval_df = pd.concat(eval_frames, ignore_index=True)

    return build_prediction_frame_asof(
        season=season,
        eval_df=eval_df,
        games_long_m=m_regular,
        games_long_w=w_regular,
        seeds_m=m_seeds,
        seeds_w=w_seeds,
        cfg=cfg,
        starting_elo_m=starting_elo_m,
        starting_elo_w=starting_elo_w,
    )


def save_backtest_outputs(results: Dict[str, pd.DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for name, df in results.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    print(f"Backtest outputs saved to {output_dir}/")
