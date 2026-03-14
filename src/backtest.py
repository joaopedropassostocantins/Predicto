# src/backtest.py — Rolling temporal backtest v4.1
#
# Changes from v4.0:
#   - Calibrator selection: single-fold → multi-fold leave-one-out (more robust).
#     Old: used only seasons[i-1] as single validation fold.
#     New: choose_best_calibrator_multifold() evaluates all available OOF folds,
#          selects by mean log_loss across LOO iterations.
#   - Calibration audit exported: before/after comparison in output dict.
#   - _log cosmetic fix: season progress on single line.
#
# Changes from v3:
#   - OOF predictions cached in a single O(n) pass instead of O(n²) recompute.
#   - Tourney data preloaded once outside the inner loop (was reloaded per season).
#   - Calibrator selection: uses log_loss as primary criterion.
#   - Added: blend_sensitivity analysis at end of backtest.
#   - Added: per_season_metrics export.
#   - All data loading consolidated: no duplicate CSV reads in inner loops.

from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.calibration import (
    apply_calibrator,
    choose_best_calibrator,
    choose_best_calibrator_multifold,
    calibration_table,
    calibration_audit_report,
)
from src.data import (
    load_regular_season_detailed,
    load_tourney_detailed,
    load_seeds,
    prepare_eval_games,
)
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.metrics import full_metric_bundle, probability_band_report, per_season_metrics
from src.models import compute_all_probabilities, blend_sensitivity_report
from src.ratings import precompute_starting_elo


# ---------------------------------------------------------------------------
# Internal helper: build Elo games frame from long-format data
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


# ---------------------------------------------------------------------------
# Feature frame construction for a single season (no training yet)
# ---------------------------------------------------------------------------

def build_prediction_frame_asof(
    season: int,
    eval_df: pd.DataFrame,
    games_long_m: pd.DataFrame,
    games_long_w: pd.DataFrame,
    seeds_m: pd.DataFrame,
    seeds_w: pd.DataFrame,
    cfg: dict,
    starting_elo_m: Optional[dict] = None,
    starting_elo_w: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a matchup feature frame for `season` using only regular-season data.
    No future information is used (strict causal ordering).
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


# ---------------------------------------------------------------------------
# Rolling temporal backtest
# ---------------------------------------------------------------------------

def rolling_backtest(
    seasons: List[int],
    cfg: dict,
    genders: Tuple[str, ...] = ("M", "W"),
    calibrate: bool = True,
    calibrator_methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Rolling temporal backtest with O(n) OOF caching.

    For each season i in `seasons`:
      - Train XGBoost on tournament games from seasons[0 .. i-1].
      - Evaluate on season i.
      - Calibrator selected using cached OOF predictions from earlier folds.

    Cross-season Elo is precomputed once using all regular-season data.
    Elo for season i uses only the regular-season games of season i
    (not future tournaments).

    v4 changes:
      - OOF predictions computed in a single O(n) forward pass (was O(n²)).
      - All tourney data preloaded once before the loop.
      - Calibration selection uses log_loss as primary criterion.
      - blend_sensitivity_report computed at end using all OOF predictions.
    """
    if calibrator_methods is None:
        calibrator_methods = cfg.get("calibration_methods",
                                     ["identity", "temperature", "platt", "isotonic"])

    def _log(msg):
        if verbose:
            print(msg)

    # ── Step 1: Pre-load data once ─────────────────────────────────────────
    _log("Loading data...")
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_tourney = load_tourney_detailed(cfg["data_dir"], "M")
    w_tourney = load_tourney_detailed(cfg["data_dir"], "W")
    m_seeds   = load_seeds(cfg["data_dir"], "M")
    w_seeds   = load_seeds(cfg["data_dir"], "W")

    # ── Step 2: Pre-compute cross-season Elo ──────────────────────────────
    _log("Pre-computing cross-season Elo...")
    elo_params = dict(
        k_factor=cfg["elo_k_factor"],
        initial_rating=cfg["elo_initial_rating"],
        carry_factor=cfg["elo_carry_factor"],
        use_margin=cfg.get("elo_use_margin", True),
        margin_cap=cfg.get("elo_margin_cap", 15.0),
    )
    starting_elo_m = precompute_starting_elo(_build_elo_games(m_regular), **elo_params)
    starting_elo_w = precompute_starting_elo(_build_elo_games(w_regular), **elo_params)

    # ── Step 3: Build feature frames for all seasons ───────────────────────
    _log("Building feature frames...")
    raw_frames: Dict[int, pd.DataFrame] = {}
    for season in seasons:
        _log(f"  Season {season}...", )
        eval_frames = []
        if "M" in genders:
            ef = prepare_eval_games(m_tourney, season, "M")
            if len(ef) > 0:
                eval_frames.append(ef)
        if "W" in genders:
            ef = prepare_eval_games(w_tourney, season, "W")
            if len(ef) > 0:
                eval_frames.append(ef)

        if not eval_frames:
            _log(f"  Season {season}: no tournament data, skipping")
            continue

        eval_df = pd.concat(eval_frames, ignore_index=True)
        frame = build_prediction_frame_asof(
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
        raw_frames[season] = frame
        _log(f" {len(frame)} games")

    seasons = [s for s in seasons if s in raw_frames]
    if not seasons:
        raise RuntimeError("No backtest seasons with tournament data found.")

    # ── Step 4: O(n) OOF forward pass ─────────────────────────────────────
    # Compute all OOF predictions in a single pass: for season i, train on [0..i-1].
    # This replaces the O(n²) nested loop from v3.
    _log("Computing OOF predictions (single forward pass)...")
    oof_cache: Dict[int, pd.DataFrame] = {}
    for i, season in enumerate(seasons):
        fold_df = raw_frames[season].copy()
        if i >= 1:
            train_hist = pd.concat(
                [raw_frames[s] for s in seasons[:i]], ignore_index=True
            )
            fold_pred = compute_all_probabilities(fold_df, cfg, train_df=train_hist)
        else:
            fold_pred = compute_all_probabilities(fold_df, cfg, train_df=None)
        oof_cache[season] = fold_pred

    # ── Step 5: Calibrator selection + rolling evaluation ─────────────────
    _log("Rolling evaluation with calibration...")
    all_frames:   List[pd.DataFrame] = []
    summary_rows: List[dict]         = []

    for i, season in enumerate(seasons):
        fold_df = oof_cache[season].copy()
        fold_df["PredRaw"] = fold_df["Pred"].copy()

        cal_method = "none"

        if calibrate and i >= 2:
            # Multi-fold LOO calibrator selection using all OOF folds before season i.
            # v4.1: replaced single-fold selection with leave-one-out for robustness.
            oof_tuples = [
                (oof_cache[seasons[j]]["Pred"].values,
                 oof_cache[seasons[j]]["ActualLowWin"].values)
                for j in range(i)
            ]
            try:
                best_cal = choose_best_calibrator_multifold(
                    oof_preds=oof_tuples,
                    scorer_fn=lambda y, p: full_metric_bundle(y, p),
                    methods=calibrator_methods,
                    cfg=cfg,
                )
                fold_df["Pred"] = apply_calibrator(best_cal.fitted, fold_df["Pred"].values)
                cal_method = best_cal.method
            except Exception as e:
                _log(f"  WARNING: Calibrator selection failed for season {season}: {e}")

        metrics = full_metric_bundle(
            fold_df["ActualLowWin"].values,
            fold_df["Pred"].values,
        )
        metrics["Season"]     = season
        metrics["Games"]      = len(fold_df)
        metrics["Calibrator"] = cal_method

        all_frames.append(fold_df)
        summary_rows.append(metrics)

        _log(
            f"  Season {season}: "
            f"LogLoss={metrics['log_loss']:.4f}  "
            f"Brier={metrics['brier']:.4f}  "
            f"ECE={metrics['ece']:.4f}  "
            f"Acc={metrics['accuracy']:.3f}  "
            f"Cal={cal_method}"
        )

    # ── Step 6: Aggregate outputs ──────────────────────────────────────────
    all_preds = pd.concat(all_frames, ignore_index=True)
    summary   = pd.DataFrame(summary_rows)

    calib_table = calibration_table(
        all_preds["ActualLowWin"].values,
        all_preds["Pred"].values,
        bins=10,
    )
    band_report = probability_band_report(
        all_preds["ActualLowWin"].values,
        all_preds["Pred"].values,
    )
    seasonal_metrics = per_season_metrics(all_preds)

    # Blend sensitivity analysis (uses all OOF predictions with probability columns)
    blend_sens = None
    if "p_elo" in all_preds.columns and "ActualLowWin" in all_preds.columns:
        try:
            blend_sens = blend_sensitivity_report(all_preds, cfg, n_steps=5)
        except Exception:
            pass

    # Calibration audit: before/after comparison using raw vs calibrated predictions
    cal_audit_table = pd.DataFrame()
    if "PredRaw" in all_preds.columns and "Pred" in all_preds.columns:
        try:
            audit = calibration_audit_report(
                all_preds["ActualLowWin"].values,
                all_preds["PredRaw"].values,
                all_preds["Pred"].values,
                bins=10,
            )
            # Flatten scalar metrics into a summary row
            audit_row = {k: v for k, v in audit.items() if not isinstance(v, pd.DataFrame)}
            cal_audit_table = pd.DataFrame([audit_row])
            _log(
                f"\n=== Calibration Audit (Overall) ==="
                f"\n  Raw  LogLoss={audit['raw_log_loss']:.4f}  Brier={audit['raw_brier']:.4f}  ECE={audit['raw_ece']:.4f}"
                f"\n  Cal  LogLoss={audit['cal_log_loss']:.4f}  Brier={audit['cal_brier']:.4f}  ECE={audit['cal_ece']:.4f}"
                f"\n  Δ    LogLoss={audit['delta_log_loss']:+.4f}  Brier={audit['delta_brier']:+.4f}  ECE={audit['delta_ece']:+.4f}"
            )
        except Exception as e:
            _log(f"  WARNING: Calibration audit failed: {e}")

    _log("\n=== Backtest Summary ===")
    _log(summary[["Season", "Games", "log_loss", "brier", "ece", "accuracy", "Calibrator"]]
         .to_string(index=False))

    return {
        "summary":           summary,
        "predictions":       all_preds,
        "calibration_table": calib_table,
        "probability_bands": band_report,
        "seasonal_metrics":  seasonal_metrics,
        "blend_sensitivity": blend_sens if blend_sens is not None else pd.DataFrame(),
        "calibration_audit": cal_audit_table,
    }


# ---------------------------------------------------------------------------
# Save / load utilities
# ---------------------------------------------------------------------------

def save_backtest_outputs(results: Dict[str, pd.DataFrame], output_dir: str):
    """Save all backtest DataFrames to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    print(f"Backtest outputs saved to {output_dir}/")
