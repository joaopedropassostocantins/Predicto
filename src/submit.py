# src/submit.py — Submission generation v4.1
#
# Changes from v4.0:
#   - _select_calibrator_from_oof: replaced single-fold with multi-fold LOO.
#     Old: only seasons[-1] used as validation (fragile with small n).
#     New: choose_best_calibrator_multifold() — mean log_loss over LOO folds.
#   - Calibration audit exported alongside submission.
#
# Changes from v3:
#   - Calibrator selection: log_loss as primary criterion (was Brier).
#   - validate_submission() added: checks format, range, NaN before writing.

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data import (
    load_regular_season_detailed,
    load_tourney_detailed,
    load_seeds,
    load_sample_submission,
    parse_submission_ids,
    prepare_eval_games,
)
from src.features import build_team_features, attach_team_features, make_matchup_features
from src.models import compute_all_probabilities
from src.calibration import (
    apply_calibrator,
    choose_best_calibrator,
    choose_best_calibrator_multifold,
    calibration_audit_report,
)
from src.metrics import full_metric_bundle
from src.ratings import precompute_starting_elo


# ---------------------------------------------------------------------------
# Submission generation pipeline
# ---------------------------------------------------------------------------

def generate_submission(cfg: dict, output_path: str = "submission.csv") -> pd.DataFrame:
    """
    Full submission generation pipeline for the target season.

    Steps
    -----
    1. Load all regular-season + tourney data.
    2. Pre-compute cross-season Elo over full history.
    3. Build labelled feature frames for all backtest seasons.
    4. Select best calibrator via rolling held-out validation (log_loss criterion).
    5. Train final XGBoost on ALL backtest seasons.
    6. Build features for target season (from sample submission).
    7. Apply model + calibrator → validate → write submission.csv.
    """
    print(f"=== Generating submission for season {cfg['target_season']} ===")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    m_regular = load_regular_season_detailed(cfg["data_dir"], "M")
    w_regular = load_regular_season_detailed(cfg["data_dir"], "W")
    m_tourney = load_tourney_detailed(cfg["data_dir"], "M")
    w_tourney = load_tourney_detailed(cfg["data_dir"], "W")
    m_seeds   = load_seeds(cfg["data_dir"], "M")
    w_seeds   = load_seeds(cfg["data_dir"], "W")

    # ── 2. Cross-season Elo ───────────────────────────────────────────────
    print("Pre-computing cross-season Elo...")
    elo_params = dict(
        k_factor=cfg["elo_k_factor"],
        initial_rating=cfg["elo_initial_rating"],
        carry_factor=cfg["elo_carry_factor"],
        use_margin=cfg.get("elo_use_margin", True),
        margin_cap=cfg.get("elo_margin_cap", 15.0),
    )

    def _elo_games(df):
        return (
            df[df["Win"] == 1][["Season", "DayNum", "TeamID", "OppTeamID", "Margin"]]
            .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
            .drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID"])
            .reset_index(drop=True)
        )

    starting_elo_m = precompute_starting_elo(_elo_games(m_regular), **elo_params)
    starting_elo_w = precompute_starting_elo(_elo_games(w_regular), **elo_params)

    # ── 3. Build labelled feature frames for backtest seasons ──────────────
    print("Building historical feature frames (with labels)...")
    backtest_seasons = cfg["backtest_seasons"]
    raw_frames: dict = {}

    for season in backtest_seasons:
        print(f"  Season {season}...", end=" ", flush=True)
        eval_frames = []
        for g, tourney in [("M", m_tourney), ("W", w_tourney)]:
            ef = prepare_eval_games(tourney, season, g)
            if len(ef) > 0:
                eval_frames.append(ef)

        if not eval_frames:
            print("no tournament data, skipping")
            continue

        eval_df = pd.concat(eval_frames, ignore_index=True)
        m_feat = build_team_features(
            m_regular, season, cfg["recent_games_window"], cfg["alpha_ci"],
            starting_elo=starting_elo_m, cfg=cfg,
        )
        w_feat = build_team_features(
            w_regular, season, cfg["recent_games_window"], cfg["alpha_ci"],
            starting_elo=starting_elo_w, cfg=cfg,
        )
        frame = attach_team_features(eval_df, m_feat, w_feat, m_seeds, w_seeds, cfg)
        frame = make_matchup_features(frame, cfg=cfg)
        raw_frames[season] = frame
        print(f"{len(frame)} games")

    available_seasons = sorted(raw_frames.keys())
    if not available_seasons:
        raise RuntimeError("No backtest seasons with data found. Check data_dir.")

    # ── 4. Compute OOF predictions for calibrator selection ────────────────
    print("Computing OOF predictions for calibrator selection...")
    oof_cache: dict = {}
    for i, season in enumerate(available_seasons):
        fold = raw_frames[season].copy()
        if i >= 1:
            hist = pd.concat([raw_frames[s] for s in available_seasons[:i]], ignore_index=True)
            oof_cache[season] = compute_all_probabilities(fold, cfg, train_df=hist)
        else:
            oof_cache[season] = compute_all_probabilities(fold, cfg, train_df=None)

    # Select calibrator using last season as validation, all others as training
    print("Selecting calibrator...")
    best_cal = _select_calibrator_from_oof(oof_cache, available_seasons, cfg)
    print(
        f"Best calibrator: {best_cal.method}  "
        f"(val LogLoss={best_cal.metrics.get('log_loss', float('nan')):.4f}  "
        f"Brier={best_cal.metrics.get('brier', float('nan')):.4f})"
    )

    # ── 5. Train final model on ALL backtest seasons ───────────────────────
    print("Training final model on all backtest seasons...")
    all_train = pd.concat([raw_frames[s] for s in available_seasons], ignore_index=True)

    # ── 6. Build target-season features ───────────────────────────────────
    target = cfg["target_season"]
    print(f"Building features for target season {target}...")

    sample_sub = load_sample_submission(cfg["data_dir"])
    sub_df     = parse_submission_ids(sample_sub)
    sub_target = sub_df[sub_df["Season"] == target].copy()

    if len(sub_target) == 0:
        print("  Sample submission has no rows for target season. Generating from seeds.")
        sub_target = _generate_all_matchups(target, m_seeds, w_seeds)
        if len(sub_target) == 0:
            raise ValueError(
                f"No matchup pairs found for season {target}. "
                f"MNCAATourneySeeds.csv has no seeds for {target} and no sample submission row matched. "
                f"Use SampleSubmissionStage2.csv which contains the actual {target} bracket pairs."
            )

    m_feat_t = build_team_features(
        m_regular, target, cfg["recent_games_window"], cfg["alpha_ci"],
        starting_elo=starting_elo_m, cfg=cfg,
    )
    w_feat_t = build_team_features(
        w_regular, target, cfg["recent_games_window"], cfg["alpha_ci"],
        starting_elo=starting_elo_w, cfg=cfg,
    )

    pred_df = attach_team_features(sub_target, m_feat_t, w_feat_t, m_seeds, w_seeds, cfg)
    pred_df = make_matchup_features(pred_df, cfg=cfg)

    # ── 7. Predict + calibrate + validate ─────────────────────────────────
    pred_df = compute_all_probabilities(pred_df, cfg, train_df=all_train)
    pred_df["PredRaw"] = pred_df["Pred"].copy()
    pred_df["Pred"]    = apply_calibrator(best_cal.fitted, pred_df["Pred"].values)

    # Validate
    validate_submission(pred_df)

    # Calibration audit on OOF (before vs after on training seasons)
    all_oof = pd.concat([oof_cache[s] for s in available_seasons], ignore_index=True)
    if "PredRaw" not in all_oof.columns:
        all_oof["PredRaw"] = all_oof["Pred"].copy()
    all_oof["PredCal"] = apply_calibrator(best_cal.fitted, all_oof["Pred"].values)

    try:
        audit = calibration_audit_report(
            all_oof["ActualLowWin"].values,
            all_oof["PredRaw"].values,
            all_oof["PredCal"].values,
            bins=10,
        )
        print(
            f"\n=== Calibration Audit (OOF Training Seasons) ==="
            f"\n  Raw  LogLoss={audit['raw_log_loss']:.4f}  "
            f"Brier={audit['raw_brier']:.4f}  ECE={audit['raw_ece']:.4f}"
            f"\n  Cal  LogLoss={audit['cal_log_loss']:.4f}  "
            f"Brier={audit['cal_brier']:.4f}  ECE={audit['cal_ece']:.4f}"
            f"\n  Δ    LogLoss={audit['delta_log_loss']:+.4f}  "
            f"Brier={audit['delta_brier']:+.4f}  ECE={audit['delta_ece']:+.4f}"
        )
    except Exception as e:
        print(f"  WARNING: Calibration audit failed: {e}")

    # Write
    submission = pred_df[["ID", "Pred"]].copy()
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved: {output_path}  ({len(submission)} rows)")
    print(f"Pred range: [{submission['Pred'].min():.4f}, {submission['Pred'].max():.4f}]  "
          f"mean={submission['Pred'].mean():.4f}")
    return submission


# ---------------------------------------------------------------------------
# Calibrator selection (log_loss primary)
# ---------------------------------------------------------------------------

def _select_calibrator_from_oof(oof_cache: dict, seasons: list, cfg: dict):
    """
    Select best calibrator using multi-fold LOO from all OOF predictions.

    v4.1: Replaced single-fold validation with leave-one-out across all seasons.
    This is more robust when n_seasons is small (5-10 typical for NCAAB).

    Old v4.0: used only seasons[-1] as single validation fold (fragile).
    New v4.1: choose_best_calibrator_multifold() averages log_loss over LOO folds.
    """
    from src.calibration import fit_calibrator, CalibrationResult

    if len(seasons) < 2:
        cal = fit_calibrator("identity", np.full(10, 0.5), np.array([0, 1] * 5), cfg=cfg)
        return CalibrationResult(
            method="identity", fitted=cal,
            metrics={"brier": 0.25, "log_loss": 0.693, "accuracy": 0.5, "ece": 0.1, "auc": 0.5},
        )

    # Build OOF tuples (pred, label) for each season in chronological order
    oof_tuples = [
        (oof_cache[s]["Pred"].values, oof_cache[s]["ActualLowWin"].values)
        for s in seasons
    ]

    return choose_best_calibrator_multifold(
        oof_preds=oof_tuples,
        scorer_fn=lambda y, p: full_metric_bundle(y, p),
        methods=cfg.get("calibration_methods", ["identity", "temperature", "platt", "isotonic"]),
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Validation and fallback helpers
# ---------------------------------------------------------------------------

def validate_submission(pred_df: pd.DataFrame) -> None:
    """
    Validate submission frame before writing.

    Checks:
      - No NaN in Pred column.
      - All predictions in [0, 1].
      - ID column present.
    Raises AssertionError on any violation.
    """
    assert "Pred" in pred_df.columns, "Missing 'Pred' column"
    assert "ID"   in pred_df.columns, "Missing 'ID' column"
    n_nan = int(pred_df["Pred"].isna().sum())
    assert n_nan == 0, f"{n_nan} NaN values in predictions"
    assert (pred_df["Pred"] >= 0).all() and (pred_df["Pred"] <= 1).all(), \
        "Predictions out of [0, 1] range"
    assert len(pred_df) > 0, "Empty submission"


def _generate_all_matchups(
    season: int,
    m_seeds: pd.DataFrame,
    w_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """Generate all possible tournament matchup pairs from seed data (fallback)."""
    rows = []
    for gender, seeds_df in [("M", m_seeds), ("W", w_seeds)]:
        teams = sorted(seeds_df[seeds_df["Season"] == season]["TeamID"].unique())
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                rows.append({
                    "ID":         f"{season}_{t1}_{t2}",
                    "Season":     season,
                    "TeamIDLow":  t1,
                    "TeamIDHigh": t2,
                    "Gender":     gender,
                    "LowHome":    0,
                    "HighHome":   0,
                })
    return pd.DataFrame(rows)


def main():
    submission = generate_submission(CONFIG, output_path="submission.csv")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
