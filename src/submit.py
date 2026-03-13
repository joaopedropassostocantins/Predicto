# src/submit.py — Submission generation for March ML Mania 2026 — v3.0

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
from src.calibration import apply_calibrator, choose_best_calibrator
from src.metrics import full_metric_bundle
from src.ratings import precompute_starting_elo


def generate_submission(cfg: dict, output_path: str = "submission.csv") -> pd.DataFrame:
    """
    Full submission pipeline for the target season.

    Steps
    -----
    1. Load all regular season + tourney data.
    2. Pre-compute cross-season Elo over the full history.
    3. Build features for every backtest season (with labels).
    4. Train XGBoost on ALL backtest seasons.
    5. Select best calibrator using a rolling held-out validation scheme.
    6. Build features for the target season (from the sample submission).
    7. Apply model + calibrator → write submission.csv.
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
    def _elo_games(df):
        return (
            df[df["Win"] == 1][["Season", "DayNum", "TeamID", "OppTeamID", "Margin"]]
            .rename(columns={"TeamID": "WTeamID", "OppTeamID": "LTeamID"})
            .drop_duplicates(subset=["Season", "DayNum", "WTeamID", "LTeamID"])
            .reset_index(drop=True)
        )

    starting_elo_m = precompute_starting_elo(
        _elo_games(m_regular),
        k_factor=cfg["elo_k_factor"],
        initial_rating=cfg["elo_initial_rating"],
        carry_factor=cfg["elo_carry_factor"],
        use_margin=cfg.get("elo_use_margin", False),
        margin_cap=cfg.get("elo_margin_cap", 30.0),
    )
    starting_elo_w = precompute_starting_elo(
        _elo_games(w_regular),
        k_factor=cfg["elo_k_factor"],
        initial_rating=cfg["elo_initial_rating"],
        carry_factor=cfg["elo_carry_factor"],
        use_margin=cfg.get("elo_use_margin", False),
        margin_cap=cfg.get("elo_margin_cap", 30.0),
    )

    # ── 3. Build labelled feature frames for backtest seasons ─────────────
    print("Building historical feature frames (with labels)...")
    backtest_seasons = cfg["backtest_seasons"]
    raw_frames: dict = {}

    for season in backtest_seasons:
        print(f"  Season {season}...", end=" ", flush=True)
        eval_frames = [
            prepare_eval_games(m_tourney, season, "M"),
            prepare_eval_games(w_tourney, season, "W"),
        ]
        eval_df = pd.concat([f for f in eval_frames if len(f) > 0], ignore_index=True)
        if len(eval_df) == 0:
            print("no tournament data, skipping")
            continue

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
    if len(available_seasons) == 0:
        raise RuntimeError("No backtest seasons with data found.  Check data_dir.")

    # ── 4. Select best calibrator using rolling held-out validation ────────
    print("Selecting calibrator via rolling validation...")
    best_cal = _select_calibrator(raw_frames, available_seasons, cfg)
    print(f"Best calibrator: {best_cal.method}  "
          f"(val Brier={best_cal.metrics['brier']:.4f})")

    # ── 5. Train final XGBoost on ALL backtest seasons ────────────────────
    print("Training final XGBoost on all backtest seasons...")
    all_train = pd.concat([raw_frames[s] for s in available_seasons], ignore_index=True)

    # ── 6. Build target-season features from sample submission ────────────
    target = cfg["target_season"]
    print(f"Building features for target season {target}...")

    sample_sub = load_sample_submission(cfg["data_dir"])
    sub_df     = parse_submission_ids(sample_sub)
    sub_target = sub_df[sub_df["Season"] == target].copy()

    if len(sub_target) == 0:
        # Fallback: generate all matchups from seeds if submission is empty
        print("  Sample submission has no rows for target season. Generating from seeds.")
        sub_target = _generate_all_matchups(target, m_seeds, w_seeds)

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

    # ── 7. Predict and calibrate ──────────────────────────────────────────
    pred_df = compute_all_probabilities(pred_df, cfg, train_df=all_train)
    pred_df["Pred"] = apply_calibrator(best_cal.fitted, pred_df["Pred"].values)

    # Validate output
    assert pred_df["Pred"].isna().sum() == 0, "NaN values in predictions!"
    assert (pred_df["Pred"] >= 0).all() and (pred_df["Pred"] <= 1).all(), \
        "Predictions out of [0, 1] range!"

    submission = pred_df[["ID", "Pred"]].copy()
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}  ({len(submission)} rows)")
    print(f"Pred stats: min={submission['Pred'].min():.4f}  "
          f"max={submission['Pred'].max():.4f}  "
          f"mean={submission['Pred'].mean():.4f}")
    return submission


# ---------------------------------------------------------------------------
# Calibrator selection helper
# ---------------------------------------------------------------------------

def _select_calibrator(raw_frames: dict, seasons: list, cfg: dict):
    """
    Select the best calibrator using a rolling held-out scheme.

    We use the LAST season in `seasons` as the validation set.
    XGBoost is trained on all preceding seasons to produce the
    validation predictions that the calibrators are evaluated on.
    """
    from src.calibration import choose_best_calibrator
    from src.models import compute_all_probabilities

    if len(seasons) < 2:
        # Not enough seasons for calibration; return identity
        from src.calibration import fit_calibrator
        dummy_p = np.full(10, 0.5)
        dummy_y = np.array([0, 1] * 5)
        cal = fit_calibrator("identity", dummy_p, dummy_y, cfg=cfg)
        from src.calibration import CalibrationResult
        return CalibrationResult(
            method="identity", fitted=cal,
            metrics={"brier": 0.25, "accuracy": 0.5, "log_loss": 0.693}
        )

    val_season   = seasons[-1]
    train_seasons = seasons[:-1]

    # OOF predictions for calibrator training (rolling)
    cal_oof = []
    for i, s in enumerate(train_seasons):
        fold = raw_frames[s].copy()
        if i >= 1:
            hist = pd.concat([raw_frames[ss] for ss in train_seasons[:i]], ignore_index=True)
            fold = compute_all_probabilities(fold, cfg, train_df=hist)
        else:
            fold = compute_all_probabilities(fold, cfg, train_df=None)
        cal_oof.append(fold)
    cal_train_df = pd.concat(cal_oof, ignore_index=True)

    # Validation predictions: XGB trained on all train_seasons
    val_hist = pd.concat([raw_frames[s] for s in train_seasons], ignore_index=True)
    val_df   = compute_all_probabilities(raw_frames[val_season].copy(), cfg, train_df=val_hist)

    return choose_best_calibrator(
        p_train=cal_train_df["Pred"].values,
        y_train=cal_train_df["ActualLowWin"].values,
        p_valid=val_df["Pred"].values,
        y_valid=val_df["ActualLowWin"].values,
        scorer_fn=lambda y, p: full_metric_bundle(y, p),
        methods=cfg.get("calibration_methods", ["identity", "temperature", "platt", "isotonic"]),
        cfg=cfg,
    )


def _generate_all_matchups(season: int, m_seeds: pd.DataFrame, w_seeds: pd.DataFrame) -> pd.DataFrame:
    """Generate all possible tournament matchup pairs from seed data."""
    rows = []
    for gender, seeds_df in [("M", m_seeds), ("W", w_seeds)]:
        teams = sorted(seeds_df[seeds_df["Season"] == season]["TeamID"].unique())
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                rows.append({
                    "ID":        f"{season}_{t1}_{t2}",
                    "Season":    season,
                    "TeamIDLow": t1,
                    "TeamIDHigh": t2,
                    "Gender":    gender,
                    "LowHome":   0,
                    "HighHome":  0,
                })
    return pd.DataFrame(rows)


def main():
    submission = generate_submission(CONFIG, output_path="submission.csv")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
