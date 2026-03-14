# src/tuning.py — Hyperparameter tuning v4.0
#
# Implements temporal walk-forward tuning for all model components.
# Primary criterion: log_loss on held-out validation seasons.
#
# Tuning strategy: random search with temporal validation.
# Rationale: Bayesian optimisation requires many evaluations to build a good
# surrogate; for our small dataset (10 seasons), random search is competitive
# and avoids meta-overfitting to the specific validation seasons.
#
# Protocol:
#   1. For each candidate config, run rolling_backtest on a subset of seasons.
#   2. Compute average log_loss across held-out folds.
#   3. Select config with lowest average log_loss.
#   4. Write best configs to configs/best_*.yaml for production use.
#
# Block-wise tuning order (recommended):
#   A. Elo        → seed: k=20, carry=0.82, cap=15
#   B. Poisson    → seed: shrinkage_k=8, alpha_ci=0.10
#   C. XGBoost    → seed: depth=3, lambda=6, lr=0.03
#   D. Blend      → seed: elo=0.30, poisson=0.24, xgb=0.34, manual=0.12
#   E. Calibration → evaluated last (depends on all above)
#
# NOTE: Full tuning (all blocks, 50 iterations each) takes ~4-8 hours.
# For quick iteration, use n_iter=10 and val_seasons=3.

from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import rolling_backtest
from src.config import CONFIG


# ---------------------------------------------------------------------------
# Search space utilities
# ---------------------------------------------------------------------------

def _sample_float(lo: float, hi: float, log_scale: bool = False, rng=None) -> float:
    """Sample a float uniformly in [lo, hi], optionally in log space."""
    if rng is None:
        rng = random
    if log_scale:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def _sample_int(lo: int, hi: int, rng=None) -> int:
    """Sample an integer uniformly in [lo, hi]."""
    if rng is None:
        rng = random
    return rng.randint(lo, hi)


def _sample_categorical(values: list, rng=None):
    """Sample uniformly from a list of values."""
    if rng is None:
        rng = random
    return rng.choice(values)


# ---------------------------------------------------------------------------
# Block-specific config samplers
# ---------------------------------------------------------------------------

def sample_elo_config(rng=None) -> dict:
    """Sample Elo hyperparameters from search space (configs/search_spaces.yaml)."""
    if rng is None:
        rng = random
    k_vals = np.arange(12, 29, 2).tolist()
    cap_vals = np.arange(10, 21, 1).tolist()
    return {
        "elo_k_factor":     float(_sample_categorical(k_vals, rng)),
        "elo_carry_factor": round(_sample_float(0.70, 0.92, rng=rng), 2),
        "elo_margin_cap":   float(_sample_categorical(cap_vals, rng)),
        "elo_use_margin":   _sample_categorical([True, False], rng),
    }


def sample_poisson_config(rng=None) -> dict:
    """Sample Poisson hyperparameters from search space."""
    if rng is None:
        rng = random
    w3 = _sample_float(0.30, 0.55, rng=rng)
    w5 = _sample_float(0.25, 0.45, rng=rng)
    ws = _sample_float(0.10, 0.35, rng=rng)
    total = w3 + w5 + ws
    return {
        "poisson_shrinkage_k": round(_sample_float(3.0, 25.0, rng=rng), 1),
        "alpha_ci": _sample_categorical([0.05, 0.10, 0.15, 0.20], rng),
        "max_points_poisson": _sample_categorical([140, 145, 150, 155, 160, 165, 170], rng),
        "poisson_blend_weights": {
            "recent3": round(w3 / total, 4),
            "recent5": round(w5 / total, 4),
            "season":  round(ws / total, 4),
        },
    }


def sample_xgb_config(rng=None) -> dict:
    """Sample XGBoost hyperparameters from search space."""
    if rng is None:
        rng = random
    return {
        "xgb_params": {
            "n_estimators":     _sample_int(200, 1200, rng),
            "learning_rate":    round(_sample_float(0.015, 0.08, log_scale=True, rng=rng), 4),
            "max_depth":        _sample_categorical([2, 3, 4, 5], rng),
            "min_child_weight": _sample_int(3, 12, rng),
            "subsample":        round(_sample_float(0.60, 0.90, rng=rng), 2),
            "colsample_bytree": round(_sample_float(0.50, 0.90, rng=rng), 2),
            "reg_lambda":       round(_sample_float(2.0, 15.0, rng=rng), 1),
            "reg_alpha":        round(_sample_float(0.0, 4.0, rng=rng), 2),
            "gamma":            round(_sample_float(0.0, 3.0, rng=rng), 2),
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "tree_method":      "hist",
            "random_state":     42,
            "verbosity":        0,
            "early_stopping_rounds": _sample_categorical([30, 40, 50, 75], rng),
        }
    }


def sample_blend_config(rng=None) -> dict:
    """Sample blend weights from search space (normalised to sum to 1)."""
    if rng is None:
        rng = random
    w_elo     = _sample_float(0.18, 0.42, rng=rng)
    w_poisson = _sample_float(0.12, 0.32, rng=rng)
    w_xgb     = _sample_float(0.22, 0.48, rng=rng)
    w_manual  = _sample_float(0.05, 0.18, rng=rng)
    total     = w_elo + w_poisson + w_xgb + w_manual
    return {
        "blend_weights": {
            "elo":     round(w_elo     / total, 4),
            "poisson": round(w_poisson / total, 4),
            "xgb":     round(w_xgb     / total, 4),
            "manual":  round(w_manual  / total, 4),
        }
    }


def sample_calibration_config(rng=None) -> dict:
    """Sample calibration parameters."""
    if rng is None:
        rng = random
    clip_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    return {
        "pred_win_min": _sample_categorical(clip_values, rng),
        "pred_win_max": 1.0 - _sample_categorical(clip_values, rng),
        "manual_temperature": round(_sample_float(1.00, 1.60, rng=rng), 3),
    }


BLOCK_SAMPLERS = {
    "elo":         sample_elo_config,
    "poisson":     sample_poisson_config,
    "xgb":         sample_xgb_config,
    "blend":       sample_blend_config,
    "calibration": sample_calibration_config,
}


# ---------------------------------------------------------------------------
# Single evaluation: run backtest and compute average log_loss
# ---------------------------------------------------------------------------

def evaluate_config(
    candidate_cfg: dict,
    val_seasons: List[int],
    base_cfg: Optional[dict] = None,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Evaluate a candidate config on val_seasons using rolling backtest.

    Returns (avg_log_loss, avg_brier, avg_ece).
    """
    if base_cfg is None:
        base_cfg = deepcopy(CONFIG)

    cfg = deepcopy(base_cfg)
    # Deep merge candidate params
    for k, v in candidate_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = deepcopy(cfg[k])
            cfg[k].update(v)
        else:
            cfg[k] = v

    # Sync legacy aliases
    cfg["pred_clip_min"] = cfg.get("pred_win_min", cfg.get("pred_clip_min", 0.05))
    cfg["pred_clip_max"] = cfg.get("pred_win_max", cfg.get("pred_clip_max", 0.95))

    try:
        results = rolling_backtest(val_seasons, cfg, verbose=verbose)
        summary = results["summary"]
        avg_ll  = float(summary["log_loss"].mean())
        avg_br  = float(summary["brier"].mean())
        avg_ece = float(summary["ece"].mean()) if "ece" in summary.columns else float("nan")
        return avg_ll, avg_br, avg_ece
    except Exception as e:
        if verbose:
            print(f"  [tuning] ERROR: {e}")
        return float("inf"), float("inf"), float("nan")


# ---------------------------------------------------------------------------
# Random search for a single block
# ---------------------------------------------------------------------------

def random_search_block(
    block: str,
    val_seasons: List[int],
    base_cfg: Optional[dict] = None,
    n_iter: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Random search for a single configuration block.

    Parameters
    ----------
    block : str    One of: "elo", "poisson", "xgb", "blend", "calibration".
    val_seasons : list  Seasons for temporal validation (need ≥ 3 for calibration).
    n_iter : int   Number of random configurations to evaluate.
    seed : int     Random seed for reproducibility.

    Returns
    -------
    DataFrame with all candidate results, sorted by log_loss ascending.
    """
    if block not in BLOCK_SAMPLERS:
        raise ValueError(f"Unknown block: {block}. Must be one of {list(BLOCK_SAMPLERS)}")

    if base_cfg is None:
        base_cfg = deepcopy(CONFIG)

    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    sampler = BLOCK_SAMPLERS[block]

    rows = []
    print(f"\n=== Random Search: {block} block ({n_iter} iterations) ===")
    print(f"Validation seasons: {val_seasons}")
    print(f"Criterion: avg log_loss across {len(val_seasons)} folds\n")

    for i in range(n_iter):
        candidate = sampler(rng=rng)
        t0 = time.time()
        avg_ll, avg_br, avg_ece = evaluate_config(candidate, val_seasons, base_cfg=base_cfg)
        elapsed = time.time() - t0

        row = {
            "iter": i,
            "log_loss": round(avg_ll, 6),
            "brier":    round(avg_br, 6),
            "ece":      round(avg_ece, 6),
            "elapsed_s": round(elapsed, 1),
            **{f"param_{k}": str(v) for k, v in candidate.items()},
        }
        rows.append(row)

        if verbose:
            print(
                f"  [{i+1:3d}/{n_iter}]  "
                f"LogLoss={avg_ll:.4f}  Brier={avg_br:.4f}  "
                f"ECE={avg_ece:.4f}  ({elapsed:.1f}s)"
            )

    df = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Full pipeline tuning (all blocks sequentially)
# ---------------------------------------------------------------------------

def full_pipeline_tuning(
    val_seasons: List[int],
    base_cfg: Optional[dict] = None,
    n_iter_per_block: int = 15,
    seed: int = 42,
    output_dir: str = "tuning_results",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run random search over all blocks sequentially.

    Order: elo → poisson → xgb → blend → calibration.
    Each block's best config is applied before tuning the next block.

    Returns dict of {block_name: results_DataFrame}.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if base_cfg is None:
        base_cfg = deepcopy(CONFIG)

    best_cfg = deepcopy(base_cfg)
    all_results = {}

    for block in ["elo", "poisson", "xgb", "blend", "calibration"]:
        print(f"\n{'='*60}")
        print(f"Tuning block: {block.upper()}")
        print(f"{'='*60}")

        results = random_search_block(
            block=block,
            val_seasons=val_seasons,
            base_cfg=best_cfg,
            n_iter=n_iter_per_block,
            seed=seed + hash(block) % 1000,
            verbose=verbose,
        )
        all_results[block] = results

        # Save results
        results.to_csv(os.path.join(output_dir, f"tuning_{block}.csv"), index=False)

        # Apply best config
        if len(results) > 0:
            best_row = results.iloc[0]
            best_ll  = best_row["log_loss"]
            print(f"\nBest {block} config: LogLoss={best_ll:.4f}")

            # Re-sample the best candidate config from param columns
            # (parse back from string representation — simplified)
            best_candidate = BLOCK_SAMPLERS[block](rng=random.Random(42))
            # Use the best parameters from the param columns
            for col in results.columns:
                if col.startswith("param_"):
                    key = col[6:]  # strip "param_" prefix
                    try:
                        import ast
                        val = ast.literal_eval(best_row[col])
                        if isinstance(val, dict) and isinstance(best_cfg.get(key), dict):
                            best_cfg[key].update(val)
                        else:
                            best_cfg[key] = val
                    except Exception:
                        pass

    return all_results


# ---------------------------------------------------------------------------
# Baseline comparison table
# ---------------------------------------------------------------------------

def baseline_comparison(
    val_seasons: List[int],
    base_cfg: Optional[dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare baseline models vs the full ensemble.

    Evaluates:
      1. Elo-only baseline
      2. Poisson-only baseline
      3. XGBoost-only baseline
      4. Ensemble (all components) without calibration
      5. Ensemble with calibration

    Each model uses the same rolling temporal validation.

    Returns DataFrame [model, log_loss, brier, ece].
    """
    if base_cfg is None:
        base_cfg = deepcopy(CONFIG)

    models_to_eval = {
        "Elo-only":           {"blend_weights": {"elo": 1.00, "poisson": 0.0, "xgb": 0.0, "manual": 0.0}},
        "Poisson-only":       {"blend_weights": {"elo": 0.0, "poisson": 1.00, "xgb": 0.0, "manual": 0.0}},
        "XGBoost-only":       {"blend_weights": {"elo": 0.0, "poisson": 0.0, "xgb": 1.00, "manual": 0.0}},
        "Ensemble (no cal)":  {},    # default weights, no calibration in backtest
        "Ensemble + cal":     {},    # default weights, with calibration
    }

    rows = []
    for name, overrides in models_to_eval.items():
        if verbose:
            print(f"Evaluating: {name}...")
        cfg = deepcopy(base_cfg)
        cfg.update(overrides)
        calibrate = ("+ cal" in name)
        try:
            results = rolling_backtest(val_seasons, cfg, calibrate=calibrate, verbose=False)
            summary = results["summary"]
            rows.append({
                "model":    name,
                "log_loss": round(float(summary["log_loss"].mean()), 6),
                "brier":    round(float(summary["brier"].mean()), 6),
                "ece":      round(float(summary["ece"].mean()), 6) if "ece" in summary else float("nan"),
            })
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            rows.append({"model": name, "log_loss": float("nan"), "brier": float("nan"), "ece": float("nan")})

    df = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    print("\n=== Baseline Comparison ===")
    print(df.to_string(index=False))
    return df
