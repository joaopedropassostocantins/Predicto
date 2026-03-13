# src/model.py

import numpy as np
from scipy.stats import poisson


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def poisson_match_win_probability(lambda_a: float, lambda_b: float, max_points: int = 220) -> float:
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


def compute_manual_probability(df, cfg):
    out = df.copy()
    out["score_raw"] = 0.0

    for feature_name, weight in cfg["weights"].items():
        out["score_raw"] += weight * out[feature_name]

    out["p_manual"] = sigmoid(out["score_raw"] / cfg["temperature_manual"])
    out["p_manual"] = out["p_manual"].clip(cfg["pred_clip_min"], cfg["pred_clip_max"])
    return out


def compute_poisson_probability(df, cfg):
    out = df.copy()
    p_list = []

    for row in out.itertuples(index=False):
        p = poisson_match_win_probability(
            lambda_a=float(row.low_expected_points_matchup),
            lambda_b=float(row.high_expected_points_matchup),
            max_points=cfg["max_points_poisson"],
        )
        logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
        p_adj = sigmoid(logit / cfg["temperature_poisson"])
        p_list.append(float(np.clip(p_adj, cfg["pred_clip_min"], cfg["pred_clip_max"])))

    out["p_poisson"] = p_list
    return out


def compute_seed_probability(df, cfg):
    out = df.copy()
    out["p_seed"] = sigmoid(out["seed_diff"] / 3.0)
    out["p_seed"] = out["p_seed"].clip(cfg["pred_clip_min"], cfg["pred_clip_max"])
    return out


def blend_probabilities(df, cfg):
    out = df.copy()

    bm = cfg["blend_manual"]
    bp = cfg["blend_poisson"]
    bs = cfg["blend_seed"]
    total = bm + bp + bs
    bm, bp, bs = bm / total, bp / total, bs / total

    out["Pred"] = bm * out["p_manual"] + bp * out["p_poisson"] + bs * out["p_seed"]
    out["Pred"] = out["Pred"].clip(cfg["pred_clip_min"], cfg["pred_clip_max"])
    return out


def compute_all_probabilities(df, cfg):
    out = compute_manual_probability(df, cfg)
    out = compute_poisson_probability(out, cfg)
    out = compute_seed_probability(out, cfg)
    out = blend_probabilities(out, cfg)
    return out
