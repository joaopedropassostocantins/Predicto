# src/model.py
# SUBSTITUA O ARQUIVO INTEIRO

import numpy as np
from scipy.stats import poisson


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_probs(p, lo, hi):
    return np.clip(p, lo, hi)


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
        if feature_name not in out.columns:
            raise KeyError(f"Feature ausente no dataframe: {feature_name}")
        out["score_raw"] += weight * out[feature_name]

    out["p_manual"] = sigmoid(out["score_raw"] / cfg["temperature_manual"])
    out["p_manual"] = clip_probs(out["p_manual"], cfg["pred_clip_min"], cfg["pred_clip_max"])
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

        p = np.clip(p, 1e-9, 1 - 1e-9)
        logit = np.log(p / (1 - p))
        p_adj = sigmoid(logit / cfg["temperature_poisson"])
        p_list.append(float(clip_probs(p_adj, cfg["pred_clip_min"], cfg["pred_clip_max"])))

    out["p_poisson"] = p_list
    return out


def compute_seed_probability(df, cfg):
    out = df.copy()
    out["p_seed"] = sigmoid(out["seed_diff"] / cfg["seed_temperature"])
    out["p_seed"] = clip_probs(out["p_seed"], cfg["pred_clip_min"], cfg["pred_clip_max"])
    return out


def compute_rank_probability(df, cfg):
    out = df.copy()

    if "rank_diff" not in out.columns:
        out["p_rank"] = 0.5
        return out

    # Menor rank = melhor, então inverter o sinal
    out["p_rank"] = sigmoid((-out["rank_diff"]) / cfg["rank_temperature"])
    out["p_rank"] = clip_probs(out["p_rank"], cfg["pred_clip_min"], cfg["pred_clip_max"])
    return out


def blend_probabilities(df, cfg):
    out = df.copy()

    weights = {
        "manual": cfg["blend_manual"],
        "poisson": cfg["blend_poisson"],
        "seed": cfg["blend_seed"],
        "rank": cfg.get("blend_rank", 0.0),
    }

    active = {
        "manual": out["p_manual"],
        "poisson": out["p_poisson"],
        "seed": out["p_seed"],
        "rank": out["p_rank"],
    }

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("A soma dos pesos do blend deve ser > 0.")

    for k in weights:
        weights[k] /= total

    out["Pred"] = (
        weights["manual"] * active["manual"]
        + weights["poisson"] * active["poisson"]
        + weights["seed"] * active["seed"]
        + weights["rank"] * active["rank"]
    )
    out["Pred"] = clip_probs(out["Pred"], cfg["pred_clip_min"], cfg["pred_clip_max"])
    return out


def compute_all_probabilities(df, cfg):
    out = compute_manual_probability(df, cfg)
    out = compute_poisson_probability(out, cfg)
    out = compute_seed_probability(out, cfg)
    out = compute_rank_probability(out, cfg)
    out = blend_probabilities(out, cfg)
    return out
