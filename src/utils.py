
import numpy as np
from src.config import CONFIG

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def clip_probs(p, lo=CONFIG["pred_clip_min"], hi=CONFIG["pred_clip_max"]):
    return np.clip(p, lo, hi)
