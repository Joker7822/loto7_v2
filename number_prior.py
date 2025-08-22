# number_prior.py
import numpy as np

def blend_prior(z_freq, recent_weight=0.6, global_weight=0.4, temperature=0.8):
    # z_freq: dict {"global": np.ndarray(37), "recent": np.ndarray(37)}
    z = global_weight * z_freq["global"] + recent_weight * z_freq["recent"]
    # 温度ソフトマックスで確率化（小さいほど鋭い）
    p = np.exp(z / max(1e-9, temperature))
    p = p / p.sum()
    return p  # shape (37,)

def apply_penalties(p, rules=None):
    # ルール違反を直接除外しない。まずは番号確率に軽い平滑化のみ適用（本番は組合せ段階でペナルティ）
    return p / p.sum()
