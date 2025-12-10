import numpy as np
from collections import Counter
import math

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

def avg_token_length(vocab):
    toks = [t for t in vocab if t != "▁"]
    if not toks:
        return 0.0
    return float(np.mean([len(t) for t in toks]))

def token_length_entropy(vocab):
    lens = [len(t) for t in vocab if t != "▁"]
    if not lens:
        return 0.0
    ctr = Counter(lens)
    total = sum(ctr.values())
    p = np.array([c/total for c in ctr.values()], dtype=float)
    p = p[p>0]
    return float(-np.sum(p * np.log2(p)))

def kl_divergence(p_hist, q_hist):
    # p and q are dict: token -> freq
    # convert to aligned arrays
    keys = set(p_hist) | set(q_hist)
    p = np.array([p_hist.get(k, 0) for k in keys], dtype=float)
    q = np.array([q_hist.get(k, 0) for k in keys], dtype=float)
    p = p / max(1.0, p.sum())
    q = q / max(1.0, q.sum())
    # smooth
    eps = 1e-9
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log2(p / q)))
