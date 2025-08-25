
import math
import numpy as np
import pandas as pd

def _safe_softmax(x, temperature=1.0):
    x = np.array(x, dtype=float)
    x = (x - x.max()) / max(1e-9, temperature)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-9)

def build_marginals_from_sets(candidates, confidences, extra_logits=None):
    """
    candidates: list[list[int]]  (1..37)
    confidences: list[float]
    extra_logits: list[np.ndarray]  each of shape (37,) optional
    Returns: np.ndarray shape (37,) of marginal probs in [0,1], sum ~ 1
    """
    n = 37
    # weight by softmax of confidences for stability
    w = _safe_softmax(confidences, temperature=0.7)
    marg = np.zeros(n, dtype=float)
    for cand, ww in zip(candidates, w):
        for num in cand:
            if 1 <= num <= n:
                marg[num-1] += ww
    if extra_logits:
        # combine via average after softmax
        extras = []
        for arr in extra_logits:
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if arr.size == n:
                extras.append(_safe_softmax(arr, temperature=1.0))
        if extras:
            e = np.mean(np.stack(extras, axis=0), axis=0)
            marg = 0.6 * marg / (marg.sum() + 1e-9) + 0.4 * e
    # normalize
    marg = marg / (marg.sum() + 1e-9)
    # floor to avoid -inf log
    marg = np.clip(marg, 1e-12, 1.0)
    return marg

def compute_pair_pmi(history_df):
    """
    history_df: DataFrame with column '本数字' as list[int]
    Returns: dict[(i,j)] -> PMI value, with 1<=i<j<=37
    """
    n = 37
    counts = np.zeros(n, dtype=float)
    pair_counts = np.zeros((n, n), dtype=float)
    total_rows = 0
    for nums in history_df['本数字']:
        if not isinstance(nums, (list, tuple)):
            # try to parse simple string like "1 2 3 4 5 6 7"
            try:
                s = str(nums).strip("[]").replace(",", " ")
                nums = [int(v) for v in s.split() if v.isdigit()]
            except Exception:
                continue
        nums = sorted(set([x for x in nums if 1 <= x <= n]))
        if len(nums) < 2:
            continue
        total_rows += 1
        for i in nums:
            counts[i-1] += 1
        for a in range(len(nums)):
            for b in range(a+1, len(nums)):
                i, j = nums[a]-1, nums[b]-1
                pair_counts[i, j] += 1

    if total_rows == 0:
        return {}

    # probabilities with Laplace smoothing
    p1 = (counts + 1.0) / (total_rows + n)
    p12 = (pair_counts + 1.0) / (total_rows + n*n/2.0)
    pmi = {}
    for i in range(n):
        for j in range(i+1, n):
            pmi[(i+1, j+1)] = math.log(p12[i, j] / (p1[i] * p1[j]))
    return pmi

def _score_set(nums, logp, pair_pmi=None, alpha_pair=0.3):
    # nums is sorted unique ints 1..37 length 7
    s = sum(logp[n-1] for n in nums)
    if pair_pmi and alpha_pair != 0.0:
        for a in range(len(nums)):
            for b in range(a+1, len(nums)):
                s += alpha_pair * pair_pmi.get((nums[a], nums[b]), 0.0)
    return s

def decode_topK_sets(marginals, pair_pmi=None, K=10, alpha_pair=0.3, beam_size=256):
    """
    Heuristic beam search:
      - Start with empty set, iteratively add numbers maximizing objective:
            sum(log p_i) + alpha_pair * sum(PMI pairs)
      - Keep top beam_size partials each step.
      - Finally return top K complete sets of size 7.
    Returns list of (set_list, score)
    """
    n = 37
    p = np.clip(np.array(marginals, dtype=float).reshape(-1), 1e-12, 1.0)
    logp = np.log(p)
    # initialize beam with singletons (top likely numbers)
    order = list(range(1, n+1))
    order.sort(key=lambda i: -p[i-1])
    beam = [([i], logp[i-1]) for i in order[:beam_size]]

    for step in range(1, 7):  # need total 7 numbers
        new_beam = []
        for chosen, score in beam:
            last = chosen[-1]
            start_idx = order.index(last) + 1 if last in order else 0
            for i in order[start_idx:]:
                if i in chosen:
                    continue
                cand = chosen + [i]
                cand.sort()
                sc = _score_set(cand, logp, pair_pmi, alpha_pair)
                new_beam.append((cand, sc))
        # prune
        new_beam.sort(key=lambda x: -x[1])
        beam = new_beam[:beam_size]
        if not beam:
            break

    # final top-K
    beam.sort(key=lambda x: -x[1])
    unique = []
    seen = set()
    for cand, sc in beam:
        tup = tuple(cand)
        if tup not in seen:
            unique.append((cand, sc))
            seen.add(tup)
        if len(unique) >= K:
            break
    return unique
