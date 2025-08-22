
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

NumberSet = List[int]

def _compute_draw_feature_targets(history_df: pd.DataFrame) -> Dict[str, float]:
    """
    Infer soft targets from historical draws (uses the last ~150 by default if provided).
    Expects columns: '本数字' (list-like), '抽せん日' (datetime-like).
    """
    if history_df is None or history_df.empty or "本数字" not in history_df.columns:
        # Reasonable defaults for Loto7 (1..37 choose 7) based on rough empirical ranges.
        return {
            "odd_ratio": 0.5,    # target ~3-4 odd
            "sum": 140.0,        # mid of typical 7-number sum distribution
            "range": 22.0,       # distance between max and min
            "runs": 1.5          # count of consecutive pairs
        }

    df = history_df.copy()
    if "抽せん日" in df.columns:
        df = df.sort_values("抽せん日").tail(150)

    def features(nums: List[int]):
        nums = sorted(nums)
        arr = np.array(nums, dtype=int)
        odd_ratio = (arr % 2 != 0).sum() / 7.0
        total = int(arr.sum())
        rng = int(arr.max() - arr.min())
        diffs = np.diff(arr)
        runs = int((diffs == 1).sum())
        return odd_ratio, total, rng, runs

    odd_list, sum_list, range_list, run_list = [], [], [], []
    for nums in df["本数字"].tolist():
        if not isinstance(nums, (list, tuple)) or len(nums) != 7:
            continue
        o, s, r, rn = features(nums)
        odd_list.append(o)
        sum_list.append(s)
        range_list.append(r)
        run_list.append(rn)

    def robust_mean(xs, default):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else default

    return {
        "odd_ratio": robust_mean(odd_list, 0.5),
        "sum":       robust_mean(sum_list, 140.0),
        "range":     robust_mean(range_list, 22.0),
        "runs":      robust_mean(run_list, 1.5),
    }


def _set_features(nums: NumberSet):
    nums = sorted(nums)
    arr = np.array(nums, dtype=int)
    odd_ratio = (arr % 2 != 0).sum() / 7.0
    total = int(arr.sum())
    rng = int(arr.max() - arr.min())
    diffs = np.diff(arr)
    runs = int((diffs == 1).sum())
    return odd_ratio, total, rng, runs


def _penalty_against_targets(nums: NumberSet, targets: Dict[str, float]) -> float:
    """Quadratic penalties around soft targets. Smaller is better."""
    odd_ratio, total, rng, runs = _set_features(nums)
    p = 0.0
    p += 6.0 * (odd_ratio - targets["odd_ratio"]) ** 2
    p += 0.0008 * (total - targets["sum"]) ** 2
    p += 0.0025 * (rng - targets["range"]) ** 2
    p += 0.35 * (runs - targets["runs"]) ** 2
    return float(p)


def _beam_expand(frontiers, candidates, probs, targets, max_branch, hard_caps):
    """
    Expand beam with heuristic filtering.
    frontiers: list of (nums_sorted_tuple, score_without_penalty)
    returns updated list of (nums, score_with_penalty)
    """
    new_frontiers = []
    # Pre-pick promising next numbers to keep branching factor reasonable
    top_idx = np.argsort(probs)[::-1][:max_branch * 3]
    next_pool = [candidates[i] for i in top_idx]
    next_probs = probs[top_idx]

    for nums, base_score in frontiers:
        last = nums[-1] if nums else 0
        odd_cnt = sum(1 for n in nums if n % 2 == 1)
        even_cnt = len(nums) - odd_cnt
        run_cnt = 0
        if len(nums) >= 2:
            diffs = np.diff(nums)
            run_cnt = int((diffs == 1).sum())

        for n, p in zip(next_pool, next_probs):
            if n <= last:
                continue
            # hard caps (pruning):
            if len(nums) >= 1 and n == last + 1 and run_cnt >= hard_caps["max_runs"]:
                continue
            # projected odd/even feasibility
            new_odd = odd_cnt + (n % 2)
            new_len = len(nums) + 1
            # cannot exceed hard limits too early
            if new_odd > hard_caps["max_odd"] or (new_len - new_odd) > hard_caps["max_even"]:
                continue

            new_nums = nums + (n,)
            new_score = base_score + math.log(max(p, 1e-12))
            if len(new_nums) == 7:
                # final penalty
                pen = _penalty_against_targets(list(new_nums), targets)
                new_frontiers.append((new_nums, new_score - pen))
            else:
                # optimistic penalty using partial features (weak)
                new_frontiers.append((new_nums, new_score))
    return new_frontiers


class GachiSolver:
    """
    Constrained beam search that builds 7-number sets from calibrated marginals.
    - Uses soft penalties fitted from recent historical draws.
    - Enforces lightweight hard caps for feasibility and variety.
    """
    def __init__(self, beam_size: int = 256, max_branch: int = 16,
                 max_runs: int = 2, max_odd: int = 5, max_even: int = 5):
        self.beam_size = beam_size
        self.max_branch = max_branch
        self.hard_caps = {
            "max_runs": max_runs,
            "max_odd": max_odd,
            "max_even": max_even,
        }

    def _marginals_from_predictions(self, numbers_only: List[NumberSet], confidence_scores: List[float]) -> np.ndarray:
        """
        Reuse the same marginalization idea as existing _stable_diverse_selection:
        weight each candidate by confidence and count presence.
        """
        conf = np.array(confidence_scores, dtype=float)
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
        weights = np.exp(conf / 0.4)
        weights = weights / (weights.sum() + 1e-9)

        marg = np.zeros(37, dtype=float)
        for cand, w in zip(numbers_only, weights):
            for n in cand:
                marg[n - 1] += w

        # temperature + floor to keep exploration alive
        marg = marg + 1e-6
        marg = marg / marg.sum()
        return marg

    def propose_from_marginals(
        self,
        numbers_only: List[NumberSet],
        confidence_scores: List[float],
        latest_data: Optional[pd.DataFrame] = None,
        history_df: Optional[pd.DataFrame] = None,
        n_elite: int = 60,
        seed: Optional[int] = None
    ) -> List[Tuple[NumberSet, float]]:
        """
        Returns: list of (numbers, conf_boost) sorted by descending score.
        """
        if seed is not None:
            np.random.seed(seed)

        marg = self._marginals_from_predictions(numbers_only, confidence_scores)

        # Build soft targets from history
        targets = _compute_draw_feature_targets(history_df if history_df is not None else latest_data)

        # Beam search
        candidates = np.arange(1, 38, dtype=int)
        probs = marg.copy()

        # start with empty set
        frontiers = [(tuple(), 0.0)]
        for _ in range(7):
            expanded = _beam_expand(
                frontiers=frontiers,
                candidates=candidates,
                probs=probs,
                targets=targets,
                max_branch=self.max_branch,
                hard_caps=self.hard_caps,
            )
            # keep top beam_size partials
            expanded.sort(key=lambda x: x[1], reverse=True)
            frontiers = expanded[: self.beam_size]

        # filter to completed sets (len==7)
        finals = [f for f in frontiers if len(f[0]) == 7]
        # deduplicate
        seen = set()
        unique = []
        for nums, sc in finals:
            if nums not in seen:
                seen.add(nums)
                unique.append((list(nums), sc))

        # normalize score into [0, 0.25] as confidence boost
        if unique:
            scores = np.array([sc for _, sc in unique], dtype=float)
            s_min, s_max = float(scores.min()), float(scores.max())
            if s_max > s_min:
                conf_boosts = 0.25 * (scores - s_min) / (s_max - s_min)
            else:
                conf_boosts = np.zeros_like(scores)
            ranked = sorted([(nums, float(cb)) for (nums, _), cb in zip(unique, conf_boosts)],
                            key=lambda x: (-x[1], sum(x[0])))
        else:
            ranked = []

        return ranked[: n_elite]
