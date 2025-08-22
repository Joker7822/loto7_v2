
import numpy as np
from itertools import combinations
from math import log
try:
    from ortools.sat.python import cp_model
    HAS_OR_TOOLS = True
except Exception:
    HAS_OR_TOOLS = False


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x)
    e = np.exp(x)
    s = np.nansum(e) + 1e-9
    return e / s


def build_marginals_from_sets(candidates, confidences, extra_logits=None, temperature=0.35):
    """
    candidates: List[List[int]]  # 1..37 の7個セット
    confidences: List[float]
    extra_logits: Optional[List[np.ndarray]]  # 追加の37次元スコア（GNNやPPO等）
    return: np.ndarray shape (37,)  # 各番号の周辺確率スコア（正規化済）
    """
    if candidates is None or len(candidates) == 0:
        return np.ones(37) / 37.0

    # 信頼度 → 重み（温度付きsoftmax）
    conf = np.array(confidences, dtype=float)
    conf = (conf - np.nanmin(conf)) / (np.nanmax(conf) - np.nanmin(conf) + 1e-9)
    weights = _softmax(conf / max(1e-6, temperature))

    p = np.zeros(37, dtype=float)
    for cand, w in zip(candidates, weights):
        for n in cand:
            if 1 <= int(n) <= 37:
                p[int(n) - 1] += w

    if extra_logits:
        for logits in extra_logits:
            p += _softmax(logits)

    p = p / (p.sum() + 1e-9)
    return p


def compute_pair_pmi(past_draws):
    """
    過去データ（DataFrame with column '本数字' = List[int]）からペアPMI行列(37x37)を計算
    """
    nums = list(range(1, 38))
    count_single = {i: 1 for i in nums}  # +1 平滑化
    count_pair = {(i, j): 1 for i in nums for j in nums if i < j}
    N = 1  # +1 平滑化

    for row in past_draws['本数字']:
        if isinstance(row, (list, tuple)):
            row = [int(x) for x in row if 1 <= int(x) <= 37]
            s = set(row)
            for i in s:
                count_single[i] = count_single.get(i, 1) + 1
            for i, j in combinations(sorted(s), 2):
                count_pair[(i, j)] = count_pair.get((i, j), 1) + 1
            N += 1

    pmi = np.zeros((37, 37), dtype=float)
    for i, j in combinations(range(1, 38), 2):
        pij = count_pair[(i, j)] / N
        pi = count_single[i] / N
        pj = count_single[j] / N
        val = log((pij / (pi * pj) + 1e-9))
        pmi[i - 1, j - 1] = val
        pmi[j - 1, i - 1] = val
    return pmi


def _score_set(nums, p, pmi=None, alpha_pair=0.3):
    # 個別周辺確率の対数和 + ペアPMIの和（重み付き）
    s = sum(log(p[n - 1] + 1e-9) for n in nums)
    if pmi is not None and alpha_pair > 0:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                s += alpha_pair * pmi[nums[i] - 1, nums[j] - 1]
    return float(s)


def _validity_checks(nums):
    # 連番4個以上を禁止 / 範囲が狭すぎる（<12）・広すぎる（>34）を弱く制限 / 奇偶バランス(2-5~5-2)
    nums = sorted(nums)
    consec = 1
    max_consec = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    if max_consec >= 4:
        return False
    r = nums[-1] - nums[0]
    if r < 12 or r > 34:
        return False
    odd = sum(n % 2 for n in nums)
    if not (2 <= odd <= 5):
        return False
    return True


def decode_topK_sets(p, pmi, K=10, alpha_pair=0.3, method='ilp'):
    """
    p: shape (37,) 周辺確率
    pmi: shape (37,37) ペアPMI
    return: List[Tuple[List[int], float]]  # (組合せ, スコア)
    """
    K = max(1, int(K))

    if method == 'ilp' and HAS_OR_TOOLS:
        # ILP: maximize Σ log(p_i) x_i + α Σ PMI_{ij} y_ij
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i+1}") for i in range(37)]
        y = {}
        for i in range(37):
            for j in range(i + 1, 37):
                y[(i, j)] = model.NewBoolVar(f"y_{i+1}_{j+1}")
                # 線形化制約
                model.Add(y[(i, j)] <= x[i])
                model.Add(y[(i, j)] <= x[j])
                model.Add(y[(i, j)] >= x[i] + x[j] - 1)

        model.Add(sum(x) == 7)

        # 軟制約（連番や範囲など）は目的関数ペナルティで入れる
        BIG = 1e-2  # 目的関数のスケールに合わせた弱いペナルティ
        # 連番ペナルティ
        for i in range(36):
            model.Add(x[i] + x[i + 1] <= 2)  # 形式的に許容。強制しない

        # 目的関数
        obj = 0
        for i in range(37):
            coeff = float(log(p[i] + 1e-9))
            obj += coeff * x[i]
        for i in range(37):
            for j in range(i + 1, 37):
                coeff = float(alpha_pair * pmi[i, j])
                obj += coeff * y[(i, j)]
        # 小さなペナルティで分散を促す（端に寄りすぎない）
        for i in range(37):
            center_pen = abs((i + 1) - 19) / 19.0
            obj += -BIG * center_pen * x[i]

        model.Maximize(obj)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0

        results = []
        used_sets = set()

        for _ in range(K):
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            chosen = [i + 1 for i in range(37) if solver.Value(x[i]) == 1]
            if not _validity_checks(chosen):
                # NGセットなら最小カットで禁止して次へ
                model.Add(sum(x[i - 1] for i in chosen) <= 6)
                continue
            sc = _score_set(chosen, p, pmi, alpha_pair)
            tup = tuple(chosen)
            if tup in used_sets:
                model.Add(sum(x[i - 1] for i in chosen) <= 6)
                continue
            used_sets.add(tup)
            results.append((chosen, sc))
            # 同一解を禁止
            model.Add(sum(x[i - 1] for i in chosen) <= 6)

        return results if results else _beam_search_fallback(p, pmi, K, alpha_pair)

    # Fallback
    return _beam_search_fallback(p, pmi, K, alpha_pair)


def _beam_search_fallback(p, pmi, K=10, alpha_pair=0.3, width=64):
    # シンプルなビーム探索
    beams = [([], 0.0)]
    for n in range(1, 38):
        new_beams = []
        for cand, sc in beams:
            # スキップ
            new_beams.append((cand, sc))
            # 追加
            if len(cand) < 7:
                new = cand + [n]
                if len(new) <= 7:
                    s = _score_set(new, p, pmi, alpha_pair)
                    new_beams.append((new, s))
        # 上位で枝刈り
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:width]

    finals = [(cand, sc) for cand, sc in beams if len(cand) == 7 and _validity_checks(cand)]
    finals.sort(key=lambda x: x[1], reverse=True)
    return finals[:K]
