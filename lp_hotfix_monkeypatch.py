# lp_hotfix_monkeypatch.py  ← 丸ごと置き換え

import numpy as np
import pandas as pd
import random
import traceback

# --- まず lottery_prediction を安全に import
try:
    import lottery_prediction as lp
except Exception as e:
    lp = None
    print(f"[HOTFIX] lottery_prediction の import 失敗: {e}")

# --- extended_predictor 側（存在すれば使う）
ExtendedCls = None
try:
    from extended_predictor import LotoPredictorGachi as ExtendedCls
except Exception as e:
    print(f"[HOTFIX] extended_predictor の import 失敗（許容）: {e}")

# --- 対象クラスの収集（存在確認してから）
TargetClasses = []
if lp is not None and hasattr(lp, "LotoPredictor"):
    TargetClasses.append(lp.LotoPredictor)
if ExtendedCls is not None:
    TargetClasses.append(ExtendedCls)

# ===== フォールバック予測 =====
def _fallback_predict(self, latest_data, num_candidates=50):
    numbers_only, confidence_scores = [], []
    freq = {i: 1 for i in range(1, 38)}
    try:
        if isinstance(latest_data, pd.DataFrame) and '本数字' in latest_data.columns:
            for nums in latest_data['本数字']:
                if isinstance(nums, (list, tuple)):
                    for n in nums:
                        try:
                            v = int(n)
                            if 1 <= v <= 37:
                                freq[v] = freq.get(v, 1) + 1
                        except Exception:
                            continue
                else:
                    s = str(nums).strip('[]').replace(',', ' ')
                    for tok in s.split():
                        if tok.isdigit():
                            v = int(tok)
                            if 1 <= v <= 37:
                                freq[v] = freq.get(v, 1) + 1
    except Exception as e:
        print("[SAFE] 頻度計算失敗:", e)

    keys = list(range(1, 38))
    weights = np.array([freq[k] for k in keys], dtype=float)
    weights = weights / (weights.sum() + 1e-9)
    rng = np.random.default_rng(42)
    made = set()
    trials = max(num_candidates * 5, 200)
    for _ in range(trials):
        cand = rng.choice(keys, size=7, replace=False, p=weights)
        cand = tuple(sorted(int(x) for x in cand))
        if cand in made:
            continue
        made.add(cand)
        numbers_only.append(list(cand))
        conf = float(sum(freq[n] for n in cand)) / (7.0 * (weights.max() * len(keys)))
        confidence_scores.append(conf)
        if len(numbers_only) >= num_candidates:
            break
    if not numbers_only:
        for _ in range(num_candidates):
            cand = sorted(random.sample(range(1, 38), 7))
            numbers_only.append(cand)
            confidence_scores.append(0.5)
    return numbers_only, confidence_scores

# --- 可能ならオリジナル predict を拾う（どれか1つでも見つかればOK）
_ORIG_PREDICT = None
for cls in TargetClasses:
    try:
        maybe = getattr(cls, "predict", None)
        if callable(maybe):
            _ORIG_PREDICT = maybe
            break
    except Exception:
        pass

def _safe_predict(self, latest_data, num_candidates=50):
    try:
        # 可能なら preprocess_data で“最低限の妥当性”を確認
        try:
            import lottery_prediction as lp_mod
            pre = lp_mod.preprocess_data(latest_data)
            if not (isinstance(pre, tuple) and len(pre) >= 1 and pre[0] is not None and len(pre[0]) > 0):
                print("[SAFE] preprocess_data 不成立 → フォールバック")
                return _fallback_predict(self, latest_data, num_candidates=num_candidates)
        except Exception as e:
            print("[SAFE] preprocess_data 例外 → フォールバック:", e)
            return _fallback_predict(self, latest_data, num_candidates=num_candidates)

        # オリジナル predict を優先
        if _ORIG_PREDICT is not None:
            try:
                res = _ORIG_PREDICT(self, latest_data, num_candidates=num_candidates)
                if not (isinstance(res, (tuple, list)) and len(res) == 2):
                    print("[SAFE] 親predictの戻り値が不正 → フォールバック")
                    return _fallback_predict(self, latest_data, num_candidates=num_candidates)
                return res
            except Exception as e:
                print(f"[SAFE] 親predictで例外: {e}")
                traceback.print_exc()
                return _fallback_predict(self, latest_data, num_candidates=num_candidates)

        # 親が無い場合
        return _fallback_predict(self, latest_data, num_candidates=num_candidates)

    except Exception as e:
        print(f"[SAFE] _safe_predict 未捕捉例外: {e}")
        traceback.print_exc()
        return _fallback_predict(self, latest_data, num_candidates=num_candidates)

# --- 見つかったクラスに安全 predict をアタッチ
if not TargetClasses:
    print("[HOTFIX] 対象クラスが見つからなかったため、パッチをスキップしました。")
else:
    for cls in TargetClasses:
        setattr(cls, "predict", _safe_predict)
    print(f"[HOTFIX] predict を安全ラップしました（対象クラス: {len(TargetClasses)}）")
