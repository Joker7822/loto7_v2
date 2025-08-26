
import numpy as np
import pandas as pd
import random
import traceback
import lottery_prediction as lp

_ORIG_PREDICT = lp.LotoPredictor.predict

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
    import numpy as _np
    weights = _np.array([freq[k] for k in keys], dtype=float)
    weights = weights / (weights.sum() + 1e-9)
    rng = _np.random.default_rng(42)
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

def _safe_predict(self, latest_data, num_candidates=50):
    numbers_only, confidence_scores = [], []
    try:
        try:
            X, _, _ = lp.preprocess_data(latest_data)
            X_df = pd.DataFrame(X)
            if not getattr(self, "feature_names", None):
                print("Warning: self.feature_names が未定義です → フォールバック作成")
                self.feature_names = [str(c) for c in X_df.columns]
            try:
                X_df.columns = self.feature_names
            except Exception as e:
                print("[WARN] feature_names の適用に失敗:", e)
                if len(self.feature_names) != X_df.shape[1]:
                    self.feature_names = [str(c) for c in X_df.columns]
        except Exception as e:
            print("[SAFE] preprocess_data 失敗（フォールバックへ）:", e)

        ready_automl = hasattr(self, "regression_models") and self.regression_models and all(m is not None for m in self.regression_models)
        ready_lstm = hasattr(self, "lstm_model") and self.lstm_model is not None
        ready_feats = getattr(self, "feature_names", None) is not None

        if ready_automl and ready_lstm and ready_feats:
            return _ORIG_PREDICT(self, latest_data, num_candidates=num_candidates)

        return _fallback_predict(self, latest_data, num_candidates=num_candidates)

    except Exception as e:
        print(f"[SAFE] 親predictで例外: {e}")
        traceback.print_exc()
        return _fallback_predict(self, latest_data, num_candidates=num_candidates)

lp.LotoPredictor.predict = _safe_predict
