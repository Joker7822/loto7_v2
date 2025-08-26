
import numpy as np
import pandas as pd
import random
import torch

from lottery_prediction import LotoPredictor, preprocess_data
from gachi_decoder import build_marginals_from_sets, compute_pair_pmi, decode_topK_sets

class LotoPredictorGachi(LotoPredictor):
    def predict(self, latest_data, num_candidates=50):
        try:
            return super().predict(latest_data, num_candidates=num_candidates)
        except Exception as e:
            print(f"[SAFE] ベースpredictで例外: {e}")
            return self._fallback_predict(latest_data, num_candidates=num_candidates)

    def _fallback_predict(self, latest_data, num_candidates=50):
        numbers_only, confidence_scores = [], []

        # frequency table with Laplace smoothing
        freq = {i: 1 for i in range(1, 38)}
        try:
            if isinstance(latest_data, pd.DataFrame) and '本数字' in latest_data.columns:
                for nums in latest_data['本数字']:
                    if isinstance(nums, (list, tuple)):
                        for n in nums:
                            if 1 <= int(n) <= 37:
                                freq[int(n)] = freq.get(int(n), 1) + 1
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

    def predict_gachi(self, latest_data, k_sets=10, alpha_pair=0.3):
        base = self.predict(latest_data, num_candidates=max(200, k_sets * 20))
        if base is None or base[0] is None:
            return None, None
        candidates, conf = base

        extra_logits = []
        try:
            from gnn_core import build_cooccurrence_graph
            graph_data = build_cooccurrence_graph(latest_data)
            if hasattr(self, "gnn_model") and self.gnn_model is not None:
                self.gnn_model.eval()
                with torch.no_grad():
                    gnn_scores = self.gnn_model(graph_data.x, graph_data.edge_index).squeeze().cpu().numpy()
                extra_logits.append(gnn_scores)
        except Exception as _e:
            print("[INFO] GNNログitの取得をスキップ:", _e)

        p = build_marginals_from_sets(candidates, conf, extra_logits=extra_logits)
        try:
            hist = latest_data.copy()
            pair_pmi = compute_pair_pmi(hist)
        except Exception as _e:
            print("[INFO] PMI計算をスキップ:", _e)
            pair_pmi = None

        top = decode_topK_sets(p, pair_pmi, K=k_sets, alpha_pair=alpha_pair)
        if not top:
            return candidates[:k_sets], conf[:k_sets]

        decoded, scores = zip(*top)
        s = np.array(scores, dtype=float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return list(decoded), list(s)
