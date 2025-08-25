# extended_predictor.py
import numpy as np
import torch

from lottery_prediction import LotoPredictor
from gachi_decoder import (
    build_marginals_from_sets,
    compute_pair_pmi,
    decode_topK_sets,
)

class LotoPredictorGachi(LotoPredictor):
    """
    LotoPredictor を拡張し、predict_gachi を追加したクラス
    """

    def predict_gachi(self, latest_data, k_sets=10, alpha_pair=0.3):
        # まず通常predictで候補を取得
        base = self.predict(latest_data, num_candidates=200)
        if base is None or base[0] is None:
            return None, None
        candidates, conf = base

        # 周辺確率
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

        # gachi_decoderで周辺確率とPMIを計算
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
