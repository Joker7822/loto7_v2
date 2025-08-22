# portfolio.py
from typing import List, Tuple
import numpy as np

def jaccard(a,b):
    sa, sb = set(a), set(b)
    inter = len(sa&sb); uni = len(sa|sb)
    return inter/uni if uni else 0.0

def select_diverse(candidates: List[Tuple[List[int], float]], k=100, alpha=0.6):
    # alpha: スコア重視(1.0)〜多様性重視(0.0) の重み
    chosen=[]
    if not candidates: return chosen
    chosen.append(candidates[0][0])
    for _ in range(1, k):
        best=None; best_val=-1
        for c,score in candidates[:5000]:
            # 多様性 = 1 - 平均Jaccard
            if any(c==x for x in chosen): continue
            div = 1 - np.mean([jaccard(c,x) for x in chosen])
            val = alpha*score + (1-alpha)*div
            if val>best_val:
                best_val=val; best=c
        if best is None: break
        chosen.append(best)
    return chosen
