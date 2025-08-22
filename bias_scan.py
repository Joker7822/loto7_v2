# bias_scan.py
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import chisquare

NUM_MAX = 37

def load_draws(csv_path: str) -> pd.DataFrame:
    # 期待: 列に本数字7個 (例: N1..N7) がある前提。なければ適宜合わせてください。
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    cols = [c for c in df.columns if str(c).strip().isdigit() or c.upper().startswith("N")]
    if len(cols) < 7:
        raise ValueError("抽選数字の列が見つかりません。列名をN1..N7等に合わせてください。")
    return df[cols].astype(int)

def global_frequency_tests(draws_df: pd.DataFrame):
    flat = draws_df.values.ravel()
    cnt = Counter(flat)
    obs = np.array([cnt.get(i, 0) for i in range(1, NUM_MAX+1)], dtype=float)
    exp = np.full(NUM_MAX, flat.size / NUM_MAX)
    chi_stat, p = chisquare(obs, exp)
    z = (obs - exp) / np.sqrt(exp + 1e-9)
    return {"chi2": chi_stat, "pvalue": float(p), "freq": obs, "zscore": z}

def window_chi2(draws_df: pd.DataFrame, window=100):
    res=[]
    for end in range(window, len(draws_df)+1):
        win = draws_df.iloc[end-window:end].values.ravel()
        cnt = Counter(win)
        obs = np.array([cnt.get(i, 0) for i in range(1, NUM_MAX+1)], dtype=float)
        exp = np.full(NUM_MAX, win.size / NUM_MAX)
        chi_stat, p = chisquare(obs, exp)
        res.append((end, chi_stat, float(p)))
    return pd.DataFrame(res, columns=["end_idx","chi2","pvalue"])

def pairwise_dependence(draws_df: pd.DataFrame):
    # 共起と独立期待の残差（標準化残差）を返す
    m = np.zeros((NUM_MAX, NUM_MAX), dtype=float)
    flat = draws_df.values
    for row in flat:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                a, b = row[i]-1, row[j]-1
                m[a,b]+=1; m[b,a]+=1
    # 期待値：各番号の出現確率から独立仮定で推定
    freq = np.bincount(flat.ravel(), minlength=NUM_MAX+1)[1:]
    p = freq / freq.sum()
    exp = np.outer(p, p) * m.sum()
    z = (m - exp) / np.sqrt(exp + 1e-9)
    np.fill_diagonal(z, 0.0)
    return z, m, exp
