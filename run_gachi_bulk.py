
# run_gachi_bulk_rescue.py
# ------------------------------------------------------------------
# 自前の Predictor クラスが見つからなくても動く“レスキュー版”。
# - 直近データから周辺確率（頻度）＋ペア共起スコアを推定
# - ランダム探索＋多様化（Jaccardペナルティ）で Top-K を生成
# - 生成結果を loto7_predictions_gachi.csv に保存
# 既存の Predictor 実装が復旧したら、元の run_gachi_bulk.py に戻せます。
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

def parse_number_string(s):
    if s is None:
        return []
    s = str(s).strip("[]").replace(",", " ").replace("'", "").replace('"', "")
    toks = [t for t in s.split() if t.isdigit()]
    return [int(t) for t in toks]

def extract_numbers_cols(df):
    # 列名に依存しないように、本数字/ボーナスを推定
    main_col = None
    bonus_col = None
    for c in df.columns:
        if "本数字" in c:
            main_col = c
        if "ボーナス" in c:
            bonus_col = c
    if main_col is None:
        # 代表的カラム名のフォールバック
        candidates = [c for c in df.columns if "予測番号" in c or "numbers" in c.lower()]
        if candidates:
            main_col = candidates[0]
        else:
            raise RuntimeError("本数字カラムが見つかりません（例：'本数字'）")
    if bonus_col is None:
        bonus_col = None  # 無くてもOK
    return main_col, bonus_col

def build_stats(hist_df, main_col):
    # ヒストリーから 1..37 の頻度とペア共起頻度を作成
    freq = Counter()
    pair = Counter()
    for nums_raw in hist_df[main_col].tolist():
        nums = parse_number_string(nums_raw)
        nums = [n for n in nums if 1 <= n <= 37]
        for n in nums:
            freq[n] += 1
        for a, b in combinations(sorted(set(nums)), 2):
            pair[(a, b)] += 1
    return freq, pair

def sample_candidates(freq, pair, rng, num_samples=2000, k=7):
    keys = list(range(1, 38))
    fsum = sum(freq.get(i, 1) for i in keys)
    weights = np.array([freq.get(i, 1)/fsum for i in keys], dtype=float)
    cand_sets = []
    scores = []
    for _ in range(num_samples):
        # 頻度重み付きサンプリング（重複なし）
        sel = tuple(sorted(rng.choice(keys, size=k, replace=False, p=weights)))
        # スコア = 個別頻度の和 + ペア共起の和
        s = sum(freq.get(n, 0) for n in sel)
        for a, b in combinations(sel, 2):
            if a > b: a, b = b, a
            s += pair.get((a, b), 0)
        cand_sets.append(sel)
        scores.append(float(s))
    return cand_sets, scores

def diverse_topk(cands, scores, topk=10, lambda_div=0.6):
    # Greedy 多様化（Jaccardペナルティ）
    order = np.argsort(-np.array(scores))
    cands = [cands[i] for i in order]
    bases = [scores[i] for i in order]
    selected = []
    for i, cand in enumerate(cands):
        penalty = 0.0
        A = set(cand)
        for s in selected:
            inter = len(A & set(s))
            uni = len(A | set(s))
            penalty += (inter / max(1, uni))
        value = bases[i] - lambda_div * penalty
        # 挿入位置最適化は省略、単純に選択
        selected.append(cand)
        if len(selected) >= topk:
            break
    # 疑似信頼度：min-max 正規化
    sel_scores = []
    for cand in selected:
        s = sum(scores[j] for j, cc in enumerate(cands) if cc == cand)
        sel_scores.append(s)
    arr = np.array(sel_scores, dtype=float)
    if arr.max() > arr.min():
        conf = ((arr - arr.min()) / (arr.max() - arr.min())).tolist()
    else:
        conf = [0.5]*len(selected)
    return selected, conf

def save_gachi_to_csv(pairs, drawing_date, filename="loto7_predictions_gachi.csv"):
    drawing_date = pd.to_datetime(drawing_date).strftime("%Y-%m-%d")
    row = {"抽せん日": drawing_date}
    for i, (numbers, confidence) in enumerate(pairs[:10], 1):
        row[f"ガチ予測{i}"] = ", ".join(map(str, numbers))
        row[f"ガチ信頼度{i}"] = round(float(confidence), 3)
    df_row = pd.DataFrame([row])
    try:
        existing = pd.read_csv(filename, encoding="utf-8-sig")
        if "抽せん日" in existing.columns:
            existing = existing[existing["抽せん日"] != drawing_date]
        df_out = pd.concat([existing, df_row], ignore_index=True)
    except Exception:
        df_out = df_row
    df_out.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[INFO] 保存: {filename} ({drawing_date})")

def main():
    df = pd.read_csv("loto7.csv", encoding="utf-8-sig")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["抽せん日"]).sort_values("抽せん日").reset_index(drop=True)
    print("[INFO] 読み込み:", len(df), "件")

    main_col, _ = extract_numbers_cols(df)
    rng = np.random.default_rng(42)

    for i in range(10, len(df)):
        hist = df.iloc[:i].copy()
        test_date = df.iloc[i]["抽せん日"]
        try:
            # 統計構築
            freq, pair = build_stats(hist, main_col)
            # 候補サンプリング
            cands, scores = sample_candidates(freq, pair, rng, num_samples=2000, k=7)
            # 多様化でTop-K
            decoded, conf = diverse_topk(cands, scores, topk=10, lambda_div=0.6)
            pairs = list(zip(decoded, conf))
            save_gachi_to_csv(pairs, test_date, filename="loto7_predictions_gachi.csv")
            print(f"[OK] {test_date.date()}")
        except Exception as e:
            print(f"[WARN] {test_date.date()} 失敗:", e)

    print("[DONE] run_gachi_bulk_rescue 完了")

if __name__ == "__main__":
    main()
