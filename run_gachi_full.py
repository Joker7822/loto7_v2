
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gachi_full.py
-----------------
ガチ予測（predict_gachi）で予測CSVを生成し、そのまま評価まで一発実行。

使い方（例）:
    python run_gachi_full.py \
        --truth loto7.csv \
        --outpred loto7_predictions_gachi.csv \
        --summary loto7_evaluation_summary_gachi.txt \
        --k 20 \
        --alpha 0.35 \
        --history 120

前提:
- 同ディレクトリに lottery_prediction.py / gachi_decoder.py が存在
- lottery_prediction.py 内に以下が定義されていること:
    - preprocess_data(dataframe) -> (X, y, feature_names)
    - class LotoPredictor(input_size, hidden_size, output_size)
    - def save_predictions_to_csv(predictions, drawing_date, filename="...")
    - def evaluate_prediction_accuracy_with_bonus(predictions_file, results_file)
"""

import argparse
import ast
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# 同ディレクトリ優先
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from lottery_prediction import (
    preprocess_data,
    LotoPredictor,
    save_predictions_to_csv,
    evaluate_prediction_accuracy_with_bonus as eval_with_bonus,
)

# -------------------------
# ユーティリティ
# -------------------------
def _parse_numbers_row(row) -> list:
    """1行から本数字7個のリストを推定して返す（フォーマットに頑強に対応）"""
    # 1) '本数字' 列が list/文字列としてあるケース
    for key in ['本数字', 'main_numbers', 'numbers', 'nums']:
        if key in row and pd.notna(row[key]):
            val = row[key]
            if isinstance(val, (list, tuple)):
                nums = [int(x) for x in val]
                if len(nums) >= 7:
                    return nums[:7]
            if isinstance(val, str):
                try:
                    obj = ast.literal_eval(val)
                    if isinstance(obj, (list, tuple)):
                        nums = [int(x) for x in obj]
                        if len(nums) >= 7:
                            return nums[:7]
                except Exception:
                    # カンマ区切り "1,2,3,4,5,6,7"
                    parts = [p.strip() for p in val.replace('[','').replace(']','').split(',') if p.strip()]
                    if len(parts) >= 7 and all(p.isdigit() for p in parts[:7]):
                        return [int(p) for p in parts[:7]]
    # 2) 数字の列が分かれているケース
    numeric_cols = [c for c in row.index if str(c).strip() not in ('ボーナス1','ボーナス2','bonus1','bonus2')]
    # 数値で埋まっている最初の7列を拾う
    nums = []
    for c in numeric_cols:
        val = row[c]
        try:
            iv = int(val)
            if 1 <= iv <= 37:
                nums.append(iv)
                if len(nums) == 7:
                    return nums
        except Exception:
            continue
    return None

def load_truth_as_latest_dataframe(csv_path: str, history: int = 120) -> pd.DataFrame:
    """実績CSVから直近 history 件を抽出し、'本数字' 列のみを含むDataFrameを返す"""
    df = pd.read_csv(csv_path)
    # 日付列の推定（あればソート）
    date_col = None
    for cand in ['抽せん日', 'date', '抽選日', 'drawing_date']:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        except Exception:
            pass
    # 本数字の抽出
    rows = []
    for _, r in df.tail(history).iterrows():
        nums = _parse_numbers_row(r)
        if nums is not None and len(nums) >= 7:
            rows.append({'本数字': nums[:7]})
    if not rows:
        raise ValueError("実績CSVから '本数字' を抽出できませんでした。列名やフォーマットを確認してください。")
    return pd.DataFrame(rows)

def guess_drawing_date_from_truth(csv_path: str) -> str:
    """出力CSVの抽せん日を推定（truthの最終日+7日 or そのまま最終日）"""
    try:
        df = pd.read_csv(csv_path)
        for cand in ['抽せん日', 'date', '抽選日', 'drawing_date']:
            if cand in df.columns:
                d = pd.to_datetime(df[cand]).max()
                if pd.isna(d):
                    break
                # 次回分として+7日を仮置き（週1抽選想定）。うまくなければ最終日にする。
                nxt = d + pd.Timedelta(days=7)
                return nxt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")

# -------------------------
# メイン処理
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="loto7.csv", help="実績CSVのパス（真値）")
    ap.add_argument("--outpred", default="loto7_predictions_gachi.csv", help="出力する予測CSVのパス")
    ap.add_argument("--summary", default="loto7_evaluation_summary_gachi.txt", help="出力する評価サマリtxt")
    ap.add_argument("--k", type=int, default=20, help="ガチ予測で生成するセット数")
    ap.add_argument("--alpha", type=float, default=0.35, help="PMI重み（alpha_pair）")
    ap.add_argument("--history", type=int, default=120, help="最新何件の履歴を特徴量に使うか")
    ap.add_argument("--date", default=None, help="予測CSVに書く抽せん日（YYYY-MM-DD）。未指定なら推定")
    ap.add_argument("--hidden", type=int, default=128, help="LSTMのhidden size")
    args = ap.parse_args()

    # 1) 履歴を読み込んで latest_data を構築
    print(f"[INFO] 履歴読み込み: {args.truth} (history={args.history})")
    latest_data = load_truth_as_latest_dataframe(args.truth, history=args.history)

    # 2) 特徴量次元を推定して LotoPredictor を初期化
    X, _, _ = preprocess_data(latest_data)
    if X is None or len(X) == 0:
        raise RuntimeError("特徴量生成に失敗しました。truth CSVのフォーマットを確認してください。")
    input_size = X.shape[1]
    print(f"[INFO] 推定 input_size = {input_size}")
    predictor = LotoPredictor(input_size, args.hidden, 7)

    # 3) ガチ予測を生成
    print(f"[INFO] ガチ予測を生成: k={args.k}, alpha={args.alpha}")
    preds, conf = predictor.predict_gachi(latest_data, k_sets=args.k, alpha_pair=args.alpha)
    if preds is None or not preds:
        raise RuntimeError("ガチ予測の生成に失敗しました。")

    # 4) 予測CSVへ保存
    drawing_date = args.date or guess_drawing_date_from_truth(args.truth)
    print(f"[INFO] 予測CSVに保存します: {args.outpred} (抽せん日={drawing_date})")
    rows = list(zip(preds, conf))
    save_predictions_to_csv(rows, drawing_date, filename=args.outpred)

    # 5) そのまま評価
    print(f"[INFO] 評価を実行: pred={args.outpred}, truth={args.truth}")
    eval_with_bonus(args.outpred, args.truth)

    # 6) まとめをtxtに保存（評価関数が標準出力のみでも最低限残す）
    # ここでは簡易に出力ヘッダのみ。詳細を残すなら evaluate 関数側をdict返しに拡張してください。
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("=== ガチ予測 一発実行サマリ ===\n")
        f.write(f"predictions: {args.outpred}\n")
        f.write(f"truth: {args.truth}\n")
        f.write(f"k_sets: {args.k}\n")
        f.write(f"alpha_pair: {args.alpha}\n")
        f.write(f"history_used: {args.history}\n")
        f.write(f"drawing_date: {drawing_date}\n")
    print(f"[INFO] サマリを保存しました: {args.summary}")

if __name__ == "__main__":
    main()
