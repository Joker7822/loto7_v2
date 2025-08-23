
"""
evaluate_gachi_predictions.py
-----------------------------
既存の lottery_prediction.py に追加した predict_gachi()／評価関数を前提に、
ガチ予測の結果CSVを評価してサマリを出力します。

使い方:
    python evaluate_gachi_predictions.py \
        --pred loto7_predictions_gachi.csv \
        --truth loto7.csv \
        --out summary_gachi.txt
"""
import argparse
import os
import sys

# 同ディレクトリ優先のため
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lottery_prediction import evaluate_prediction_accuracy_with_bonus as eval_with_bonus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", "--predictions", dest="predictions_file",
                    default="loto7_predictions_gachi.csv", help="ガチ予測のCSVパス")
    ap.add_argument("--truth", "--results", dest="results_file",
                    default="loto7.csv", help="実績CSV（真値）のパス")
    ap.add_argument("--out", "--summary", dest="summary_file",
                    default="loto7_evaluation_summary_gachi.txt", help="出力サマリtxt")

    args = ap.parse_args()

    preds = args.predictions_file
    truth = args.results_file
    out = args.summary_file

    print(f"[INFO] 評価開始: pred={preds} truth={truth}")
    stats = eval_with_bonus(preds, truth)  # 既存関数が dict を返す想定（返り値はなければ None）

    # 既存関数が標準出力するだけの実装でも、ここでファイルに追記しておく
    lines = []
    lines.append("=== 予測精度の統計情報（ガチ予測） ===")
    if isinstance(stats, dict):
        # 柔軟に展開
        for k, v in stats.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(詳細は関数の標準出力を参照)")

    content = "\\n".join(lines) + "\\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[INFO] サマリを保存しました: {out}")

if __name__ == "__main__":
    main()
