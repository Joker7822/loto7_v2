#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime

from lottery_prediction import (
    evaluate_prediction_accuracy_with_bonus_compat,
    git_commit_and_push,
    bulk_predict_all_past_draws,
    evaluate_prediction_accuracy_with_bonus,
    LotoPredictor,
    preprocess_data,
    pd,
    set_global_seed,
    write_evolution_log,
    generate_evolution_graph_from_csv,
)

def main():
    # === ステップ 0: 乱数固定 ===
    set_global_seed(42)

    # === ステップ 1: 古い予測を削除 ===
    print("[STEP 1] 古い予測ファイルを削除")
    pred_file = "loto7_predictions.csv"
    if os.path.exists(pred_file):
        os.remove(pred_file)
        print(f"  - removed {pred_file}")

    # === ステップ 2: 過去全抽せんに対して一括予測 ===
    print("[STEP 2] 一括予測を実行")
    bulk_predict_all_past_draws()

    # === ステップ 3: 予測の的中評価 ===
    print("[STEP 3] 予測の評価を実行（本数字＋ボーナス）")
    accuracy_df = evaluate_prediction_accuracy_with_bonus_compat(
        prediction_file="loto7_predictions.csv",
        output_csv="loto7_prediction_evaluation_with_bonus.csv",
        output_txt="loto7_evaluation_summary.txt"
    )

    # === ステップ 4: 再学習のためのデータ準備とモデル構築 ===
    print("[STEP 4] モデル再学習")
    # 学習データの読み込み
    src_csv = "loto7.csv"
    if os.path.exists(src_csv):
        df = pd.read_csv(src_csv, encoding="utf-8-sig")
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        res = preprocess_data(df)
        X = res[0] if isinstance(res, tuple) and len(res)==3 else None
        input_size = X.shape[1] if X is not None and X.size > 0 else 10
        data_max_date = df["抽せん日"].max()
    else:
        print(f"[WARN] {src_csv} が見つかりません。input_size=10 で継続します。")
        input_size = 10
        data_max_date = None
        df = None

    predictor = LotoPredictor(input_size=input_size, hidden_size=128, output_size=7)
    if df is not None:
        predictor.train_model(df, accuracy_results=accuracy_df)
    else:
        print("[WARN] 学習データが無いので train_model をスキップ")

    # === ステップ 5: サマリログ出力 ===
    print("[STEP 5] 再学習サマリを出力")
    top_feats = []
    try:
        # 任意：特徴量の重要度抽出に対応していれば拾う
        from lottery_prediction import create_advanced_features, extract_strong_features
        feat_df = create_advanced_features(df) if df is not None else None
        if feat_df is not None and accuracy_df is not None:
            top_feats = extract_strong_features(accuracy_df, feat_df)
    except Exception as e:
        print("[INFO] top_features の抽出をスキップ:", e)

    write_evolution_log(
        evaluation_df=accuracy_df,
        data_max_date=data_max_date,
        seed=42,
        model_dir="models",
        top_features=top_feats,
        csv_path="logs/evolution.csv",
        json_dir="logs/evolution_detail"
    )

    # === ステップ 6: 進化グラフ生成 ===
    print("[STEP 6] 進化グラフを更新")
    generate_evolution_graph_from_csv(
        csv_path="logs/evolution.csv",
        metric="f1",
        output_file="logs/evolution_graph.png"
    )

    # === ステップ 7: 変更をGitに反映（任意） ===
    try:
        git_commit_and_push("models", "Updated models after evolution")
        git_commit_and_push("logs", "Update evolution logs and graph")
        print("[STEP 7] Git 反映完了")
    except Exception as e:
        print("[WARN] Git 反映をスキップ:", e)

    print("[✅ 完了] モデルの強化学習とログ出力が完了しました")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] モデル再学習中にエラー: {e}")
