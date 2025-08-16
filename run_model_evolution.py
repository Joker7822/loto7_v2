#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime

from lottery_prediction import (
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
    accuracy_df = evaluate_prediction_accuracy_with_bonus(
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
        X, _, _ = preprocess_data(df)
        input_size = X.shape[1] if X is not None and X.size > 0 else 10
        data_max_date = df["抽せん日"].max()
    else:
        print(f"[WARN] {src_csv} が見つかりません。input_size=10 で継続します。")
        input_size = 10
        data_max_date = None
        df = None

    predictor = LotoPredictor(input_size=input_size, hidden_size=128, output_size=7)
    if df is not None:
        predictor.train_model(df, accuracy_results=accuracy_df, model_dir="models/latest")
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

    print("[STEP 8] セカンダリレポジトリへモデルを同期（環境変数で制御）")
    sync_to_secondary_repo(models_dir="models/latest")
    print("[✅ 完了] モデルの強化学習とログ出力が完了しました")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] モデル再学習中にエラー: {e}")



import subprocess
import shutil

def sync_to_secondary_repo(models_dir="models/latest"):
    """
    models/latest を別リポジトリへ上書き保存（環境変数で制御）
    必要な環境変数:
      - SECONDARY_REPO      例: "Joker7822/loto7"
      - SECONDARY_BRANCH    例: "main"（省略可: デフォルト main）
      - MODEL_REPO_TOKEN    PAT (repo 書き込み権限)
    """
    secondary_repo = os.environ.get("SECONDARY_REPO")
    token = os.environ.get("MODEL_REPO_TOKEN")
    branch = os.environ.get("SECONDARY_BRANCH", "main")

    if not secondary_repo or not token:
        print("[SYNC] SECONDARY_REPO or MODEL_REPO_TOKEN が未設定のためスキップします。")
        return

    if not os.path.isdir(models_dir):
        print(f"[SYNC] 同期元 {models_dir} が存在しないためスキップします。")
        return

    repo_dir = "model_repo_sync_tmp"
    try:
        if os.path.isdir(repo_dir):
            shutil.rmtree(repo_dir)

        print(f"[SYNC] Clone {secondary_repo}@{branch}")
        subprocess.run([
            "git", "clone", "--depth", "1",
            f"https://x-access-token:{token}@github.com/{secondary_repo}.git",
            repo_dir
        ], check=True)

        # checkout branch (create if missing)
        subprocess.run(["git", "-C", repo_dir, "checkout", branch], check=False)

        target_dir = os.path.join(repo_dir, "models", "latest")
        os.makedirs(target_dir, exist_ok=True)

        print("[SYNC] コピー: models/latest -> secondary repo")
        # 上書きコピー（既存削除 → コピー）
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(models_dir, target_dir)

        # 変更検知 → コミット＆プッシュ
        subprocess.run(["git", "-C", repo_dir, "add", "-A"], check=True)
        # 差分がない場合はコミットしない
        diff = subprocess.run(["git", "-C", repo_dir, "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            print("[SYNC] 変更なし。push をスキップします。")
            return

        subprocess.run(["git", "-C", repo_dir, "config", "user.name", "github-actions"], check=True)
        subprocess.run(["git", "-C", repo_dir, "config", "user.email", "github-actions@github.com"], check=True)
        msg = f"Sync latest model from {os.environ.get('GITHUB_REPOSITORY','local')}"
        subprocess.run(["git", "-C", repo_dir, "commit", "-m", msg], check=True)
        subprocess.run(["git", "-C", repo_dir, "push", "origin", branch], check=True)
        print("[SYNC] Secondary repo へ同期完了。")
    except Exception as e:
        print(f"[SYNC][WARN] 同期中にエラー: {e}")
    finally:
        try:
            if os.path.isdir(repo_dir):
                shutil.rmtree(repo_dir)
        except Exception:
            pass
