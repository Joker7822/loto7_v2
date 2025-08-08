import os
from lottery_prediction import (
    git_commit_and_push,
    bulk_predict_all_past_draws,
    evaluate_prediction_accuracy_with_bonus,
    LotoPredictor,
    preprocess_data,
    pd,
    set_global_seed
)

# === ステップ 1: 初期化 ===
print("[STEP 1] 古い予測ファイルを削除")
pred_file = "loto7_predictions.csv"
if os.path.exists(pred_file):
    os.remove(pred_file)
    print(f"[INFO] {pred_file} を削除しました")

# === ステップ 2: 過去全データで未来リーク防止予測 ===
print("[STEP 2] すべての抽せん回について予測を実施中...")
bulk_predict_all_past_draws()

# === ステップ 3: 精度評価（等級・一致数など）===
print("[STEP 3] 予測結果の精度を評価中...")
accuracy_df = evaluate_prediction_accuracy_with_bonus()

# === ステップ 4: モデル再学習 ===
print("[STEP 4] 予測精度を使ってモデルを再学習中...")

try:
    set_global_seed(42)
    df = pd.read_csv("loto7.csv")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
    X, _, _ = preprocess_data(df)
    input_size = X.shape[1] if X is not None and X.size > 0 else 10

    predictor = LotoPredictor(input_size=input_size, hidden_size=128, output_size=7)
    predictor.train_model(df, accuracy_results=accuracy_df)

    print("[✅ 完了] モデルの強化学習が完了しました")
    git_commit_and_push("models", "Updated models after evolution")

except Exception as e:
    print(f"[ERROR] モデル再学習中にエラー: {e}")


