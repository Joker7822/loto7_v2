
import pandas as pd
from extended_predictor import LotoPredictorGachi as LotoPredictor
from lottery_prediction import (
    preprocess_data,
    evaluate_prediction_accuracy_with_bonus,
)

def save_gachi_to_csv(predictions, drawing_date, filename="loto7_predictions_gachi.csv"):
    drawing_date = pd.to_datetime(drawing_date).strftime("%Y-%m-%d")
    row = {"抽せん日": drawing_date}
    for i, (numbers, confidence) in enumerate(predictions[:10], 1):
        row[f"ガチ予測{i}"] = ', '.join(map(str, numbers))
        row[f"ガチ信頼度{i}"] = round(float(confidence), 3)
    df = pd.DataFrame([row])
    try:
        existing = pd.read_csv(filename, encoding="utf-8-sig")
        if "抽せん日" in existing.columns:
            existing = existing[existing["抽せん日"] != drawing_date]
        df = pd.concat([existing, df], ignore_index=True)
    except Exception:
        pass
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[INFO] ガチ予測を {filename} に保存しました。")

def main():
    # 1) データ読み込み（loto7.csv 必須）
    results = pd.read_csv("loto7.csv", encoding="utf-8-sig")
    results["抽せん日"] = pd.to_datetime(results["抽せん日"], errors="coerce")
    latest_date = results["抽せん日"].max()

    # 2) 前処理で input_size を決定
    X, _, _ = preprocess_data(results)
    input_size = X.shape[1]

    # 3) 予測器を作成（必要なら学習済みモデルをロード）
    predictor = LotoPredictor(input_size=input_size, hidden_size=128, output_size=7)
    try:
        predictor.load_saved_models("models/tmp")
    except Exception as e:
        print("[WARN] 学習済みモデルのロードに失敗:", e)

    # 4) ガチ予測
    decoded_sets, confidences = predictor.predict_gachi(results, k_sets=10, alpha_pair=0.3)

    # 5) 画面表示（Top10）
    print("\n=== ガチ予測 Top 10 ===")
    for idx, (nums, conf) in enumerate(zip(decoded_sets[:10], confidences[:10]), 1):
        print(f"{idx:02d}: {sorted(nums)}  信頼度={conf:.3f}")

    # 6) 保存
    save_gachi_to_csv(list(zip(decoded_sets, confidences)), latest_date)

    # 7) 任意: 評価（ボーナス含む）
    try:
        evaluate_prediction_accuracy_with_bonus("loto7_predictions_gachi.csv", "loto7.csv")
    except Exception as e:
        print("[WARN] 評価に失敗しました:", e)

if __name__ == "__main__":
    main()
