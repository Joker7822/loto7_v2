
import pandas as pd
from lottery_prediction import (
    LotoPredictor,
    preprocess_data,
    save_predictions_to_csv,
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
        existing = existing[existing["抽せん日"] != drawing_date]
        df = pd.concat([existing, df], ignore_index=True)
    except Exception:
        pass
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[INFO] ガチ予測を {filename} に保存しました。")

def main():
    results = pd.read_csv("loto7.csv", encoding="utf-8-sig")
    results["抽せん日"] = pd.to_datetime(results["抽せん日"], errors="coerce")
    latest_date = results["抽せん日"].max()

    X, _, _ = preprocess_data(results)
    input_size = X.shape[1]

    predictor = LotoPredictor(input_size=input_size, hidden_size=128, output_size=7)
    try:
        predictor.load_saved_models("models/tmp")
    except Exception as e:
        print("[WARN] 学習済みモデルのロードに失敗:", e)

    decoded_sets, confidences = predictor.predict_gachi(results, k_sets=10, alpha_pair=0.3)

    save_gachi_to_csv(list(zip(decoded_sets, confidences)), latest_date)

    try:
        evaluate_prediction_accuracy_with_bonus("loto7_predictions_gachi.csv", "loto7.csv")
    except Exception as e:
        print("[WARN] 評価に失敗しました:", e)

if __name__ == "__main__":
    main()
