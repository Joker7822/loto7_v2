
# run_gachi_bulk.py (fixed): use LotteryPredictor (fallback to LotoPredictor) and apply gachi_best_params.json
import os
import json
import pandas as pd

# --- Predictor import (robust) ---
Predictor = None
import_error_msgs = []

try:
    # Preferred current name
    from lottery_prediction import LotteryPredictor as Predictor
except Exception as e:
    import_error_msgs.append(f"[INFO] LotteryPredictor not found: {e}")
    try:
        # Older name
        from lottery_prediction import LotoPredictor as Predictor
    except Exception as e2:
        import_error_msgs.append(f"[INFO] LotoPredictor not found: {e2}")
        try:
            # Extended version if present in the repo
            from extended_predictor import LotoPredictorGachi as Predictor
        except Exception as e3:
            import_error_msgs.append(f"[INFO] extended_predictor.LotoPredictorGachi not found: {e3}")

if Predictor is None:
    raise ImportError("No suitable Predictor class found. Tried: LotteryPredictor, LotoPredictor, extended_predictor.LotoPredictorGachi\n"
                      + "\n".join(import_error_msgs))

# --- Helpers from lottery_prediction (optional) ---
set_global_seed = None
preprocess_data = None
save_self_predictions = None
save_predictions_to_csv = None
evaluate_prediction_accuracy_with_bonus = None
git_commit_and_push = None
_save_all_models_no_self = None
save_gachi_to_csv = None
make_compat_for_evaluator = None

try:
    from lottery_prediction import (
        set_global_seed, preprocess_data, save_self_predictions,
        save_predictions_to_csv, evaluate_prediction_accuracy_with_bonus,
        git_commit_and_push, _save_all_models_no_self
    )
except Exception as e:
    print("[WARN] Some helpers not available from lottery_prediction:", e)

# If a helper is absent, provide light fallbacks
def _fallback_save_gachi_to_csv(predictions, drawing_date, filename="loto7_predictions_gachi.csv"):
    drawing_date = pd.to_datetime(drawing_date).strftime("%Y-%m-%d")
    row = {"抽せん日": drawing_date}
    for i, (numbers, confidence) in enumerate(predictions[:10], 1):
        row[f"ガチ予測{i}"] = ", ".join(map(str, numbers))
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

try:
    from run_gachi_bulk import save_gachi_to_csv as _sg
    save_gachi_to_csv = _sg
except Exception:
    try:
        from lottery_prediction import save_gachi_to_csv as _sg2
        save_gachi_to_csv = _sg2
    except Exception:
        save_gachi_to_csv = _fallback_save_gachi_to_csv

def load_best_params(path="gachi_best_params.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            bp = json.load(f)
        k_sets = int(bp.get("k_sets", 10))
        alpha_pair = float(bp.get("alpha_pair", 0.3))
        return k_sets, alpha_pair
    except Exception as e:
        print("[INFO] gachi_best_params.json が見つからないか読み込み失敗:", e)
        return 10, 0.3

def main():
    set_seed = set_global_seed or (lambda s: None)
    set_seed(42)

    df = pd.read_csv("loto7.csv", encoding="utf-8-sig")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
    df = df.sort_values("抽せん日").reset_index(drop=True)
    print("[INFO] 抽せんデータ読み込み完了:", len(df), "件")

    # Build a predictor
    predictor = None
    try:
        # Try to infer input size if preprocess is available
        if preprocess_data is not None:
            X, _, _ = preprocess_data(df.iloc[:-1].copy())
            input_size = X.shape[1] if X is not None and hasattr(X, "shape") else 128
            try:
                predictor = Predictor(input_size, 128, 7)
            except Exception:
                predictor = Predictor()
        else:
            predictor = Predictor()
    except Exception as e:
        print("[WARN] Predictor init failed, trying no-arg:", e)
        predictor = Predictor()

    # Try training (if the class provides it)
    try:
        if hasattr(predictor, "train_model"):
            predictor.train_model(df.iloc[:-1].copy())
    except Exception as e:
        print("[WARN] train_model 失敗（フォールバックで続行）:", e)

    # Load best gachi params
    k_sets, alpha_pair = load_best_params()

    # Iterate over draws and write predictions
    for i in range(10, len(df)):
        test_date = df.iloc[i]["抽せん日"]
        hist = df.iloc[:i].copy()
        try:
            decoded_sets, confidences = predictor.predict_gachi(hist, k_sets=k_sets, alpha_pair=alpha_pair)
            if decoded_sets is None:
                raise RuntimeError("predict_gachi returned None")
            print(f"[OK] {test_date.date()} decoded {len(decoded_sets)} sets")
            save_gachi_to_csv(list(zip(decoded_sets, confidences)), test_date, filename="loto7_predictions_gachi.csv")
        except Exception as e:
            print(f"[WARN] {test_date.date()} ガチ予測に失敗:", e)

    # Optional: make compatibility csv & evaluate if available
    try:
        if make_compat_for_evaluator:
            make_compat_for_evaluator("loto7_predictions_gachi.csv", "loto7_predictions_for_eval.csv")
        if evaluate_prediction_accuracy_with_bonus:
            evaluate_prediction_accuracy_with_bonus("loto7_predictions_for_eval.csv", "loto7.csv")
    except Exception as e:
        print("[WARN] 評価に失敗:", e)

    print("[DONE] run_gachi_bulk (fixed) 完了")

if __name__ == "__main__":
    main()
