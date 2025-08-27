
import lp_hotfix_monkeypatch
import pandas as pd
import os

from extended_predictor import LotoPredictorGachi as LotoPredictor
from lottery_prediction import (
    set_global_seed,
    preprocess_data,
    save_self_predictions,
    save_predictions_to_csv,
    evaluate_prediction_accuracy_with_bonus,
    git_commit_and_push,
    _save_all_models_no_self,
)

def make_compat_for_evaluator(in_csv="loto7_predictions_gachi.csv",
                              out_csv="loto7_predictions_for_eval.csv"):
    df = pd.read_csv(in_csv, encoding="utf-8-sig")
    out = df[["抽せん日"]].copy()
    for i in range(1, 6):
        src_p = f"ガチ予測{i}"
        src_c = f"ガチ信頼度{i}"
        if src_p in df.columns:
            out[f"予測{i}"] = df[src_p]
        if src_c in df.columns:
            out[f"信頼度{i}"] = df[src_c]
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 互換CSVを生成: {out_csv}")

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

def bulk_predict_all_past_draws():
    set_global_seed(42)

    df = pd.read_csv("loto7.csv", encoding="utf-8-sig")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
    df = df.sort_values("抽せん日").reset_index(drop=True)
    print("[INFO] 抽せんデータ読み込み完了:", len(df), "件")

    pred_file = "loto7_predictions.csv"

    skip_dates = set()
    if os.path.exists(pred_file):
        try:
            pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
            if "抽せん日" in pred_df.columns:
                skip_dates = set(pd.to_datetime(pred_df["抽せん日"], errors='coerce').dropna().dt.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"[WARNING] 予測ファイル読み込みエラー: {e}")
    else:
        with open(pred_file, "w", encoding="utf-8-sig") as f:
            f.write("抽せん日,予測1,信頼度1,予測2,信頼度2,予測3,信頼度3,予測4,信頼度4,予測5,信頼度5\n")

    predictor_cache = {}

    for i in range(10, len(df)):
        set_global_seed(1000 + i)

        test_date = df.iloc[i]["抽せん日"]
        test_date_str = test_date.strftime("%Y-%m-%d")

        if test_date_str in skip_dates:
            print(f"[INFO] 既に予測済み: {test_date_str} → スキップ")
            continue

        print(f"\n=== {test_date_str} の予測を開始 ===")
        train_data = df.iloc[:i].copy()
        latest_data = df.iloc[i-10:i].copy()

        X, _, _ = preprocess_data(train_data)
        if X is None:
            print(f"[WARNING] {test_date_str} の学習データが無効です")
            continue

        input_size = X.shape[1]

        if i % 50 == 0 or input_size not in predictor_cache:
            print(f"[INFO] モデル再学習: {test_date_str} 時点")
            predictor = LotoPredictor(input_size, 128, 7)
            try:
                predictor.train_model(train_data)
            except Exception as e:
                print("[WARN] train_model 失敗（フォールバックで続行）:", e)
            predictor_cache[input_size] = predictor
        else:
            predictor = predictor_cache[input_size]

        try:

            res = predictor.predict(latest_data)

            if not (isinstance(res, tuple) and len(res) == 2):

                raise RuntimeError("predict() must return (predictions, confidence_scores)")

            predictions, confidence_scores = res
        except Exception as e:
            print(f"[ERROR] 通常predict失敗: {e}")
            predictions, confidence_scores = None, None

        if predictions is None:
            print(f"[ERROR] {test_date_str} の通常予測に失敗しました（スキップ）")
        else:
            try:
                verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), train_data)  # noqa
            except Exception:
                verified_predictions = list(zip(predictions, confidence_scores))
            save_self_predictions(verified_predictions)
            save_predictions_to_csv(verified_predictions, test_date)
            git_commit_and_push("loto7_predictions.csv", "Auto update loto7_predictions.csv [skip ci]")

        try:
            decoded_sets, confidences = predictor.predict_gachi(df.iloc[:i], k_sets=10, alpha_pair=0.3)
            if decoded_sets is not None:
                print("\n=== ガチ予測 Top 10 ===")
                for idx, (nums, conf) in enumerate(zip(decoded_sets[:10], confidences[:10]), 1):
                    print(f"{idx:02d}: {sorted(nums)}  信頼度={conf:.3f}")
                save_gachi_to_csv(list(zip(decoded_sets, confidences)), test_date)
        except Exception as e:
            print("[WARN] ガチ予測に失敗しました:", e)

        model_dir = f"models/{test_date_str}"
        try:
            _save_all_models_no_self(predictor, model_dir)
        except Exception as e:
            print("[WARN] モデル保存に失敗:", e)

        try:
            make_compat_for_evaluator("loto7_predictions_gachi.csv", "loto7_predictions_for_eval.csv")
            evaluate_prediction_accuracy_with_bonus("loto7_predictions_for_eval.csv", "loto7.csv")
        except Exception as e:
            print("[WARN] 評価に失敗:", e)

    try:
        future_date = df["抽せん日"].max() + pd.Timedelta(days=7)
        future_date_str = future_date.strftime("%Y-%m-%d")

        if future_date_str not in skip_dates:
            print(f"\n=== {future_date_str} の未来予測を開始 ===")
            latest_data = df.tail(10).copy()
            train_data = df.copy()

            X, _, _ = preprocess_data(train_data)
            if X is None:
                print("[WARNING] 未来予測用の学習データが無効です")
            else:
                input_size = X.shape[1]
                if input_size not in predictor_cache:
                    predictor = LotoPredictor(input_size, 128, 7)
                    try:
                        predictor.train_model(train_data)
                    except Exception as e:
                        print("[WARN] train_model 失敗（フォールバックで続行）:", e)
                    predictor_cache[input_size] = predictor
                else:
                    predictor = predictor_cache[input_size]

                try:

                    res = predictor.predict(latest_data)

                    if not (isinstance(res, tuple) and len(res) == 2):

                        raise RuntimeError("predict() must return (predictions, confidence_scores)")

                    predictions, confidence_scores = res
                except Exception as e:
                    print(f"[ERROR] 未来の通常predict失敗: {e}")
                    predictions, confidence_scores = None, None

                if predictions is not None:
                    try:
                        verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), train_data)  # noqa
                    except Exception:
                        verified_predictions = list(zip(predictions, confidence_scores))
                    save_self_predictions(verified_predictions)
                    save_predictions_to_csv(verified_predictions, future_date)
                    git_commit_and_push("loto7_predictions.csv", "Auto predict future draw [skip ci]")

                try:
                    decoded_sets, confidences = predictor.predict_gachi(df, k_sets=10, alpha_pair=0.3)
                    if decoded_sets is not None:
                        save_gachi_to_csv(list(zip(decoded_sets, confidences)), future_date)
                except Exception as e:
                    print("[WARN] 未来のガチ予測に失敗:", e)

                print(f"[INFO] 未来予測（{future_date_str}）完了")
        else:
            print(f"[INFO] 未来予測（{future_date_str}）は既に実行済みです")

    except Exception as e:
        print("[WARN] 未来予測ブロックで例外:", e)

if __name__ == "__main__":
    bulk_predict_all_past_draws()
