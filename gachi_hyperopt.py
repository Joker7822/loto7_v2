
import optuna
import pandas as pd
import numpy as np
import json
from lottery_prediction import LotoPredictor, preprocess_data, parse_number_string, classify_rank

def load_loto_data(csv_path="loto7.csv"):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    def norm_list(series):
        out = []
        for v in series.tolist():
            if isinstance(v, list):
                out.append(v)
            else:
                try:
                    out.append(parse_number_string(str(v)))
                except Exception:
                    out.append([])
        return out
    df["本数字"] = norm_list(df["本数字"])
    df["ボーナス数字"] = norm_list(df["ボーナス数字"])
    df = df.dropna(subset=["抽せん日"]).sort_values("抽せん日").reset_index(drop=True)
    return df

def split_by_date(df, cutoff_ratio=0.8):
    idx = int(len(df) * cutoff_ratio)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()

def score_sets(pred_sets, actual_main, actual_bonus):
    rank_to_value = {"1等": 6, "2等": 5, "3等": 4, "4等": 3, "5等": 2, "6等": 1, "該当なし": 0}
    best = 0
    for item in pred_sets:
        nums = item[0] if isinstance(item, (list, tuple)) and len(item)>0 and isinstance(item[0], (int, np.integer)) else item
        main_match = len(set(nums) & set(actual_main))
        bonus_match = len(set(nums) & set(actual_bonus))
        r = classify_rank(main_match, bonus_match)
        best = max(best, rank_to_value.get(r, 0))
    return best

def make_predictor_from_train(train_df):
    X, y, scaler = preprocess_data(train_df)
    if X is None or len(X)==0:
        raise RuntimeError("Preprocess returned empty features")
    input_size = X.shape[1]
    predictor = LotoPredictor(input_size, 128, 7)
    try:
        predictor.train_model(train_df)
    except Exception as e:
        print("[WARN] train_model failed, fallback may be used:", e)
    return predictor

def objective(trial):
    alpha_pair = trial.suggest_float("alpha_pair", 0.0, 1.5)
    k_sets = trial.suggest_int("k_sets", 5, 20)
    cutoff = trial.suggest_float("cutoff_ratio", 0.7, 0.9)
    df = load_loto_data("loto7.csv")
    train, valid = split_by_date(df, cutoff_ratio=cutoff)
    predictor = make_predictor_from_train(train)
    total = 0
    count = 0
    for i in range(len(valid)):
        hist = pd.concat([train, valid.iloc[:i]], ignore_index=True)
        try:
            decoded_sets, conf = predictor.predict_gachi(hist, k_sets=k_sets, alpha_pair=alpha_pair)
        except Exception as e:
            print("[WARN] predict_gachi failed on iter", i, ":", e)
            continue
        actual_main = valid.iloc[i]["本数字"]
        actual_bonus = valid.iloc[i]["ボーナス数字"]
        total += score_sets(list(zip(decoded_sets, conf)), actual_main, actual_bonus)
        count += 1
    return (total / count) if count else 0.0

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    print("Best params:", study.best_params)
    with open("gachi_best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
