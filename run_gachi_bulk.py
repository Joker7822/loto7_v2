
# run_gachi_bulk_autodiscover.py
# Dynamically discovers a Predictor class (LotteryPredictor / LotoPredictor / LotoPredictorGachi, etc.)
# and runs predict_gachi with tuned params if available.

import importlib
import inspect
import json
import pandas as pd

PREDICTOR = None
modules_tried = []
errors = []

def try_import(modname):
    try:
        m = importlib.import_module(modname)
        modules_tried.append(modname)
        return m
    except Exception as e:
        errors.append(f"[INFO] fail import {modname}: {e}")
        return None

def find_predictor_class(mod):
    candidates = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        lname = name.lower()
        if "predictor" in lname:
            if hasattr(obj, "predict_gachi") or hasattr(obj, "predict"):
                candidates.append(obj)
    return candidates

def choose_predictor():
    for modname in ["extended_predictor", "lottery_prediction", "lp_hotfix_monkeypatch"]:
        mod = try_import(modname)
        if not mod:
            continue
        if modname == "lp_hotfix_monkeypatch":
            mod2 = try_import("lottery_prediction")
            if mod2:
                mod = mod2
        preds = find_predictor_class(mod)
        if preds:
            return preds[0], mod
    return None, None

def get_preprocess():
    for modname in ["lottery_prediction", "extended_predictor"]:
        m = try_import(modname)
        if not m:
            continue
        fn = getattr(m, "preprocess_data", None)
        if callable(fn):
            return fn
    return None

def load_best_params(path="gachi_best_params.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            bp = json.load(f)
        return int(bp.get("k_sets", 10)), float(bp.get("alpha_pair", 0.3))
    except Exception as e:
        errors.append(f"[INFO] gachi_best_params.json load failed: {e}")
        return 10, 0.3

def main():
    Predictor, owner_mod = choose_predictor()
    if Predictor is None:
        print("\\n".join(errors))
        raise ImportError("No Predictor-like class with predict_gachi/predict found in: " + ", ".join(modules_tried))

    print(f"[INFO] Using Predictor: {Predictor.__name__} from module {owner_mod.__name__}")
    preprocess = get_preprocess()

    df = pd.read_csv("loto7.csv", encoding="utf-8-sig")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["抽せん日"]).sort_values("抽せん日").reset_index(drop=True)

    # instantiate
    predictor = None
    if preprocess:
        try:
            X, _, _ = preprocess(df.iloc[:-1].copy())
            in_size = X.shape[1] if hasattr(X, "shape") else 128
        except Exception:
            in_size = 128
    else:
        in_size = 128

    for sig in [(in_size, 128, 7), (), (in_size,)]:
        try:
            predictor = Predictor(*sig)
            break
        except Exception as e:
            errors.append(f"[INFO] ctor {sig} failed: {e}")
    if predictor is None:
        print("\\n".join(errors))
        raise RuntimeError("Could not instantiate predictor with common signatures.")

    train_fn = getattr(predictor, "train_model", None)
    if callable(train_fn):
        try:
            train_fn(df.iloc[:-1].copy())
        except Exception as e:
            errors.append(f"[WARN] train_model failed: {e}")

    k_sets, alpha_pair = load_best_params()

    def save_gachi(pairs, drawing_date, filename="loto7_predictions_gachi.csv"):
        drawing_date = pd.to_datetime(drawing_date).strftime("%Y-%m-%d")
        row = {"抽せん日": drawing_date}
        for i, (numbers, confidence) in enumerate(pairs[:10], 1):
            row[f"ガチ予測{i}"] = ", ".join(map(str, numbers))
            row[f"ガチ信頼度{i}"] = round(float(confidence), 3)
        df_row = pd.DataFrame([row])
        try:
            existing = pd.read_csv(filename, encoding="utf-8-sig")
            if "抽せん日" in existing.columns:
                existing = existing[existing["抽せん日"] != drawing_date]
            df_out = pd.concat([existing, df_row], ignore_index=True)
        except Exception:
            df_out = df_row
        df_out.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"[INFO] 保存: {filename} ({drawing_date})")

    use_gachi = hasattr(predictor, "predict_gachi")
    for i in range(10, len(df)):
        hist = df.iloc[:i].copy()
        day = df.iloc[i]["抽せん日"]
        try:
            if use_gachi:
                decoded, conf = predictor.predict_gachi(hist, k_sets=k_sets, alpha_pair=alpha_pair)
                pairs = list(zip(decoded, conf))
            else:
                nums, confs = predictor.predict(hist, num_candidates=10)
                pairs = list(zip(nums, confs))
            save_gachi(pairs, day, filename="loto7_predictions_gachi.csv")
            print(f"[OK] {day.date()}")
        except Exception as e:
            errors.append(f"[WARN] {day.date()} failed: {e}")
            continue

    if errors:
        print("---- LOG ----")
        print("\\n".join(errors))

    print("[DONE] run_gachi_bulk_autodiscover 完了")

if __name__ == "__main__":
    main()
