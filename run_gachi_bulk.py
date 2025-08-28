# run_gachi_bulk.py (replacement)
import pandas as pd
import json
from lottery_prediction import LotoPredictor

def bulk_predict(csv_path="loto7.csv", out_path="predictions.csv"):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    predictor = LotoPredictor(...)
    preds = []
    try:
        with open("gachi_best_params.json","r",encoding="utf-8") as f:
            bp = json.load(f)
        k_sets = int(bp.get("k_sets",10))
        alpha_pair = float(bp.get("alpha_pair",0.3))
    except Exception:
        k_sets, alpha_pair = 10,0.3
    for i in range(len(df)):
        try:
            decoded_sets, confs = predictor.predict_gachi(df.iloc[:i], k_sets=k_sets, alpha_pair=alpha_pair)
            preds.append(decoded_sets)
        except Exception as e:
            print("[WARN] predict_gachi failed at", i, e)
            preds.append([])
    pd.DataFrame({"predictions":preds}).to_csv(out_path,index=False)
    print("Saved",out_path)

if __name__=="__main__":
    bulk_predict()
