# optimize_all_models.py (replacement)
import optuna
import pandas as pd
from lottery_prediction import parse_number_string, classify_rank
from diffusion_module import train_diffusion_ddpm

RANK_VALUE = {"1等":6,"2等":5,"3等":4,"4等":3,"5等":2,"6等":1,"該当なし":0}

def load_truth(csv_path="loto7_prediction_evaluation_with_bonus.csv"):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    def to_list(col):
        out = []
        for v in df[col].tolist():
            try:
                out.append(parse_number_string(str(v)))
            except Exception:
                out.append([])
        return out
    if "当選本数字" not in df.columns or "当選ボーナス" not in df.columns:
        raise RuntimeError("評価CSVに当選列がありません")
    main = to_list("当選本数字")
    bonus = to_list("当選ボーナス")
    return list(zip(main, bonus))

def vecs_to_sets(vectors, topk=7):
    import numpy as np
    out = []
    for v in vectors:
        v = np.asarray(v).reshape(-1)
        if v.size != 37:
            v = np.resize(v, (37,))
        idx = np.argsort(-v)[:topk] + 1
        out.append(sorted(set(idx.tolist()))[:7])
    return out

def score_sets(cands, truth_rows):
    tot = 0.0
    for main, bonus in truth_rows:
        best = 0
        sm, sb = set(main), set(bonus)
        for cand in cands:
            r = classify_rank(len(sm & set(cand)), len(sb & set(cand)))
            best = max(best, RANK_VALUE.get(r,0))
        tot += best
    return tot / max(1,len(truth_rows))

def optimize_diffusion(data_bin):
    truth_rows = load_truth()
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [32,64])
        epochs = trial.suggest_int("epochs", 300, 1500)
        model,_,_ = train_diffusion_ddpm(data_bin, epochs=epochs, batch_size=batch_size)
        if hasattr(model,"sample"):
            samples = model.sample(400)
        elif hasattr(model,"generate_samples"):
            samples = model.generate_samples(400)
        else:
            return 0.0
        sets = vecs_to_sets(samples, topk=7)
        return score_sets(sets, truth_rows)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    return study.best_params

def optimize_stacking_wrapper(base_preds, y_train):
    # placeholder: real OOF preds must be provided by caller
    return {"note":"Provide real OOF preds to optimize_stacking"}

if __name__ == "__main__":
    print("Replacement optimize_all_models ready. Integrate into pipeline.")
