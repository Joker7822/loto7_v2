import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import json
import os

from lottery_prediction import git_commit_and_push

from tabnet_module import train_tabnet
from autogluon.tabular import TabularPredictor
from diffusion_module import train_diffusion_ddpm
from stacking_optuna import optimize_stacking
from test_with_gnn import preprocess_data, convert_numbers_to_binary_vectors

RESULT_DIR = "optuna_results"
os.makedirs(RESULT_DIR, exist_ok=True)

def optimize_autogluon(X, y, model_index):
    def objective(trial):
        params = {
            'GBM': {'extra_trees': trial.suggest_categorical('gbm_extra_trees', [True, False])},
            'CAT': {'iterations': trial.suggest_int('cat_iterations', 300, 1000)},
            'XGB': {'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 500)},
            'RF': {'n_estimators': trial.suggest_int('rf_n_estimators', 100, 300)},
        }

        df = pd.DataFrame(X)
        df['target'] = y[:, model_index]
        predictor = TabularPredictor(label='target', verbosity=0).fit(
            df,
            hyperparameters=params,
            presets="best_quality",
            time_limit=300,
        )
        return predictor.leaderboard(silent=True).iloc[0]['score_val']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    return study.best_params

def optimize_tabnet(X, y):
    def objective(trial):
        from pytorch_tabnet.tab_model import TabNetRegressor
        model = TabNetRegressor(
            n_d=trial.suggest_int("n_d", 8, 64),
            n_a=trial.suggest_int("n_a", 8, 64),
            n_steps=trial.suggest_int("n_steps", 3, 10),
            gamma=trial.suggest_float("gamma", 1.0, 2.0),
            lambda_sparse=trial.suggest_float("lambda_sparse", 1e-5, 1e-1),
            seed=42
        )
        model.fit(
            X_train=X, y_train=y,
            eval_set=[(X, y)],
            max_epochs=50,
            patience=10,
            batch_size=trial.suggest_categorical("batch_size", [64, 128]),
            virtual_batch_size=trial.suggest_categorical("virtual_batch_size", [32, 64])
        )
        preds = model.predict(X)
        return -np.mean((y - preds) ** 2)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    return study.best_params

def optimize_diffusion(data_bin):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        epochs = trial.suggest_int("epochs", 500, 2000)

        model, _, _ = train_diffusion_ddpm(data_bin, epochs=epochs, batch_size=batch_size)
        return -epochs / batch_size  # 仮スコア（改善余地あり）

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    return study.best_params

def optimize_stacking_wrapper(train_preds, true_labels):
    return optimize_stacking(train_preds, true_labels)

def main():
    data = pd.read_csv("loto7_prediction_evaluation_with_bonus.csv", encoding='utf-8-sig')
    X, y, _ = preprocess_data(data)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n=== AutoGluon最適化開始 ===")
    for i in range(7):
        ag_params = optimize_autogluon(X_train, y_train, i)
        with open(f"{RESULT_DIR}/autogluon_pos{i}.json", "w") as f:
            json.dump(ag_params, f, indent=2)
        print(f"Pos{i} → {ag_params}")

    print("\n=== TabNet最適化開始 ===")
    tabnet_params = optimize_tabnet(X_train, y_train)
    with open(f"{RESULT_DIR}/tabnet.json", "w") as f:
        json.dump(tabnet_params, f, indent=2)
    print(f"TabNet → {tabnet_params}")

    print("\n=== Diffusion最適化開始 ===")
    data_bin = convert_numbers_to_binary_vectors(data)
    diff_params = optimize_diffusion(data_bin)
    with open(f"{RESULT_DIR}/diffusion.json", "w") as f:
        json.dump(diff_params, f, indent=2)
    print(f"Diffusion → {diff_params}")

    print("\n=== スタッキング重み最適化 ===")
    dummy_preds = {
        'lstm': y_train[:, :7],
        'automl': y_train[:, :7],
        'gan': y_train[:, :7],
        'ppo': y_train[:, :7],
    }
    stack_params = optimize_stacking_wrapper(dummy_preds, y_train.tolist())
    with open(f"{RESULT_DIR}/stacking.json", "w") as f:
        json.dump(stack_params, f, indent=2)
    print(f"Stacking → {stack_params}")
    # Push all updated Optuna params
    git_commit_and_push(RESULT_DIR, "Update Optuna hyperparameters")

if __name__ == "__main__":
    main()

