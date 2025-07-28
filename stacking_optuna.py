import numpy as np
import optuna

def optimize_stacking(train_preds, true_labels):
    def objective(trial):
        w = np.array([trial.suggest_float(k, 0, 1) for k in ['lstm', 'automl', 'gan', 'ppo']])
        if w.sum() == 0:
            return 0
        w /= w.sum()
        total = 0
        for i, t in enumerate(true_labels):
            pred = sum(w[j] * train_preds[k][i] for j, k in enumerate(['lstm', 'automl', 'gan', 'ppo']))
            pred = np.round(pred).astype(int)
            total += len(set(pred) & set(t))
        return total / len(true_labels)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params