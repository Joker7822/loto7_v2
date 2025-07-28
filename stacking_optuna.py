import numpy as np
import optuna

def optimize_stacking(train_preds, true_labels):
    def objective(trial):
        # 各モデルに対する重みを提案
        w = np.array([trial.suggest_float(k, 0, 1) for k in ['lstm', 'automl', 'gan', 'ppo']])
        if w.sum() == 0:
            return 0
        w /= w.sum()

        total = 0
        valid_count = 0

        for i, t in enumerate(true_labels):
            try:
                # 各モデルの予測を取得・チェック・加重
                vectors = []
                for j, k in enumerate(['lstm', 'automl', 'gan', 'ppo']):
                    vec = np.array(train_preds[k][i], dtype=np.float32)
                    if vec.shape != (7,):
                        raise ValueError(f"Invalid shape from {k}: {vec.shape}")
                    vectors.append(w[j] * vec)

                # 加重平均 → 四捨五入して整数化
                pred = np.sum(vectors, axis=0)
                pred = np.round(pred).astype(int)

                total += len(set(pred) & set(t))
                valid_count += 1

            except Exception as e:
                print(f"[WARN] スタッキング最適化スキップ (index={i}): {e}")
                continue

        # 有効データが1件もない場合はゼロを返す
        if valid_count == 0:
            return 0

        return total / valid_count

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params
