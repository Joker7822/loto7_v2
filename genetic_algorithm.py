import numpy as np
import torch
import random

def evolve_candidates(model_obj, X_df, generations=10, population_size=50):
    candidates = []

    for i in range(population_size):
        try:
            input_size = model_obj.lstm_model.lstm.input_size
            if X_df.shape[1] != input_size:
                print(f"[WARNING] 特徴量の次元不一致: X={X_df.shape[1]}, expected={input_size}")
                continue

            # --- ランダムサンプル抽出 ---
            idx = np.random.randint(0, len(X_df))
            sample_X = X_df.iloc[[idx]]

            # --- LSTM予測 ---
            try:
                x_tensor = torch.tensor(sample_X.values.reshape(1, 1, -1), dtype=torch.float32)
            except Exception as e:
                print(f"[GA] reshapeエラー: shape={sample_X.shape}, input_size={input_size}, error={e}")
                continue

            with torch.no_grad():
                lstm_pred = model_obj.lstm_model(x_tensor).detach().numpy()[0]
            lstm_vec = np.round(lstm_pred).astype(int)

            # --- AutoML予測 ---
            automl_pred = np.array([
                model_obj.regression_models[i].predict(sample_X)[0]
                for i in range(7)
            ])
            automl_vec = np.round(automl_pred).astype(int)

            # --- GANベクトル ---
            if hasattr(model_obj.gan_model, "generate_vector"):
                gan_vec = model_obj.gan_model.generate_vector()
            elif model_obj.gan_model:
                gan_vec = model_obj.gan_model.generate_samples(1)[0]
            else:
                gan_vec = np.random.rand(37)

            # --- PPOベクトル ---
            obs = np.zeros(37, dtype=np.float32)
            if hasattr(model_obj.ppo_model, "predict"):
                action, _ = model_obj.ppo_model.predict(obs, deterministic=False)
                ppo_vec = np.array(action)
            else:
                ppo_vec = np.random.rand(37)

            candidates.append((lstm_vec, automl_vec, gan_vec, ppo_vec))

        except Exception as e:
            print(f"[GA] 候補生成エラー (index {i}): {e}")

    return candidates
