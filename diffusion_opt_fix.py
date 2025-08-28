
import optuna
import numpy as np
import pandas as pd
import json
from lottery_prediction import convert_numbers_to_binary_vectors
from diffusion_module import train_diffusion_ddpm

def evaluate_diffusion(model, sample_size=256):
    # Try to sample from model
    if hasattr(model, "generate_samples"):
        samples = model.generate_samples(sample_size)
    elif hasattr(model, "sample"):
        samples = model.sample(sample_size)
    else:
        return 0.0
    samples = np.asarray(samples)
    # Encourage ~7 ones out of 37
    ones = samples.round().sum(axis=1)
    sparsity_score = -np.mean(np.abs(ones - 7))
    # Diversity via pairwise Hamming (fallback if SciPy not available)
    try:
        from scipy.spatial.distance import pdist
        diversity = pdist(samples.round(), metric="hamming").mean()
    except Exception:
        diversity = float(np.mean(np.std(samples.round(), axis=0)))
    return float(sparsity_score + diversity)

def objective(trial, data_bin):
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    epochs = trial.suggest_int("epochs", 300, 1200)
    model, _, _ = train_diffusion_ddpm(data_bin, epochs=epochs, batch_size=batch_size)
    return evaluate_diffusion(model, sample_size=256)

def main():
    data = pd.read_csv("loto7_prediction_evaluation_with_bonus.csv", encoding="utf-8-sig")
    data_bin = convert_numbers_to_binary_vectors(data)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, data_bin), n_trials=10)
    print("Best:", study.best_params)
    with open("diffusion_best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
