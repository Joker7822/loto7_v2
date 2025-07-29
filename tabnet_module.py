import os
import torch
import torch.nn as nn
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor


def train_tabnet(X_train, y_train, n_output=7):
    """
    TabNet を使用して7つの数字を同時に予測するマルチターゲット回帰モデルを学習。
    """
    model = TabNetRegressor(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        verbose=10,
        seed=42,
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0,
    )
    return model


def predict_tabnet(model, X):
    """
    学習済みTabNetモデルを使用して予測を実施。
    """
    preds = model.predict(X)
    return preds


def save_tabnet_model(model, save_path):
    """
    TabNet モデルを指定ディレクトリに保存する
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_model(os.path.join(save_path, "tabnet_model.zip"))
    print(f"[INFO] TabNet モデルを保存しました → {save_path}")


def load_tabnet_model(load_path):
    """
    指定ディレクトリから TabNet モデルをロードする
    """
    model = TabNetRegressor()
    model.load_model(os.path.join(load_path, "tabnet_model.zip"))
    return model
