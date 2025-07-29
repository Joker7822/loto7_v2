# === CPU専用修正版 for test_with_gnn.py ===

import os
import numpy as np
import pandas as pd
import torch
import onnxruntime
from autogluon.tabular import TabularPredictor

# ✅ デバイスを強制的に CPU に固定
DEVICE = torch.device("cpu")

# ✅ AutoGluon を CPU で強制的に学習
def force_cpu_predictor_fit(X, y, model_index):
    df_train = pd.DataFrame(X)
    df_train['target'] = y[:, model_index]
    predictor = TabularPredictor(label='target', path=f'autogluon_model_pos{model_index}').fit(
        df_train,
        excluded_model_types=['KNN', 'NN_TORCH'],
        hyperparameters={
            'GBM': {'device': 'cpu', 'num_boost_round': 300},
            'XGB': {'tree_method': 'hist', 'n_estimators': 300},
            'CAT': {'task_type': 'CPU', 'iterations': 300},
            'RF': {'n_estimators': 200}
        },
        num_gpus=0
    )
    return predictor

# ✅ ONNXモデル（LSTM）の読み込みをCPUで
onnx_session = None
onnx_model_path = "lstm_model.onnx"
if os.path.exists(onnx_model_path):
    try:
        onnx_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        print("[INFO] ONNX LSTM モデルをCPUで読み込みました")
    except Exception as e:
        print(f"[WARNING] ONNXモデルの読み込みに失敗しました: {e}")
else:
    print("[WARNING] ONNXモデルが存在しません: スキップします")

# ✅ GNN学習・推論をCPUで（past_dataが必要）
gnn_model = None
gnn_scores = np.zeros(37)

try:
    from gnn_module import LotoGNN, build_loto_graph

    # `past_data` がスコープ外なら渡すようにしてください
    if 'past_data' not in globals():
        raise ValueError("変数 'past_data' が定義されていません。GNNのグラフ生成に必要です。")

    graph_data = build_loto_graph(past_data).to(DEVICE)
    gnn_model = LotoGNN().to(DEVICE)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    gnn_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = gnn_model(graph_data.x, graph_data.edge_index)
        loss = out.mean()
        loss.backward()
        optimizer.step()

    print("[INFO] GNN モデルの学習完了（CPU）")

    # GNN推論
    gnn_model.eval()
    with torch.no_grad():
        gnn_scores = gnn_model(graph_data.x, graph_data.edge_index).squeeze().cpu().numpy()

except Exception as e:
    print(f"[WARNING] GNN モデルをスキップしました: {e}")
