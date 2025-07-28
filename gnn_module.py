# === CPU専用修正版 for test_with_gnn.py ===
# 以下のコードは、GPU環境が使えない環境でもAutoGluon、GNN、LSTM等が正しく動作するように修正されたパッチです。

import torch
from autogluon.tabular import TabularPredictor

# ✅ device を常に CPU に強制
DEVICE = torch.device("cpu")

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

# ✅ train_model 内で device を CPU に固定
device = DEVICE

# ✅ ONNX provider
import onnxruntime
onnx_session = onnxruntime.InferenceSession(
    "lstm_model.onnx",
    providers=['CPUExecutionProvider']
)

# ✅ GNNの学習部分で GPU を使わないようにする
try:
    from gnn_module import LotoGNN, build_loto_graph
    graph_data = build_loto_graph(past_data)
    gnn_model = LotoGNN().to(DEVICE)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    gnn_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = gnn_model(graph_data.to(DEVICE))
        loss = out.mean()
        loss.backward()
        optimizer.step()
    print("[INFO] GNN モデルの学習完了（CPU）")
except Exception as e:
    print(f"[WARNING] GNN モデルをスキップ: {e}")
    gnn_model = None

# ✅ GNN の予測部分（予測時も CPU）
if gnn_model:
    gnn_model.eval()
    with torch.no_grad():
        gnn_scores = gnn_model(graph_data.to(DEVICE)).squeeze().cpu().numpy()
else:
    gnn_scores = np.zeros(37)  # 予測できない場合はスコアをゼロに
