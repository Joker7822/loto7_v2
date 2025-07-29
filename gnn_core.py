# === 🔁 改良版 gnn_core.py ===
import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import pandas as pd

class LotoGNN(nn.Module):
    def __init__(self, input_dim=37, hidden_dim=64, output_dim=37):
        super(LotoGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def build_cooccurrence_graph(df):
    edge_dict = {}
    node_features = np.zeros((37, 1))  # 出現回数ベースの特徴量

    freq = np.zeros(37)
    for _, row in df.iterrows():
        numbers = row['本数字']
        if isinstance(numbers, str):
            numbers = [int(n) for n in numbers.strip("[]").split(",") if n.strip().isdigit()]
        for i in numbers:
            if 1 <= i <= 37:
                freq[i - 1] += 1
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                a, b = sorted([numbers[i], numbers[j]])
                if 1 <= a <= 37 and 1 <= b <= 37:
                    edge_dict[(a - 1, b - 1)] = edge_dict.get((a - 1, b - 1), 0) + 1

    node_features[:, 0] = freq / freq.max()
    edge_index = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index)
