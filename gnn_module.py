import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx

class LotoGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(37, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc(x)

def build_loto_graph(df):
    G = nx.Graph()
    for nums in df['本数字']:
        for i in range(7):
            for j in range(i+1, 7):
                G.add_edge(nums[i]-1, nums[j]-1)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.eye(37)
    return Data(x=x, edge_index=edge_index)