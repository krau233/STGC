from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from models.stgnn import STGNN
import torch.nn.functional as F

config = dict(
    data=dict(
        type='cora',
        time_step=10,
    ),
    train=dict(
        seed=0,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        epoch=100,
    ),
)


# 加载数据




def train(loader,model,optimizer,device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out.view(-1, out.size(2)), data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(loader,model,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            out = out.view(-1, out.size(-1))
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
            total += data.y.size(0)
    return correct / total



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = np.load('data/Cora/cora_train_set.npz')
    train_adjacency_matrix = train_data['adjacency_matrix']
    train_feature_matrix = train_data['feature_matrix']
    train_labels = train_data['labels']
    print(f"Adjacency matrix shape: {train_adjacency_matrix.shape}")
    print(f"Feature matrix shape: {train_feature_matrix.shape}")
    print(f"Labels shape: {train_labels.shape}")

    # 将邻接矩阵转换为 PyTorch Geometric 的 edge_index 格式
    train_adj = sp.coo_matrix(train_adjacency_matrix)
    train_edge_index, train_edge_weight = from_scipy_sparse_matrix(train_adj)

    # 确保 edge_index 的形状正确
    print(f"edge_index shape: {train_edge_index.shape}")

    # 创建图数据列表，每个图数据包含单个时间步的数据
    time_steps = train_feature_matrix.shape[2]
    train_data_list = []
    for t in range(time_steps):
        x = torch.tensor(train_feature_matrix[:, :, t], dtype=torch.float)  # 形状为 (140, 1433)
        y = torch.tensor(train_labels, dtype=torch.long)  # 形状为 (140,)
        data_t = Data(x=x, edge_index=train_edge_index, y=y)
        train_data_list.append(data_t)

    # 创建数据加载器
    batch_size = 2
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    # 设备设置

    model = STGNN(in_channels=train_feature_matrix.shape[1], hidden_channels=16, lstm_hidden_channels=32,
                  out_channels=len(set(train_labels.tolist()))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print(f"Number of samples in dataset: {len(train_data_list)}")
    print(f"Batch size: {batch_size}")







    test_data = np.load('data/Cora/cora_test_set.npz')
    test_adjacency_matrix = test_data['adjacency_matrix']
    test_feature_matrix = test_data['feature_matrix']
    test_labels = test_data['labels']

    test_adj = sp.coo_matrix(test_adjacency_matrix)
    test_edge_index, test_edge_weight = from_scipy_sparse_matrix(test_adj)

    test_data_list = []
    for t in range(time_steps):
        x = torch.tensor(test_feature_matrix[:, :, t], dtype=torch.float)  # 形状为 (140, 1433)
        y = torch.tensor(test_labels, dtype=torch.long)  # 形状为 (140,)
        data_t = Data(x=x, edge_index=test_edge_index, y=y)
        test_data_list.append(data_t)

    test_loader = DataLoader(test_data_list)

    for epoch in range(1, 20):
        avg_loss = train(train_loader,model,optimizer,device)
        acc = test(test_loader,model,device)
        print(f'Epoch: {epoch:03d}, Average Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')



if __name__ == '__main__':
    main()