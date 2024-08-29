import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from torch.utils import data


class Cora(data.Dataset):
    def __init__(self, data_type):
        self.load_cora(data_type)
        self.preprocess_cora()

    def load_cora(self, data_type):
        # Load Cora dataset
        dataset_cora = Planetoid(root="../data", name='Cora')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = dataset_cora[0].to(device)
        self.feature_matrix = data.x
        self.labels = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        graph = to_networkx(data, to_undirected=True)
        self.adjacency_matrix = torch.from_numpy(nx.adjacency_matrix(graph).todense()).to(device)
        if data_type == 'train':
            self.feature_matrix = self.feature_matrix[train_mask]
            self.labels = self.labels[train_mask]
            self.adjacency_matrix = self.adjacency_matrix[train_mask].t()[train_mask].t()
        elif data_type == 'val':
            self.feature_matrix = self.feature_matrix[val_mask]
            self.labels = self.labels[val_mask]
            self.adjacency_matrix = self.adjacency_matrix[val_mask].t()[val_mask].t()
        elif data_type == 'test':
            self.feature_matrix = self.feature_matrix[test_mask]
            self.labels = self.labels[test_mask]
            self.adjacency_matrix = self.adjacency_matrix[test_mask].t()[test_mask].t()
        elif data_type == 'all':
            return
        # 图，邻接矩阵，特征矩阵，标签
        return

    # 添加时序噪声
    def preprocess_cora(self, num_time_steps=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        low, high = -1, 1
        # Initialize list for time-based graphs
        result = []
        result.append(self.feature_matrix)
        np.random.seed(0)
        # 可能的噪声值
        values = [0, 1, -1]
        # 每个值出现的概率
        probabilities = [0.90, 0.05, 0.05]
        for t in range(num_time_steps - 1):
            noise = torch.from_numpy(np.random.choice(values, p=probabilities, size=self.feature_matrix.shape)).to(
                device)
            noise_features = result[-1] + noise

            # Add features to nodes
            result.append(torch.clamp((noise_features.clone()), min=0, max=1))

        self.feature_matrix = torch.stack(result).to(device)


if __name__ == '__main__':

    a =[]
    for i in range(20):
        print(i)
        a.append(Cora('train').feature_matrix)
    training_set = torch.stack(a).permute(0,2,1,3)

    np.savez('../data/Cora/cora_train_set.npz', adjacency_matrix=Cora('train').adjacency_matrix.cpu().numpy(),
             feature_matrix=training_set.cpu().numpy(), labels=Cora('train').labels.cpu().numpy())


    # val_set = Cora('val')
    # np.savez('../data/Cora/cora_val_set.npz', adjacency_matrix=val_set.adjacency_matrix.cpu().numpy(),
    #          feature_matrix=val_set.feature_matrix.cpu().numpy(), labels=val_set.labels.cpu().numpy())
    #
    #
    #
    # test_set = Cora('test')
    # np.savez('../data/Cora/cora_test_set.npz', adjacency_matrix=test_set.adjacency_matrix.cpu().numpy(),
    #          feature_matrix=test_set.feature_matrix.cpu().numpy(), labels=test_set.labels.cpu().numpy())
    #
    #
    #
    # all_set = Cora('all')
    # np.savez('../data/Cora/Cora.npz', adjacency_matrix=all_set.adjacency_matrix.cpu().numpy(),
    #          feature_matrix=all_set.feature_matrix.cpu().numpy(), labels=all_set.labels.cpu().numpy())
