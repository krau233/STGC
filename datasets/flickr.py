import torch
import os
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from torch.utils import data
from torch_geometric.datasets import Flickr
from torch_geometric.transforms import NormalizeFeatures

import utils


class Flickr_dataset(data.Dataset):
    def __init__(self, data_type,num_time_steps):
        self.load_flickr(data_type)
        self.preprocess_flickr(num_time_steps)

    def load_flickr(self, data_type):
        assert data_type in ['train', 'test', 'val', 'all']
        # Load flickr dataset
        transform = NormalizeFeatures()
        current_path = os.path.dirname(__file__)
        dataset_flickr = Flickr(root=current_path[0:current_path.rindex('/')] + r'/data/Flickr',transform=transform)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = dataset_flickr[0].to(device)
        self.feature_matrix = data.x.to(device)
        self.labels = data.y.to(device)
        train_mask = data.train_mask.cpu()
        val_mask = data.val_mask.cpu()
        test_mask = data.test_mask.cpu()
        graph = to_networkx(data, to_undirected=True)

        csr_matrix =nx.adjacency_matrix(graph)

        if data_type == 'train':
            self.feature_matrix = self.feature_matrix[train_mask]
            self.labels = self.labels[train_mask]
            self.adjacency_matrix = csr_matrix[train_mask].transpose()[train_mask].transpose()
        elif data_type == 'val':
            self.feature_matrix = self.feature_matrix[val_mask]
            self.labels = self.labels[val_mask]
            self.adjacency_matrix = csr_matrix[val_mask].transpose()[val_mask].transpose()

        elif data_type == 'test':
            self.feature_matrix = self.feature_matrix[test_mask]
            self.labels = self.labels[test_mask]
            self.adjacency_matrix = csr_matrix[test_mask].transpose()[test_mask].transpose()
        elif data_type == 'all':
            self.adjacency_matrix = csr_matrix
        # 图，邻接矩阵，特征矩阵，标签
        return

    def preprocess_flickr(self,num_time_steps):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        low, high = -1, 1
        # Initialize list for time-based graphs
        result = []
        result.append(self.feature_matrix)
        np.random.seed(0)

        for t in range(num_time_steps - 1):
            noise_features = utils.add_gaussian_noise(result[-1],std=0.01)

            # Add features to nodes
            result.append(torch.clamp((noise_features.clone()), min=0, max=1))

        self.feature_matrix = torch.stack(result).to(device)


training_set = Flickr_dataset('all',10)
matrix = training_set.adjacency_matrix
matrix_dict = {
    'data': matrix.data,
    'indices': matrix.indices,
    'indptr': matrix.indptr,
    'shape': matrix.shape
}
np.savez('../data/Flickr/flickr.npz', adjacency_matrix=matrix_dict,
         feature_matrix=training_set.feature_matrix.cpu().numpy(), labels=training_set.labels.cpu().numpy())

#
# val_set = Flickr_dataset('val',10)
# np.savez('..data/Flickr/flickr_val_set.npz', adjacency_matrix=val_set.adjacency_matrix.to_dense().cpu().numpy(),
#          feature_matrix=val_set.feature_matrix.cpu().numpy(), labels=val_set.labels.cpu().numpy())
#
#
#
# test_set = Flickr_dataset('test',10)
# np.savez('..data/Flickr/flickr_test_set.npz', adjacency_matrix=test_set.adjacency_matrix.to_dense().cpu().numpy(),
#          feature_matrix=test_set.feature_matrix.cpu().numpy(), labels=test_set.labels.cpu().numpy())
#
#
#
# all_set = Flickr_dataset('all',10)
# np.savez('..data/Flickr/flickr.npz', adjacency_matrix=all_set.adjacency_matrix.to_dense().cpu().numpy(),
#          feature_matrix=all_set.feature_matrix.cpu().numpy(), labels=all_set.labels.cpu().numpy())