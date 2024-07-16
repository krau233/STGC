import torch
from torch_geometric.transforms import BaseTransform

# 独热编码
def to_one_hot(targets, num_classes):
    one_hot = torch.zeros(len(targets), num_classes)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    return one_hot

#高斯噪声
def add_gaussian_noise(matrix, mean=0.0, std=1.0):
    noise = (torch.randn(matrix.size()) * std + mean).to('cuda')
    noisy_matrix = matrix + noise
    return noisy_matrix