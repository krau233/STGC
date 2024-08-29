import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
class STCNTK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, num_timesteps):
        super(STCNTK, self).__init__()
        self.num_timesteps = num_timesteps
        self.chebconv = ChebConv(in_channels,out_channels=hidden_channels,K=K)

    def forward(self, x,edge_index):
        x, edge_index = x, edge_index
        out = torch.zeros(x.size(0), self.num_timesteps, self.chebconv.out_channels)
        for t in range(self.num_timesteps):
            xt = x[:, t, :]
            out[:, t, :] = self.chebconv(xt,edge_index)
        out = torch.fft.fft(out,dim=1)
        
        return out