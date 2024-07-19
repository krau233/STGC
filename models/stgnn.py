import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class STGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, lstm_hidden_channels, out_channels):
        super(STGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.lstm = torch.nn.LSTM(hidden_channels, lstm_hidden_channels, batch_first=True)
        self.conv2 = GCNConv(lstm_hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 根据 batch 维度将 x 重新排列为 [batch_size, num_nodes, num_features]
        batch_size = batch.max().item() + 1
        num_nodes = x.size(0) // batch_size
        x = x.view(batch_size, num_nodes, -1)
        # print(x.shape)
        # GCNConv expects input shape [num_nodes * batch_size, num_features]
        x = x.view(batch_size * num_nodes, -1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Reshape back to [batch_size, num_nodes, hidden_channels]
        x = x.view(batch_size, num_nodes, -1)
        x, _ = self.lstm(x)

        # Reshape for second GCNConv
        x = x.contiguous().view(batch_size * num_nodes, -1)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Return shape [batch_size, num_nodes, out_channels]
        return F.log_softmax(x, dim=1).view(batch_size, num_nodes, -1)
