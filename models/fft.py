import torch
import torch.nn as nn
import torch.fft as fft

class ChebConv(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(ChebConv, self).__init__()
        self.K = K
        self.weights = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_features, out_features)) for _ in range(K)])
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.bias)

    def forward(self, X, L):
        batch_size, num_nodes, seq_len, in_features =  X.shape
        X0 = X.permute(1, 2, 3, 0).contiguous()  # (num_nodes, in_features, seq_len, batch_size)
        X0 = X0.view(num_nodes, in_features * seq_len * batch_size)  # (num_nodes, in_features * seq_len * batch_size)
        out = torch.mm(L, X0)
        out = out.view(num_nodes, in_features, seq_len, batch_size)
        out = out.permute(1, 0, 2, 3).contiguous()  # (in_features, num_nodes, seq_len, batch_size)
        out = out.view(in_features, num_nodes* seq_len* batch_size)
        outputs = torch.zeros_like(X)
        outputs += (self.weights[0] @ out).view(in_features, num_nodes, seq_len, batch_size).permute(3,1,2,0)
        if self.K > 1:
            X1 = torch.mm(L, X0)
            X1 = X1.view(num_nodes, in_features, seq_len, batch_size)
            X1 = X1.permute(1, 0, 2, 3).contiguous() # (in_features, num_nodes, seq_len, batch_size)
            X1 = X1.view(in_features, num_nodes* seq_len* batch_size)
            outputs += (self.weights[1] @ X1).view(in_features, num_nodes, seq_len, batch_size).permute(3,1,2,0)
            X1 = X1.view(in_features, num_nodes, seq_len, batch_size).permute(1, 0, 2, 3).contiguous().view(num_nodes,in_features*seq_len*batch_size)#(70,1433*10*20)
            for k in range(2, self.K):

                X2 = (2 * torch.mm(L, X1) - X0)#(70,all)
                outputs += ((self.weights[k] @ X2.view(num_nodes,in_features,seq_len,batch_size).permute(1,0,2,3).contiguous().view(in_features,num_nodes* seq_len* batch_size))
                            .view(in_features, num_nodes, seq_len, batch_size).permute(3,1,2,0))
                X0, X1 = X1, X2

        return outputs + self.bias

class STCGCF(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(STCGCF, self).__init__()
        self.chebconv = ChebConv(in_features, out_features, K)

    def forward(self, X, L):
        return self.chebconv(X, L)

class FFTConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(FFTConv, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1)

    def forward(self, X):
        # X: (batch_size, num_nodes, in_features, seq_len)
        batch_size, num_nodes, seq_len, in_features = X.shape
        X_fft = fft.fft(X, dim=2)  # FFT on the last dimension (seq_len)
        X_fft = X_fft.permute(0,3,1,2).contiguous().view(batch_size,in_features,num_nodes*seq_len)
  # (batch_size, in_features, num_nodes, seq_len)
        X_fft_real = X_fft.real  # Extract real part
        X_fft_imag = X_fft.imag  # Extract imaginary part

        conv_real = self.conv(X_fft_real)
        conv_imag = self.conv(X_fft_imag)

        X_fft_out = torch.complex(conv_real, conv_imag)
        X_fft_out = X_fft_out.view(batch_size,in_features,num_nodes,seq_len).permute(0,2,3,1).contiguous()
# Combine real and imaginary parts
        X_out = fft.ifft(X_fft_out, dim=2)  # IFFT on the last dimension

        return X_out.real

if __name__ == '__main__':
    # 示例使用
    # 定义图的拉普拉斯矩阵L，输入特征矩阵X，卷积核Wf等
    L = ...  # 图的拉普拉斯矩阵
    X = ...  # 输入特征矩阵 (batch_size, num_nodes, in_features, seq_len)
    in_features = X.shape[2]
    out_features = ...  # 输出特征数量
    K = 3  # 切比雪夫多项式阶数

    stcgcf = STCGCF(in_features, out_features, K)
    fftconv = FFTConv(out_features, out_features)

    # 计算切比雪夫图卷积
    H = stcgcf(X, L)

    # 进行快速傅里叶变换（FFT）和卷积操作
    H_f = fft.fft(H, dim=-1)
    H_f_conv = fftconv(H_f)
    H_out = fft.ifft(H_f_conv, dim=-1).real

    print(H_out.shape)
