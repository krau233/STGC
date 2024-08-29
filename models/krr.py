import torch
import torch.nn as nn

class KernelRidgeRegression(nn.Module):
    def __init__(self, kernel, ridge):
        super(KernelRidgeRegression, self).__init__()
        self.kernel   = kernel
        self.ridge    = ridge
        
    def forward(self, G_t, G_s, y_t, y_s, E_t, E_s):
        K_ss      = self.kernel(G_s, G_s, E_s, E_s)
        K_ts      = self.kernel(G_t, G_s, E_t, E_s)
        n        = torch.tensor(len(G_s), device = G_s.device)
        regulizer = self.ridge * torch.trace(K_ss) * torch.eye(n, device = G_s.device) / n
        x       = K_ts @ torch.inverse(K_ss+regulizer*torch.eye(K_ss.shape[0],device= G_s.device))@y_s
        return x