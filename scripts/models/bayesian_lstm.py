import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BayesianLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

        self.sigma_min = 1e-3
        self._init_weights()   # ← 关键修正

    def _init_weights(self):
        """
        修正根因：sigma_head最后一层偏置初始化为负值
        使softplus(raw)初始输出≈log(1+e^{-2})≈0.13，
        而不是默认的≈0.69，避免σ一开始就过大
        """
        # mu_head：最后线性层输出对齐RUL量级（0~125）
        # 用小的正偏置让初始mu预测在中间值附近
        nn.init.xavier_uniform_(self.mu_head[-1].weight)
        nn.init.constant_(self.mu_head[-1].bias, 60.0)  # RUL中值

        # sigma_head：偏置初始化为-2，使初始σ≈softplus(-2)+1e-3≈0.127
        nn.init.xavier_uniform_(self.sigma_head[-1].weight)
        nn.init.constant_(self.sigma_head[-1].bias, -2.0)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        mu    = self.mu_head(h).squeeze(-1)
        raw   = self.sigma_head(h).squeeze(-1)
        sigma = F.softplus(raw) + self.sigma_min
        return mu, sigma

    def predict_mc(self, x, n_samples=50):
        self.train()
        mus, sigmas = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                mu, sigma = self(x)
                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
        mu_mean = np.mean(mus, axis=0)
        sigma_sq_mean = np.mean(np.square(sigmas), axis=0)  # mean(sigma_b^2), NOT mean(sigma_b)^2
        sigma_total = np.sqrt(sigma_sq_mean + np.var(mus, axis=0))
        return mu_mean, sigma_total

def gaussian_nll_loss(mu, sigma, y):
    """
    数值稳定版NLL：先clamp sigma避免log(0)
    """
    sigma = torch.clamp(sigma, min=1e-3)
    loss  = 0.5 * (torch.log(sigma**2) +
                   (y - mu)**2 / sigma**2)
    return loss.mean()
