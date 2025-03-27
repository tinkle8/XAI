# impactx_fraud_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ========== 模型定义 ==========

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, x):
        return self.extractor(x)


class LatentExplanationPredictor(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)


class ExplanationDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_dim=30):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)


class FraudClassifier(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, m, z):
        x = torch.cat([m, z], dim=1)
        return self.classifier(x)


# ========== 数据集定义 ==========

class FraudDataset(Dataset):
    def __init__(self, X, y, r):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.r = torch.tensor(r, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.r[idx]


# ========== 损失函数定义 ==========

def compute_loss(y_true, y_pred, r_true, r_pred, lambda_mse=1.0):
    bce = nn.BCELoss()(y_pred, y_true)
    mse = nn.MSELoss()(r_pred, r_true)
    return bce + lambda_mse * mse


# ========== 主训练循环 ==========

def train_impactx(X_train, y_train, r_train, input_dim=30, hidden_dim=64,
                  latent_dim=32, batch_size=128, num_epochs=20, lr=1e-3, lambda_mse=1.0):
    dataset = FraudDataset(X_train, y_train, r_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    M = FeatureExtractor(input_dim, hidden_dim)
    LEP = LatentExplanationPredictor(input_dim, latent_dim)
    D = ExplanationDecoder(latent_dim, input_dim)
    C = FraudClassifier(hidden_dim, latent_dim)

    optimizer = optim.Adam(list(LEP.parameters()) + list(D.parameters()) + list(C.parameters()), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, y_batch, r_batch in dataloader:
            with torch.no_grad():
                m = M(x_batch)
            z = LEP(x_batch)
            y_hat = C(m, z)
            r_hat = D(z)
            loss = compute_loss(y_batch, y_hat, r_batch, r_hat, lambda_mse)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(dataloader):.4f}")

    return M, LEP, D, C