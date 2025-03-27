# impactx_fraud_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# =======================
# 1. 模型模块定义 （M, LEP, D, C）
# =======================
"""
✅ 为何需要它？
在原论文中，M(x) 是负责提取特征的主模型（如 CNN）
	•在本项目中，主模型是 XGBoost，我们训练好了，但我们不能直接将它用于 end-to-end 的 PyTorch 训练（什么是end-to-end？）
	•所以我们用一个 “形状匹配的 MLP” 结构来模拟 M(x)，仅用于第二阶段中生成冻结特征（不更新。
✅ 为什么设置 hidden_dim=64？
	•	给出一个中等容量的表示向量，作为输入特征（m）
	•	也方便和 C 模块拼接（m || z）
"""
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

#LEP 模块
"""
✅ 作用：
	•输入原始交易向量 x，输出解释编码向量 z
	•它是一个自动学习“与解释相关的内部向量 z”的编码器
	•后续会被：
	•解码器 D 用来还原出解释图 r̂
	•分类器 C 结合特征 m 一起做分类
✅ 为什么使用 Sigmoid？
	•把 z 控制在 0~1 的范围，便于网络稳定训练（如 attention 表示）
"""
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
"""
✅ 作用：
	•输入解释编码 z
	•输出长度为 30 的解释图（r̂）：每个维度对应一个特征（V1~V28, Time, Amount）
✅ 为什么使用 MSE 作为监督？
	•我们的真实解释图来自 SHAP，是连续值
	•所以用 MSE 来让 r̂ 逼近 r（SHAP 值）
"""
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
        combined = torch.cat([m, z], dim=1)
        return self.classifier(combined)


# =======================
# 2. 自定义数据集类
# =======================

class FraudDataset(Dataset):
    def __init__(self, X, y, r):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.r = torch.tensor(r, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.r[idx]


# =======================
# 3. 联合损失函数
# =======================

def compute_loss(y_true, y_pred, r_true, r_pred, lambda_mse=1.0):
    bce_loss = nn.BCELoss()(y_pred, y_true)
    mse_loss = nn.MSELoss()(r_pred, r_true)
    return bce_loss + lambda_mse * mse_loss


# =======================
# 4. 主训练函数（阶段二）
# =======================

def train_impactx(X_train, y_train, r_train, input_dim=30, hidden_dim=64, latent_dim=32,
                  batch_size=512, num_epochs=20, lr=1e-3, lambda_mse=1.0):
    dataset = FraudDataset(X_train, y_train, r_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型组件
    M = FeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim)  # 用于提特征但不参与训练
    LEP = LatentExplanationPredictor(input_dim=input_dim, latent_dim=latent_dim)
    D = ExplanationDecoder(latent_dim=latent_dim, output_dim=input_dim)
    C = FraudClassifier(feature_dim=hidden_dim, latent_dim=latent_dim)

    # 优化器只优化 LEP、D、C
    optimizer = optim.Adam(list(LEP.parameters()) + list(D.parameters()) + list(C.parameters()), lr=lr)

    for epoch in range(num_epochs):
        M.eval()  # M 仅作为冻结特征提取器
        total_loss = 0.0

        for x_batch, y_batch, r_batch in dataloader:
            with torch.no_grad():
                m = M(x_batch)  # 冻结特征提取器

            z = LEP(x_batch)
            y_hat = C(m, z)
            r_hat = D(z)

            loss = compute_loss(y_batch, y_hat, r_batch, r_hat, lambda_mse)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {total_loss / len(dataloader):.4f}")

    return M, LEP, D, C