# 修正后的完整代码（已解决引号问题并优化可视化）
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

# === 数据生成阶段 ===
np.random.seed(42)  # 设置随机种子保证可重复性
n_samples = 200  # 样本量参数化

# 生成两个独立特征
features = np.random.uniform(-3, 3, size=(n_samples, 2))  # 合并生成
x1, x2 = features[:, 0], features[:, 1]  # 解包获取特征列

# 构造带噪声的非线性目标变量
y = np.sin(x1) + 0.5 * np.cos(x2) + np.random.normal(0, 0.2, n_samples)

# === 模型构建阶段 ===
# 创建GAM模型（两个平滑项）
gam_model = LinearGAM(s(0) + s(1))  # s(0)表示对第一个特征使用样条变换

# 模型训练
gam_model.fit(features, y)  # 注意直接使用features矩阵

# === 结果可视化阶段 ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

# 遍历每个特征绘制部分依赖图
for idx, ax in enumerate(axes):
    # 生成该特征的网格数据
    grid_data = gam_model.generate_X_grid(term=idx)  # 自动生成合适范围

    # 计算部分依赖效应
    pd_effect = gam_model.partial_dependence(term=idx, X=grid_data)

    # 绘制曲线
    ax.plot(grid_data[:, idx], pd_effect,
            color='royalblue',
            linewidth=2.5)

    # 样式设置
    ax.set_title(f'Feature x{idx + 1} Partial Dependency', fontsize=12)
    ax.set_xlabel(f'x{idx + 1} Value', fontsize=10)
    ax.set_ylabel('Contribution to Prediction', fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
