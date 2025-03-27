import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 生成可分数据集
X, y = datasets.make_classification(n_samples=100, n_features=2,
                                    n_informative=2, n_redundant=0,
                                    n_classes=2, random_state=42)

# 训练线性SVM
clf = SVC(kernel='linear')
clf.fit(X, y)

# 获取模型参数
w = clf.coef_[0]
b = clf.intercept_[0]

# 计算决策边界
x_points = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
y_decision = -(w[0]/w[1])*x_points - b/w[1]

# 计算间隔边界
margin = 1/np.linalg.norm(w)
y_upper = y_decision + margin
y_lower = y_decision - margin

# 可视化设置
plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', label='Data')
plt.plot(x_points, y_decision, 'r-', label='Decision Boundary')
plt.plot(x_points, y_upper, 'k--', alpha=0.5, label='Margin')
plt.plot(x_points, y_lower, 'k--', alpha=0.5)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='k', linewidths=1.5,
            label='Support Vectors')

# 坐标轴设置
plt.xlim(X[:, 0].min()-0.5, X[:, 0].max()+0.5)
plt.ylim(X[:, 1].min()-0.5, X[:, 1].max()+0.5)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('SVM Decision Boundary with Margins', fontsize=14)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()
