import os
import shap
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report

# 禁用 PyCharm 的 datalore.display 远程绘图
os.environ["DATALORE_DISABLED"] = "1"
matplotlib.use('Agg')
# 1. 读取数据
df = pd.read_csv("../dataset/creditcard.csv")

# 2. 数据预处理（避免数据泄漏）
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# 3. 划分特征和标签
X = df.drop(columns=["Class"])
y = df["Class"]

# 4. 划分训练和测试集（SMOTE 应该在训练集上应用）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 处理数据不均衡（仅对训练集使用 SMOTE）
smote = SMOTE(sampling_strategy=0.4, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 6. 计算 scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 7. 训练 XGBoost 模型
model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # 适用于不均衡数据
    eval_metric="auc",
    use_label_encoder=False,
    n_estimators=200,  # 增加树的数量
    learning_rate=0.05,  # 降低学习率
    max_depth=6,
    subsample=0.8,  # 采样 80% 数据
    colsample_bytree=0.8,  # 选择 80% 特征
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
print("AUC Score:", auc)
print(classification_report(y_test, y_pred))

# 8. SHAP 解释（全局）
explainer = shap.TreeExplainer(model)  # 用 TreeExplainer 更快
shap_values = explainer.shap_values(X_test[:100])  # 只取前100个样本，加速计算，这返回的是numpy.ndarray


# ✅ SHAP 解释
shap.summary_plot(shap_values, X_test[:100])
explainer = shap.TreeExplainer(model)
#为了生成柱状图将shap_values变成shap.Explanation对象
shap_values = explainer(X_test)
shap_values = shap.Explanation(values=shap_values.values,
                               base_values=shap_values.base_values,
                               data=X_test,
                               feature_names=X_test.columns)
shap.plots.bar(shap_values, max_display=30)  # 生成柱状图
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
#plt.show()  # 显示 Matplotlib 图像
