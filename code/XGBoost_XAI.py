import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#数据预处理
def preprocess_data(df):
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])
    X = df.drop(columns=["Class"])
    y = df["Class"]
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test,X_test.columns

#使用XGBoost训练模型
def train_model(X_train, y_train):
    scale_pos_weight = 10
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

#SHAP 可解释性分析函数
def shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    #全局可解释性分析
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    shap.summary_plot(shap_values, X_test)
    plt.show()

#LIME 可解释性分析函数
def lime_analysis(model, X_train,X_test, y_train,class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        training_labels=y_train.values,
        class_names = class_names,
        mode="classification",
        feature_names=X_train.columns,
    )

    #选择一个测试样本进行局部解释
    idx= 10#选择第10个样本
    sample = X_test.iloc[idx].values.reshape(1, -1)
    exp = explainer.explain_instance(sample[0], model.predict_proba,num_features=10)
    print(idx,"样本真实数据",X_test.iloc[idx])
    #LIME局部可解释性分析
    exp.show_in_notebook(show_table=True, show_all=False)
    # **解决方案 2**：如果在 PyCharm 里，使用 `as_list()` 打印结果
    print("LIME 解释结果：")
    for feature, importance in exp.as_list():
        print(f"{feature}: {importance}")
    exp.save_to_file("lime_explanation.html")
    print("LIME 解释已保存到 `lime_explanation.html`，请手动打开查看。")
#主函数
def main():
    #加载数据
    df = pd.read_csv("../dataset/creditcard.csv")
    #数据预处理
    X_train, X_test, y_train, y_test, class_names = preprocess_data(df)
    #训练模型
    model = train_model(X_train, y_train)
    #SHAP分析
    shap_analysis(model, X_test)
    #LIME分析
    lime_analysis(model, X_train, X_test, y_train, class_names=["Non-Fraud", "Fraud"])

if __name__ == "__main__":
    main()

