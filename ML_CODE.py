import os
import copy
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, LogisticRegressionCV, enet_path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.calibration import CalibratedClassifierCV
import shap


def feature_importance_combination(X, y):
    # 数据分割和标准化
    # 由于 Lasso 需要标准化，但随机森林不需要，这里先对数据进行标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 网格搜索最佳 alpha 值（正则化强度）
    lasso = Lasso(random_state=42)
    param_lasso = {'alpha': np.logspace(-5, 5, 100)}
    grid_lasso = GridSearchCV(lasso, param_lasso, cv=5)
    grid_lasso.fit(X_scaled, y)

    best_alpha = grid_lasso.best_params_['alpha']
    lasso_best = Lasso(alpha=best_alpha, random_state=42)
    lasso_best.fit(X_scaled, y)

    # 提取 Lasso 的特征重要性（系数绝对值）
    lasso_importance = np.abs(lasso_best.coef_)

    # 随机森林模型
    if isinstance(y, pd.Series) and len(np.unique(y)) > 2:
        # 回归任务
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        # 分类任务
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X_scaled, y)
    rf_importance = rf.feature_importances_

    # 综合两种模型的重要性评分（标准化后加权平均）
    # 将 Lasso 和随机森林的重要性评分标准化

    lasso_scores = pd.Series(lasso_importance, index=X.columns)
    rf_scores = pd.Series(rf_importance, index=X.columns)

    # 标准化两种模型的评分（使它们在同一尺度）
    scaler_lasso = StandardScaler()
    scaler_rf = StandardScaler()

    lasso_standardized = scaler_lasso.fit_transform(lasso_scores.values.reshape(-1, 1)).flatten()
    rf_standardized = scaler_rf.fit_transform(rf_scores.values.reshape(-1, 1)).flatten()

    # 综合评分：取两者的平均值（你可以根据需要调整权重）
    combined_importance = (lasso_standardized + rf_standardized) / 2

    # 创建特征重要性 DataFrame
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'combined_score': combined_importance,
        'lasso_score': lasso_scores.values,
        'rf_score': rf_scores.values
    })

    # 按综合评分排序，选出前 10 个特征
    feature_importance = feature_importance.sort_values(by='combined_score', ascending=False)
    top_10_features = feature_importance.head(10)['feature'].tolist()

    return {
        'top_10_features': top_10_features,
        'feature_importance': feature_importance
    }

def cal_performance(y_test, y_pred, roc_auc):
    cf_matrix = confusion_matrix(y_test, y_pred)

    D = cf_matrix[0, 0]
    C = cf_matrix[1, 0]
    A = cf_matrix[1, 1]
    B = cf_matrix[0, 1]

    Precision = A / (A + B)
    Recall = A / (A + C)
    f1_score = 2 * (Precision * Recall) / (Precision + Recall)
    sensitivity = A / (A + C)
    specificity = D / (B + D)
    prevalence = (A + C) / (A + B + C + D)
    PPV = (sensitivity * prevalence) / ((sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence)))
    NPV = (specificity * (1 - prevalence)) / (((1 - sensitivity) * prevalence) + ((specificity) * (1 - prevalence)))
    DetectionPrevalence = (A + B) / (A + B + C + D)
    BalancedAccuracy = (sensitivity + specificity) / 2
    Accuracy = (A + D) / (A + B + C + D)

    print('roc_auc:', roc_auc)
    print('Precision:', Precision)
    print('Recall:', Recall)
    print('f1_score:', f1_score)
    print('sensitivity:', sensitivity)
    print('specificity:', specificity)
    print('prevalence:', prevalence)
    print('PPV:', PPV)
    print('NPV:', NPV)
    print('DetectionPrevalence:', DetectionPrevalence)
    print('BalancedAccuracy:', BalancedAccuracy)
    print('Accuracy:', Accuracy)
    return Accuracy


# 1、读取Excel文件
data = pd.read_excel('动脉瘤机器学习.xls')
X_train = data.iloc[2:84, 3:36].values  # 选择特定的行和列
X_name = data.iloc[2:84, 3:36]
y_train = data.iloc[2:84, 38].values  # 目标列，根据需要修改

data = pd.read_excel('动脉瘤机器学习测试.xls')
X_test = data.iloc[2:22, 3:36].values
X_test_name = data.iloc[2:22, 3:36]
y_test = data.iloc[2:22, 38].values

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# 2、异常值处理
mean_values = []
for i in range(X_train.shape[1]):
    mean_value = np.nanmean(X_train[:, i])
    mean_values.append(mean_value)
    X_train[np.isnan(X_train[:, i]), i] = mean_value
    X_test[np.isnan(X_test[:, i]), i] = mean_value

# 3、数据归一化
for i in range(X_train.shape[1]):
    scaler = StandardScaler()
    X_train[:, i] = scaler.fit_transform(X_train[:, i].reshape(-1, 1)).ravel()
    X_test[:, i] = scaler.transform(X_test[:, i].reshape(-1, 1)).ravel()

# 4、数据平衡
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# 5、特征重要性分析
feature_importance = feature_importance_combination(X_train, y_train)
X_train = X_train[:, feature_importance['top_10_features']]
X_test = X_test[:, feature_importance['top_10_features']]

# 6、构造分类器（已完成参数调试）
classifiers = [LogisticRegressionCV(),
               KNeighborsClassifier(30),
               SklearnClassifier(SVC(kernel='linear',probability=True)),
               DecisionTreeClassifier(criterion="entropy", splitter="random", max_leaf_nodes=5), RandomForestClassifier(n_estimators=3, max_depth=5, max_leaf_nodes=5),
               xgb.XGBClassifier(n_estimators=2, max_depth=4, min_child_weight=1, subsample=0.6, colsample_bytree=0.6),
               MLPClassifier(hidden_layer_sizes=(32, 4), max_iter=500, early_stopping=True)]

# 7、分类器性能测试和统计
draw_shap = None
auc_values = []
classes = ['Logistic', 'KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP']
for i in range(7):
    if i == 2:
        model = CalibratedClassifierCV(LinearSVC(penalty='l2', max_iter=5000))
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        model = classifiers[i]
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    best_threshold = thresholds[np.argmax(tpr - fpr)]
    best_tpr = tpr[np.argmax(tpr - fpr)]
    best_fpr = fpr[np.argmax(tpr - fpr)]
    # 计算AUC-ROC的值
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    lw = 2
    colors = ['b','g','r','c','m','y','k','w']
    plt.plot(fpr, tpr, color=colors[i], lw=lw, label=classes[i] + ' AUC: %0.2f' % roc_auc)
    plt.scatter(best_fpr, best_tpr, color='black', s=20)


    y_pred = (y_score >= best_threshold)
    value = cal_performance(y_test, y_pred, roc_auc)
    auc_values.append(roc_auc)

    # shap值分析
    if draw_shap:
        svm_explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_train, 10))
        shap_values_numpy = svm_explainer.shap_values(X_train)
        # # SHAP computation for kNN
        # knn_explainer = shap.KernelExplainer(knn_clf.predict_proba, shap.kmeans(train_X, 50))
        # knn_shap_values = knn_explainer.shap_values(train_X.iloc[1:evalPoints, :].values)
        # # SHAP computation for MLP
        # explainer = shap.TreeExplainer(model)  # 计算shap值为numpy.array数组
        # shap_values_numpy = explainer.shap_values(X_train)

        plt.figure()
        shap.summary_plot(shap_values_numpy[1], X_train, feature_names=X_name.columns, plot_type="dot", show=False)
        plt.savefig("nonrupture_Logistic-shap_value.pdf", format='pdf', bbox_inches='tight')
        plt.figure(figsize=(10, 5), dpi=1200)
        shap.summary_plot(shap_values_numpy[1], X_name, plot_type="bar", show=False)
        plt.title('SHAP_numpy Sorted Feature Importance')
        plt.tight_layout()
        plt.savefig("nonrupture_Logistic-mean(shap_value).pdf", format='pdf', bbox_inches='tight')
        plt.close()

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('Train-ROC')
plt.legend(loc="lower right")
plt.savefig('train_roc' + '.png')
plt.close()

fig = plt.figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)

ax.bar(['Logistic', 'KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP'], auc_values, width=0.5,
       color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])

plt.ylim([0.0, 1.05])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Test-Accuracy')

ax.set_axisbelow(True)
ax.spines[['right', 'top']].set_color('C7')

for i, j in zip(['Logistic', 'KNN', 'SVM', 'DT', 'RF', 'XGB', 'MLP'], auc_values):
    ax.text(i, j, str(j), ha='center', va='bottom')

plt.savefig('Accuracy.png')
