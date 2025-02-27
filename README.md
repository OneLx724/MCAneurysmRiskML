# MCAneurysmRiskML
Using Machine Learning Models to Screen and Predict the Risk of Poor Prognosis From Middle Cerebral Artery Aneurysms
# 代码说明：Using Machine Learning Models to Screen and Predict the Risk of Poor Prognosis From Middle Cerebral Artery Aneurysms

## 摘要
本文旨在使用机器学习模型来筛查和预测中脑动脉瘤（MCAA）患者的不良预后风险。通过数据预处理、特征选择和模型训练，本研究旨在提高对MCAA患者风险的准确预测能力。

## 代码结构
1. **数据预处理**
2. **特征选择**
3. **模型训练与评估**

## 数据预处理

### 缺失值处理
在数据预处理阶段，我们首先处理了缺失值。对于每列中的缺失值，我们计算该列的均值，并用均值替换缺失值。

```python
import numpy as np

mean_values = []
for i in range(X_train.shape[1]):
    mean_value = np.nanmean(X_train[:, i])
    mean_values.append(mean_value)
    X_train[np.isnan(X_train[:, i]), i] = mean_value
    X_test[np.isnan(X_test[:, i]), i] = mean_value
```

### 数据归一化
接着，我们对数据进行了归一化处理，确保各特征的尺度一致。

```python
from sklearn.preprocessing import StandardScaler

for i in range(X_train.shape[1]):
    scaler = StandardScaler()
    X_train[:, i] = scaler.fit_transform(X_train[:, i].reshape(-1, 1)).ravel()
    X_test[:, i] = scaler.transform(X_test[:, i].reshape(-1, 1)).ravel()
```

### 数据平衡
为了提高模型的泛化能力，我们对数据进行了平衡处理。

```python
from imblearn.over_sampling import SMOTE

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
```

## 特征选择
我们使用特征重要性分析来选择对预测不良预后最有影响力的特征。

```python
feature_importance = feature_importance_combination(X_train, y_train)
X_train = X_train[:, feature_importance['top_10_features']]
X_test = X_test[:, feature_importance['top_10_features']]
```

## 模型训练与评估
我们选择了多种机器学习模型，包括逻辑回归、KNN、SVM、决策树、随机森林和XGBoost，并对每个模型进行了交叉验证和性能评估。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

models = [
    ('Logistic', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(probability=True)),
    ('DT', DecisionTreeClassifier()),
    ('RF', RandomForestClassifier()),
    ('XGB', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

auc_values = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    auc_values.append(roc_auc)
```

## 结果展示
我们将每个模型的AUC值绘制在一个柱状图中，以便直观比较。

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)

ax.bar(['Logistic', 'KNN', 'SVM', 'DT', 'RF', 'XGB'], auc_values, width=0.5,
       color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])

ax.set_title('AUC of Different Models')
ax.set_xlabel('Models')
ax.set_ylabel('AUC')

ax.set_axisbelow(True)
ax.spines[['right', 'top']].set_color('C7')

for i, j in zip(['Logistic', 'KNN', 'SVM', 'DT', 'RF', 'XGB'], auc_values):
    ax.text(i, j, str(j), ha='center', va='bottom')

plt.savefig('Accuracy.png')
```

## 总结
本代码展示了如何使用机器学习模型来筛查和预测中脑动脉瘤患者的不良预后风险。通过数据预处理、特征选择和模型训练，我们实现了对MCAA患者风险的准确预测。

## 附件
源码见ML_CODE.py。

感谢您的阅读！希望这篇代码说明对您有所帮助。
