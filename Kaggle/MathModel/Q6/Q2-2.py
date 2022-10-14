# %%
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set_theme()

font = {'family' : 'Microsoft Yahei',
              'size'   : 10}
matplotlib.rc('font', **font)

plt.rcParams['axes.edgecolor'] = 'black'

# %%
data_ = pd.read_excel("dataset.xlsx")
index = []
for row in range(122):
    for j in range(4):
        index.append(row + 122*j)
data = data_.loc[index].reset_index(drop=True)
month_flag = data["月份"]
data.drop(columns=["年份", "月份"], inplace=True)
data

# %%
## 标准化
data_mean, data_std = data.mean(axis=0), data.std(axis=0)
data = (data - data_mean) / data_std
data

# %% [markdown]
# 特征相关性

# %%
plt.figure(figsize=(20, 8))
sns.heatmap(data.drop(columns=["深度(cm)", "当月湿度(mm)", "标签(mm)"]).corr(), annot=True, square=True, fmt=".2f", cmap='YlGnBu')

# %% [markdown]
# 训练

# %%
def transform(y):
    return y * data_std[-1] + data_mean[-1]


def cal_metric(label, pred):
    label = transform(label)
    pred = transform(pred)
    print(f"mae: {mean_absolute_error(label, pred)}")
    print(f"R2: {r2_score(label, pred)}")


def mae_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    mae = mean_absolute_error(transform(y), transform(y_pred))
    return mae

def r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    r2 = r2_score(transform(y), transform(y_pred))
    return r2

# %%
feature, label = data.drop(columns=["标签(mm)"]), data["标签(mm)"]
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, shuffle=False)

# %%
mlp = MLPRegressor((32, 16,), batch_size=16, learning_rate_init=0.002, 
                    learning_rate="adaptive", max_iter=1500, random_state=518,
                    verbose=False, tol=1e-10, n_iter_no_change=100)

# %%
# 五折交叉验证
mae_cv = cross_val_score(mlp, feature, label, cv=5, scoring=mae_scorer, n_jobs=-1)
print(mae_cv)
print(mae_cv.mean())

r2_cv = cross_val_score(mlp, feature, label, cv=5, scoring=r2_scorer, n_jobs=-1)
print(r2_cv)
print(r2_cv.mean())

# %%
model = mlp.fit(X_train, y_train)

# %%
cal_metric(y_test, model.predict(X_test))

# %% [markdown]
# 损失曲线, 误差曲线, R2曲线

# %%
fig, axes = plt.subplots(1, 3, figsize=(30, 8))
## 损失曲线
axes[0].plot(model.loss_curve_, label="误差")
axes[0].set_ylim([0, 0.006])
axes[0].set_xlim([0, 910])
axes[0].set_xlabel("迭代次数")
axes[0].set_ylabel("训练误差")
axes[0].set_title("训练误差随迭代次数变化")
axes[0].legend()

## 误差曲线
axes[1].plot((transform(y_train) - transform(model.predict(X_train))) / transform(y_train)*100, label="训练")
axes[1].plot((transform(y_test) - transform(model.predict(X_test))) / transform(y_test)*100, label="测试")
axes[1].set_xlabel("年份")
axes[1].set_ylabel("相对误差(%)")
axes[1].set_ylim([-30, 30])
axes[1].set_xlim([0, 488])
axes[1].set_title("模型在训练和测试集上的相对误差")
axes[1].set_xticks(ticks=range(0, 488, 48))
axes[1].set_xticklabels(labels=range(2012, 2023))
axes[1].legend()

## R2曲线
axes[2].plot(transform(model.predict(X_test)), transform(y_test), "o")
axes[2].set_xlabel("预测湿度")
axes[2].set_ylabel("真实湿度")
axes[2].plot([0, 170], [0, 170])
axes[2].set_title(f"R2 = {r2_score(transform(y_test), transform(model.predict(X_test))): .4f}")

# %% [markdown]
# 预测曲线

# %%
year = [2020]*10+[2021]*12+[2022]*3
month = list(range(3, 13)) + list(range(1, 13)) + list(range(1, 4))
xticks = [str(year[i])+'/'+str(month[i]) for i in range(len(year))]

# %%
y_test_t, y_pred_t = transform(y_test), transform(model.predict(X_test))

def plot_predict(i, title, ax, ylim=None):
    h_true, h_pred = y_test_t[i::4], y_pred_t[i::4]
    ax.plot(range(len(h_true)), h_true, label="真实湿度")
    ax.plot(range(len(h_pred)), h_pred, label="预测湿度")
    ax.set_xlabel("月份")
    ax.set_ylabel("湿度")
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xticks(range(len(h_true)))
    ax.set_xticklabels(xticks[-len(h_true):], rotation=-45)
    ax.legend()

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5)
# 预测是从2020年2月开始的, 就是指的2020年3月的湿度
## 10cm湿度预测
plot_predict(0, "100cm湿度预测", axes[0][0], [30, 120])
## 40cm湿度预测
plot_predict(1, "200cm湿度预测", axes[0][1], [150, 180])
## 100cm湿度预测
plot_predict(2, "10cm湿度预测", axes[1][0], [0, 50])
## 200cm湿度预测
plot_predict(3, "40cm湿度预测", axes[1][1], [20, 100])


