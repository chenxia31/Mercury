# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm
import time

# %%
sns.set_theme()

# %%
font = {'family' : 'Microsoft Yahei',
              'size'   : 10}
matplotlib.rc('font', **font)

label_fontsize = 12
title_fontsize = 14

plt.rcParams['axes.edgecolor'] = 'black'

# %%
data_path = "内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）.xlsx"
data = pd.read_excel(data_path, usecols=range(0, 8))
data

# %%
data.groupby(by=["year", "放牧强度（intensity）"]).std()

# %%
data.groupby(by=["year", "放牧强度（intensity）"]).mean()

# %% [markdown]
# 考虑到：  
# STC土壤全碳 = SOC土壤有机碳 + SIC土壤无机碳  
# 土壤C/N比 = STC土壤全碳 / 全氮N  
# 仅预测土壤有机碳、无机碳、全氮

# %%
data = data.drop(columns=["STC土壤全碳", "土壤C/N比"])
data["放牧强度"] = data["放牧强度（intensity）"].map({"NG": 0, "LGI": 2, "MGI": 4, "HGI": 8})

# %% [markdown]
# #### 可视化
# - 查看12个放牧小区的SOC土壤有机碳、SIC土壤无机碳、全氮N的逐年变化

# %%
def plot(df, axes):
    index = df.index.tolist()
    xticks, legend = list(zip(*index))
    legend = legend[0]
    for i in range(3):
        ax = axes[i]
        ax.plot(xticks, df.iloc[:, i], 'o-', markersize=10, label=legend)
        ax.set_xticks(xticks)
        ax.set_xlabel("年份", fontsize=label_fontsize)
        ax.set_ylabel("含量", fontsize=label_fontsize)
        ax.set_title(df.columns[i])
        ax.legend()

# %%
fig, axes = plt.subplots(1, 3, figsize=(30, 8))
for i in data.groupby(by=["year", "放牧小区（plot）"]).agg("mean").groupby("放牧小区（plot）"):
    plot(i[1], axes)

# %% [markdown]
# - SOC土壤有机碳、全氮N、SIC土壤无机碳的相关性

# %%
tmp = data.groupby(by=["year", "放牧小区（plot）"]).agg("mean").drop(columns="放牧强度")
sns.pairplot(tmp, kind="reg")

# %%
def _heatmapplot(corr: pd.DataFrame, fig, ax, title: str) -> None:
    sns.heatmap(corr, annot=True, square=True, cmap='YlGnBu', ax=ax)
    fig.subplots_adjust(bottom=0.3, left=0.15)
    ax.set_title(title)


def heatmapplot(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    _heatmapplot(data.corr(), fig, axes[0], 'Pearson相关系数')
    _heatmapplot(data.corr('spearman'), fig, axes[1], 'Spearman相关系数')
    plt.show()

heatmapplot(tmp)

# %%
def bar_plot(df, col_name):
    x_group = [0.1, 0.35, 0.5, 0.75, 0.9]
    width = 0.2
    color = ['b', 'm', 'y', 'c']
    labels = ["NG", "LGI", "MGI", "HGI"]
    years = [2012, 2014, 2016, 2018, 2020]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    for i in range(len(x_group)):
        for j in range(4):
            ax.bar(i+width*j, df.loc[(years[i], labels[j]), col_name], width=width, color=color[j], align="center")
            ax.text(i+width*j-0.05, -0.02, labels[j])
            ax.text(i+width*j-0.1, df.loc[(years[i], labels[j]), col_name]+0.01, f"{df.loc[(years[i], labels[j]), col_name]: .2f}")
        ax.text(i+width, -0.04, years[i], fontsize=12)
    ax.set_xticks([])
    ax.set(
        title=col_name, 
        ylabel="标准差"
    )
    ax.text(2.2, -0.06, "年份", fontsize=label_fontsize)
    plt.show()

# %%
std = data.groupby(by=["year", "放牧强度（intensity）"]).agg("std")

# bar_plot(std, "SOC土壤有机碳")
# bar_plot(std, "SIC土壤无机碳")
bar_plot(std, "全氮N")

# %%
mean_v = data.groupby(by=["year", "放牧强度（intensity）"]).agg("mean")
# bar_plot(mean_v, 'SOC土壤有机碳')
# bar_plot(mean_v, "SIC土壤无机碳")
# bar_plot(mean_v, "全氮N")

# %% [markdown]
# #### 模型

# %%
model_data = data.groupby(by=["year", "放牧小区（plot）"]).mean()
years = [2012, 2014, 2016, 2018, 2020]
cols = ["SOC土壤有机碳", "SIC土壤无机碳", "全氮N"]
SOC_df, SIC_df, N_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
df_list = [SOC_df, SIC_df, N_df]
for i in range(len(cols)):
    df = df_list[i]
    for year in years:
        df[year] = model_data.loc[year][cols[i]]
    df["放牧强度"] = model_data.loc[year]["放牧强度"]

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for i in range(3):
    ax = axes[i]
    _heatmapplot(df_list[i].drop(columns="放牧强度").corr(), fig, ax, model_data.columns[i])

# %%
def get_feature_label(df, df1, df2):
    year = 2
    feature, label = pd.concat((df.iloc[:, 0:year], df1.iloc[:, 1], df2.iloc[:, 1], df.iloc[:, -1]), axis=1).values, df.iloc[:, 2].values
    for i in range(year+1, 5):
        tmp = pd.concat((df.iloc[:, i-year:i], df1.iloc[:, i-1], df2.iloc[:, i-1], df.iloc[:, -1]), axis=1)
        feature = np.vstack((feature, tmp))
        label = np.hstack((label, df.iloc[:, i].values))
    return feature, label

# %%
SOC_feature, SOC_label = get_feature_label(SOC_df, SIC_df, N_df)
SIC_feature, SIC_label = get_feature_label(SIC_df, SOC_df, N_df)
N_feature, N_label = get_feature_label(N_df, SOC_df, SIC_df)

# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# %%
class MyDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.feature = torch.tensor(X, dtype=torch.float)
        self.label = torch.tensor(y, dtype=torch.float)
    
    def __getitem__(self, index):
        return self.feature[index], self.label[index]
    
    def __len__(self):
        return self.feature.shape[0]

# %%
class Model(nn.Module):
    def __init__(self, type):
        super().__init__()
        assert type in ["SOC", "SIC", "N"]
        if type == "SOC":
            self.w = nn.Parameter(torch.tensor([1., 1., -1., 1., 1.]).unsqueeze(1))
        elif type == "SIC":
            self.w = nn.Parameter(torch.tensor([1., 1., -1., -1., 1.]).unsqueeze(1))
        else:
            self.w = nn.Parameter(torch.tensor([1., 1., 1., -1., 1.]).unsqueeze(1))
        self.sigma = nn.Parameter(torch.ones(12, 1))
        self.mu = nn.Parameter(torch.zeros(12, 1))
    
    def forward(self, X):
        tmp = (X[:, -1].unsqueeze(1)-self.mu)**2 / 2 / self.sigma**2
        delta = torch.exp(-tmp) / torch.sqrt(torch.tensor(2*np.pi, dtype=torch.float)) / self.sigma
        return (torch.hstack((X[:, :-1], delta)) @ self.w).squeeze()

# %%
def MSELoss(y_pred, y):
    '''考虑相对误差的mse
    '''
    return torch.mean(((y - y_pred) / y) ** 2) * 100

# %%
class MyModel:
    def __init__(self, dataset, type):
        self.type = type
        self.model = Model(type)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
    
    def train(self, epochs, lr=0.001):
        print(f"train {self.type}...")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.mse = []
        time.sleep(0.5)
        iter_ = tqdm(range(epochs))
        for epoch in iter_:
            mse = 0
            for i, (X, y) in enumerate(self.dataloader):
                y_pred = self.predict(X)
                loss = MSELoss(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mse += loss.item()
            mse /= (i+1)
            self.mse.append(mse)
            iter_.set_description(f"epoch: {epoch+1} mse: {mse: .4f}")
    
    def predict(self, X):
        return self.model(X)
    
    def plot_error(self, name, ylim):
        plt.figure(figsize=(12, 8))
        legend = [2016, 2018, 2020]
        with torch.no_grad():
            error = []
            for i, (X, y) in enumerate(self.dataloader):
                y_pred = self.predict(X)
                error_ = (y - y_pred) / y * 100
                error.append(error_)
                plt.plot(range(len(error_)*i, len(error_)*(i+1)), error_, 'o-', markersize=10, label=legend[i])
            error = torch.hstack(error)
            plt.plot(range(len(error)), error, '--')
            plt.legend()
            plt.xticks(ticks=range(36), labels=SOC_df.index.tolist()*3, rotation=-45)
            plt.plot([-1, 36], [0, 0], '--')
            plt.xlim([-1, 36])
            plt.ylim(ylim)
            plt.xlabel("放牧小区")
            plt.ylabel("相对误差(%)")
            plt.title(name)
        
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path))

# %%
SOC_dataset = MyDataset(SOC_feature, SOC_label)
SIC_dataset = MyDataset(SIC_feature, SIC_label)
N_dataset = MyDataset(N_feature, N_label)

SOC_model = MyModel(SOC_dataset, "SOC")
SIC_model = MyModel(SIC_dataset, "SIC")
N_model = MyModel(N_dataset, "N")

# %%
SOC_model.train(20000)

# %%
SIC_model.train(25000, lr=0.0005)

# %%
N_model.train(20000, lr=0.0001)
# 0.1721

# %%
SOC_model.plot_error("SOC土壤有机碳--预测误差曲线", [-25, 25])

# %%
SIC_model.plot_error("SIC土壤无机碳--预测误差曲线", [-30, 30])

# %%
N_model.plot_error("全氮N--预测误差曲线", [-25, 25])

# %%
soc_18_20 = pd.concat((SOC_df.iloc[:, -3:-1], SIC_df.iloc[:, -2], N_df.iloc[:, -2], SOC_df.iloc[:, -1]), axis=1).values
soc_18_20 = torch.tensor(soc_18_20, dtype=torch.float)
dict(zip(SOC_df.index, SOC_model.predict(soc_18_20).tolist()))

# %%
sic_18_20 = pd.concat((SIC_df.iloc[:, -3:-1], SOC_df.iloc[:, -2], N_df.iloc[:, -2], SIC_df.iloc[:, -1]), axis=1).values
sic_18_20 = torch.tensor(sic_18_20, dtype=torch.float)
dict(zip(SIC_df.index, SIC_model.predict(sic_18_20).tolist()))

# %%
n_18_20 = pd.concat((N_df.iloc[:, -3:-1], SOC_df.iloc[:, -2], SIC_df.iloc[:, -2], N_df.iloc[:, -1]), axis=1).values
n_18_20 = torch.tensor(n_18_20, dtype=torch.float)
dict(zip(N_df.index, N_model.predict(n_18_20).tolist()))


