# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm
import time
import os

sns.set_theme()

font = {'family' : 'Microsoft Yahei',
              'size'   : 10}
matplotlib.rc('font', **font)

label_fontsize = 12
title_fontsize = 14

plt.rcParams['axes.edgecolor'] = 'black'

# %%
h_path = "附件3、土壤湿度2022—2012年.xls"
e_path = "附件4、土壤蒸发量2012—2022年.xls"
NVDI_path = "附件6、植被指数-NDVI2012-2022年.xls"
R_path = "附件9、径流量2012-2022年.xlsx"
LAI_path = "附件10、叶面积指数（LAI）2012-2022年.xls"

# %%
paths = [h_path, e_path, NVDI_path, R_path, LAI_path]
save_path = ["湿度.xls", "蒸发量.xls", "NDVI.xls", "径流量.xls", "LAI.xls"]
for j in range(len(paths)):
    df = pd.read_excel(paths[j])
    df_trans = pd.DataFrame()
    if j==4:
        df_trans = pd.read_excel(paths[j]).sort_values(by="日期")
    else:
        for i in df.groupby(by="年份"):
            df_trans = pd.concat((df_trans, i[1].drop(columns=["经度(lon)", "纬度(lat)"])), axis=0)
    df_trans = df_trans.reset_index(drop=True)
    df_trans.to_excel(save_path[j])

# %%
humidity = pd.read_excel(h_path)
evaporation = pd.read_excel(e_path)

# %%
def plot(df, cols, xticks, xlabel, ylabel):
    plt.figure(figsize=(12, 8))
    for i in cols:
        plt.plot(range(df.shape[0]), df[i], label=i)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(*xticks)
    plt.show()

def plot_humidity(df, xlabel="年份", ylabel="湿度(kg/m2)", xticks=[[], []]):
    cols = ["10cm湿度(kg/m2)", "40cm湿度(kg/m2)", "100cm湿度(kg/m2)", "200cm湿度(kg/m2)"]
    plot(df, cols, xticks, xlabel, ylabel)

def plot_evaporation(df, xlabel="年份", ylabel="土壤蒸发量(mm)", xticks=[[], []]):
    cols = ["土壤蒸发量(mm)"]
    plot(df, cols, xticks, xlabel, ylabel)

def plot_rain(df, xlabel="年份", ylabel="降水量(mm)", xticks=[[], []]):
    cols = ["降水量(mm)"]
    plot(df, cols, xticks, xlabel, ylabel)

# %% [markdown]
# 湿度  

# %%
h_trans = pd.DataFrame()
for i in humidity.groupby(by="年份"):
    h_trans = pd.concat((h_trans, i[1].drop(columns=["经度(lon)", "纬度(lat)"])), axis=0)
h_trans = h_trans.reset_index(drop=True)
h_trans

# %%
plot_humidity(h_trans, xticks=[range(0, 123, 12), range(2012, 2023)])

# %% [markdown]
# 蒸发量

# %%
e_trans = pd.DataFrame()
for i in evaporation.groupby(by="年份"):
    e_trans = pd.concat((e_trans, i[1].drop(columns=["经度(lon)", "纬度(lat)"])), axis=0)
e_trans = e_trans.reset_index(drop=True)

# %%
plot_evaporation(e_trans, xticks=[range(0, 123, 12), range(2012, 2023)])

# %% [markdown]
# 降水量

# %%
dir_path = "附件8、锡林郭勒盟气候2012-2022"
select_cols = ["年份", "月份", "降水量(mm)", "平均风速(knots)", "平均气温(℃)"]
_, _, files = list(*os.walk(dir_path))
rain = pd.DataFrame()
for file in files:
    tmp = pd.read_excel("/".join([dir_path, file]))[select_cols]
    rain = pd.concat([rain, tmp], axis=0)
rain = rain.reset_index(drop=True)
# rain.to_excel("降水量.xls")

# %%
plot_rain(rain, xticks=[range(0, 123, 12), range(2012, 2023)])

# %%
h_trans_diff = pd.concat([h_trans[["月份", "年份"]], h_trans.drop(columns=["月份", "年份"]).diff(axis=0)], axis=1).dropna(axis=0)
plot_humidity(h_trans_diff, xticks=[range(0, 123, 12), range(2012, 2023)])


