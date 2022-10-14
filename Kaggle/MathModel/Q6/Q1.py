# %% [markdown]
# # 第一问
# 1. 建立放牧强度-土壤湿度模型
# 2. 修正放牧强度-植被生物量模型

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from itertools import groupby

sns.set_theme()
font = {'family' : 'Microsoft Yahei',
              'size'   : 10}
matplotlib.rc('font', **font)
label_fontsize = 12
title_fontsize = 14

plt.rcParams['axes.edgecolor'] = 'black'

# %% [markdown]
# ### 01 指标计算
# 根据附件15的数据来计算2016-2019年的植被生物量

# %%
biomass = pd.read_excel('15_biomass.xlsx')
biomass.head()

# %%
date = list(biomass['日期'])
year = list(biomass['年份'])
cell = list(biomass['重复'])
weight = list(biomass['干重(g)'])
fweight = list(biomass['鲜重(g)'])
period = list(biomass['轮次'])
intensity = list(biomass['处理'])
block = list(biomass['放牧小区Block'])

# 返回列表中所有重复值的开始和结束位置
def repeat_index(ls):
    index = []
    for i in range(len(ls) - 1):
        if not ls[i] == ls[i + 1]:
            index.append(i)
    index.append(len(ls) - 1)
    return index

# 数组分组索引
weight_repeat_index = repeat_index(cell)
# 计算样方内植被生物量
weight_in_cell = []
weight_in_cell.append(sum(weight[0:(weight_repeat_index[0] + 1)]))
for i in range(len(weight_repeat_index) - 1):
    weight_in_cell.append(round(sum(weight[weight_repeat_index[i] + 1:weight_repeat_index[i + 1] + 1]), 2))

# 对应的年份数据处理
date_toweight = []
for i in weight_repeat_index:
    date_toweight.append(date[i])

# 数据中的2016.6.1要和2016.6.21合并，同时变成一个月的数据
for i in range(len(date_toweight)):
    if date_toweight[i] == '2016.6.1':
        date_toweight[i] = '2016.6.21'

# 创建一个字典为后续匹配做准备
date_dict = {'2016':list(np.zeros(4)), '2017':list(np.zeros(4)), '2018':list(np.zeros(4)), '2019':list(np.zeros(4)), '2020':list(np.zeros(4))}

date_repeat_index = repeat_index(date_toweight)
print(date_toweight[60])

# 计算同一个月内的平均值
weight_in_month = []
weight_in_month.append(round(np.mean(weight_in_cell[0:(date_repeat_index[0] + 1)]), 2))
for i in range(len(date_repeat_index) - 1):
    weight_in_month.append(round(np.mean(weight_in_cell[date_repeat_index[i] + 1:date_repeat_index[i + 1] + 1]), 2))
print(date_repeat_index)
print(weight_in_month)

                                                                     


# %% [markdown]
# ### 02 回归方法
# 依据初始植被生物量$W_0$、无放牧天数$D$和放牧强度$S$和一些天气数据，用回归算法来预测最终的$\Delta W$

# %%
# 这里是包含了2016.6.1日期的date数据
date_toweight = []
for i in weight_repeat_index:
    date_toweight.append(date[i])

period_toweight = []
for i in weight_repeat_index:
    period_toweight.append(period[i]) 

intensity_toweight = []
for i in weight_repeat_index:
    intensity_toweight.append(intensity[i])

year_toweight = []
for i in weight_repeat_index:
    year_toweight.append(year[i])

cell_toweight = []
for i in weight_repeat_index:
    cell_toweight.append(cell[i])

block_toweight = []
for i in weight_repeat_index:
    block_toweight.append(block[i])

csv_dict = {'年份':year_toweight, '轮次':period_toweight, '处理':intensity_toweight, '日期':date_toweight, '放牧小区':block_toweight, '样方':cell_toweight, '植被生物量':weight_in_cell}
# 将字典转为dataframe
csv_df = pd.DataFrame(csv_dict)
csv_df.to_excel('15_biomass_finished.xlsx')


# %% [markdown]
# 计算初始植被生物量$W_0$，植被变化量$\Delta W$，以及自由生长时间

# %%
# 重新读取排序好的csv数据
order_biomass = pd.read_excel('15_biomass_ordered.xlsx')
# 创建空的dataframe表格
excel_dict = {'年份':[], '轮次':[], '处理':[], '日期':[], '放牧小区':[], '样方':[], '初始生物量':[], '植被变化量':[], '自由生长时间':[]}
excel_df = pd.DataFrame(excel_dict)
# 创建order列表，用来判断excel表格的上下行是否为同一样方的连续时间数据
order_list = ['牧前', '第一轮牧后', '第二轮牧后', '第三轮牧后', '第四轮牧后']
# 计算牧草自然生长的时间
label_dict = {'无牧（0天）':0, '轻牧（3天）':3, '中牧（6天）':6, '重牧（12天）':12}
# 两次测量之间的间隔时间
free_time = {'2016.6.21':20, '2016.7.21':30, '2016.8.21':31, '2016.9.21':31,
'2017.6.26':33, '2017.7.22':26, '2017.8.27':36, '2017.9.20':24, '2018.6.25':41,
'2018.7.22':27, '2018.8.21':30, '2018.9.21':31, '2019.6.26':47, '2019.7.24':28,
'2019.8.26':33, '2019.9.18':23, '2020.6.29':31, '2020.7.20':21, '2020.8.14':25,
'2020.9.17':34}

for i in range(len(order_biomass)):
    # 去除牧前测量的情况
    if order_biomass.iloc[i, 1] == '牧前':
        continue
    else:
        label_order = order_list.index(order_biomass.iloc[i, 1])
        # print(order_list[label_order - 1], order_biomass.iloc[i-1, 2])
        '''
        确保：
        1. 前后时间上是连续的
        2. 前后是同一block的
        3. 前后是同一样方的
        '''
        if order_list[label_order - 1] == order_biomass.iloc[i-1, 1] and order_biomass.iloc[i, 4] == order_biomass.iloc[i - 1, 4] \
            and order_biomass.iloc[i, 5] == order_biomass.iloc[i - 1, 5]:
            total_days = free_time[order_biomass.iloc[i, 3]]
            pasture_days = label_dict[order_biomass.iloc[i, 2]]
            free_days = total_days - pasture_days
            new_row = list(order_biomass.iloc[i])[:-1] + [order_biomass.iloc[i - 1, 6], order_biomass.iloc[i, 6] - order_biomass.iloc[i - 1, 6], free_days]
            excel_df.loc[len(excel_df)] = new_row

# 将年份、样方和自由生长时间变为int类型
for i in range(len(excel_df)):
    excel_df.iloc[i, 0] = int(excel_df.iloc[i, 0])
    excel_df.iloc[i, 5] = int(excel_df.iloc[i, 5])
    excel_df.iloc[i, 8] = int(excel_df.iloc[i, 8])

excel_df.to_excel('15_regression_x3.xlsx')

# %% [markdown]
# 清洗整理天气数据，并进行匹配

# %%
climate_df = pd.read_excel('08_climate.xlsx')
years = list(climate_df['年份'])
months = list(climate_df['月份'])
temperature = list(climate_df['平均气温(℃)'])
rainfall = list(climate_df['降水量(mm)'])
windspeed = list(climate_df['平均风速(knots)'])
climate_df

# 将年份和月份数据打包
year_month = list(zip(years, months))

regression_x3_df = pd.read_excel('15_regression_x3.xlsx')
date = regression_x3_df['日期']
date_ls = [i.split('.') for i in date]
date_tuple = [(int(i[0]), int(i[1])) for i in date_ls]

# 气温、降水和风速等数据的匹配
temperature_regression = []
rainfall_regression = []
windspeed_regression = []
for i in date_tuple:
    date_index = year_month.index(i)
    temperature_regression.append(temperature[date_index])
    rainfall_regression.append(rainfall[date_index])
    windspeed_regression.append(windspeed[date_index])

# 将数据添加到excel表格中
regression_x6_df = regression_x3_df.copy()
regression_x6_df['平均气温(℃)'] = temperature_regression
regression_x6_df['降水量(mm)'] = rainfall_regression
regression_x6_df['平均风速(knots)'] = windspeed_regression

regression_x6_df.to_excel('15_regression_x6.xlsx')



# %% [markdown]
# 读取数据，可视化数据之间的关系

# %%
regression_df = pd.read_excel('15_regression_x6.xlsx')
regression_df

# %% [markdown]
# 研究不同年份植物生长率的变化量（随机选取G17、G6、G8、G9四个区域）

# %%
# 为原始数据增加月份数据
date = regression_df['日期']
month = [int(i.split('.')[1]) for i in date]
year = list(regression_df['年份'])
delta_plant = list(regression_df['植被变化量'])
temp = list(regression_df['平均气温(℃)'])
rain = list(regression_df['降水量(mm)'])
wind = list(regression_df['平均风速(knots)'])
regression_df['月份'] = month

cell = list(regression_df['样方'])
def plot_month_dplant_year(df):
    block_ls = ['G17', 'G6', 'G8', 'G9']
    group_dict = df.groupby(by=['放牧小区', '样方', '年份']).groups
    location = [221, 222, 223, 224]
    years = list(range(2016, 2021))
    count = 0
    plt.figure(dpi=300,figsize=(20, 20))
    for i in block_ls:
        plot_ls = []
        for j in range(2016, 2021):
            if (i, 1, j) in group_dict:
                plot_ls.append([delta_plant[c] for c in group_dict[(i,1,j)]])
        plt.subplot(location[count])
        plt.title(f'{i}小区样方1植被生物量变化量')
        plt.xlabel('月份')
        plt.ylabel('植被生物量变化量/g')
        plt.xticks([6, 7, 8, 9])
        plt.ylim(-80, 160)
        month_ticks = [6, 7, 8, 9]
        for a in range(len(plot_ls)):
            month_ticks = [int(i) for i in month_ticks]
            plt.plot(month_ticks, plot_ls[a], 'o-', markersize=10, label=f'{years[a]}年')
        plt.legend()

        count += 1
    plt.savefig('./pics/01_不同年份4小区植被变化量.jpg', dpi=300)
plot_month_dplant_year(regression_df)


    

# %%
def plot_month_climate_year(df):
    name_ls = ['平均气温', '平均降水量', '平均风速']
    label_ls = ['温度/℃', '降水量/mm', '风速/knots']
    attri_ls = [temp, rain, wind]
    group_dict = df.groupby(by=['年份', '月份']).groups
    location = [311, 312, 313]
    months = list(range(6, 10))
    plt.figure(dpi=300,figsize=(30, 30))
    for i in range(len(name_ls)):
        plt.subplot(location[i])
        for j in range(2016, 2021):
            plot_ls = []
            for k in months:
                ind = group_dict[(j, k)][0]
                plot_ls.append(attri_ls[i][ind])
            plt.plot(months, plot_ls, 'o-', markersize=10, label=f'{j}年')
        plt.title(f'2016-2020{name_ls[i]}变化趋势', fontsize=20)
        plt.xlabel('月份', fontsize=20)
        plt.ylabel(label_ls[i], fontsize=20)
        plt.xticks([6, 7, 8, 9]) 
        plt.legend(loc='upper right')
    plt.savefig('./pics/03_不同年份气候变化.jpg', dpi=300)
plot_month_climate_year(regression_df)

# %% [markdown]
# 结论：重度放牧情况下，2016-2020年的变化量相对是最小的，而其他情况下，变化量都相对较大。
# 
# 计算六种自变量与植被生物量变化量之间的相关系数。
# 
# 绘制相关系数图像

# %%
# 对植物生物量进行编号

strength_list = list(regression_df['处理'])
print(strength_list[:100])

for i in range(len(strength_list)):
    if strength_list[i] == '无牧（0天）':
        strength_list[i] = 0.0
    elif strength_list[i] == '轻牧（3天）':
        strength_list[i] = 2.0
    elif strength_list[i] == '中牧（6天）':
        strength_list[i] = 4.0
    else:
        strength_list[i] = 8.0

regression_df['处理'] = strength_list

data_pearson_df = regression_df.copy()
print(data_pearson_df)
# 按排名对pearson和spearman相关系数绘制柱状图
pearson = data_pearson_df.drop(columns=["年份", "样方"]).corr('pearson')
spearman = data_pearson_df.drop(columns=["年份", "样方"]).corr('spearman')
dplant_pearson = abs(pearson['植被变化量'])
dplant_factor = pearson.index.values
print(dplant_factor)
zip_dplant_pearson = sorted(zip(dplant_pearson, dplant_factor), reverse=True)
dplant_pearson_order = [i[0] for i in zip_dplant_pearson[1:]]
dplant_pearson_factor_order = [i[1] for i in zip_dplant_pearson[1:]]
dplant_spearman = abs(spearman['植被变化量'])
zip_dplant_spearman = sorted(zip(dplant_spearman, dplant_factor), reverse=True)
dplant_spearman_order = [i[0] for i in zip_dplant_spearman[1:]]
dplant_spearman_factor_order = [i[1] for i in zip_dplant_spearman[1:]]
print(dplant_spearman_factor_order, dplant_spearman_order)
print(dplant_pearson_factor_order, dplant_pearson_order)

plt.figure(dpi=300, figsize=(20, 10))
plt.subplot(121)
plt.bar(dplant_pearson_factor_order, dplant_pearson_order)
for i in range(len(dplant_pearson_order)):
    plt.text(i-0.1, dplant_pearson_order[i] + 0.01, round(dplant_pearson_order[i], 2), fontsize=10)
plt.title('Pearson相关系数排序')
plt.xlabel('相关指标')
plt.subplot(122)
plt.bar(dplant_spearman_factor_order, dplant_spearman_order)
for i in range(len(dplant_spearman_order)):
    plt.text(i-0.1, dplant_spearman_order[i] + 0.01, round(dplant_spearman_order[i], 2), fontsize=10)
plt.title('Spearman相关系数排序')
plt.xlabel('相关指标')
plt.savefig('./pics/02 Pearson和Spearman相关系数排序.jpg', dpi=300)
regression_df



# %%
plt.figure(dpi=300, figsize=(20, 10))
plt.subplot(121)
sns.heatmap(pearson, annot=True, square=True, cmap='YlGnBu')
plt.subplot(122)
sns.heatmap(spearman, annot=True, square=True, cmap='YlGnBu')
plt.savefig('a.jpg', dpi=300)

# %% [markdown]
# 使用lightgbm进行预测

# %%
regression_df

# %%
from sklearn import preprocessing

X = np.array(regression_df.drop(columns=['年份', '轮次', '日期', '放牧小区', '样方', '植被变化量']))
print(X)
y = np.array(regression_df['植被变化量'])

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
train_data = lgb.Dataset(data=X_train, label=y_train)
test_data = lgb.Dataset(data=X_test, label=y_test)

# gbm = lgb.LGBMRegressor(
#     objective = 'regression',
#     metric = 'l2_root',
#     learning_rate = 0.01,
#     num_leaves = 31,
#     n_estimators=5000
#     )
# print(gbm.get_params)


# param_grid = {
#     'min_child_samples':[20, 40, 60],
#     'max_depth': [3, 4, 5, 6]
# }

# gsearch = GridSearchCV(
#     gbm, 
#     param_grid=param_grid, 
#     scoring='r2', 
#     cv=10)
evals_result={}
param = {'num_leaves':40, 'max_depth':5, 'objective':'regression', 'metric':'l2_root', 'min_data_in_leaf':50, 'learning_rate':0.01, 'min_child_sample':20}

# gsearch.fit(X_train, y_train)
num_round = 1000
bst = lgb.train(param, train_data, num_round, evals_result=evals_result, valid_sets=[test_data])
bst.save_model('./lgbmodel.txt')

# print('参数的最佳取值:{0}'.format(gsearch.best_params_))
# print('最佳模型得分:{0}'.format(gsearch.best_score_))

y_pred = list(bst.predict(X_test))
print(f'R2为{r2_score(y_test, y_pred)},RMSE为{np.sqrt(mean_squared_error(y_test, y_pred))}')

plt.figure(dpi=300, figsize=(20, 10))
plt.subplot(111)
plt.show()






# %%
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

scaler1 = preprocessing.StandardScaler().fit(X)
X = scaler1.transform(X)
y = y.reshape(-1, 1)
scaler2 = preprocessing.StandardScaler().fit(y)
y = scaler2.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
clf = MLPRegressor(
    solver='lbfgs', 
    activation='relu',
    alpha=1e-5,
    hidden_layer_sizes=(10, 10, 3), 
    random_state=1)
clf.fit(X_train, y_train)
y_pred = list(clf.predict(X_test))
print(f'R2为{r2_score(y_test, y_pred)},RMSE为{np.sqrt(mean_squared_error(y_test, y_pred))}')






