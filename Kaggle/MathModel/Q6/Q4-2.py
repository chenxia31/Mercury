# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import math
plt.style.use('_mpl-gallery')
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
shengwut=pd.read_excel('shengwu.xlsx',sheet_name='2016-2020物种数据库')
weather=pd.read_excel('weather.xlsx')
zhengfa=pd.read_excel('土壤蒸发量2012—2022年各月份.xls')
zhibei=pd.read_excel('植被指数-NDVI2012-2022年.xls')
people=pd.read_excel('people.xlsx')

# %% [markdown]
# ## 气象因素分析

# %%
length=len(weather.iloc[:,5])
tempticks=[str(2012+i)+'年'+str(j)+'月' for j in range(1,13) for i in range(11)]
tempXticks=list(range(0,len(weather.iloc[:,5]),11))
tempticks=tempticks[:length:11]


plt.figure(figsize=(14,12))

plt.subplot(2,2,1)
plt.fill_between(list(range(length)),weather.iloc[:,6],weather.iloc[:,7],alpha=0.5)
plt.plot(list(range(length)),weather.iloc[:,5])
plt.xticks(tempXticks,tempticks,rotation='vertical')
plt.ylabel('气温（度）')
plt.xlabel('年份-月份（每月）')

plt.subplot(2,2,2)
plt.plot(list(range(length)),list(math.log(abs(i+0.01),10) for i in weather.iloc[:,14]))
plt.plot(list(range(length)),list(math.log(i+0.01,10) for i in weather.iloc[:,15]))
plt.xticks(tempXticks,tempticks,rotation='vertical')
plt.yticks(list(range(-2,4,1)),['0.01','0.1','1','10','100','1000'])
plt.ylabel('降水量(mm)')
plt.legend(['平均降水量','单日最大降水'])
plt.xlabel('年份-月份（每月）')

plt.subplot(2,2,3)
plt.plot(list(range(length)),weather.iloc[:,-5])
plt.plot(list(range(length)),weather.iloc[:,-4])
plt.plot(list(range(length)),weather.iloc[:,-3])
plt.xticks(tempXticks,tempticks,rotation='vertical')
plt.ylabel('风速(knot)')
plt.legend(['平均风速','平均最大持续风速','单日最大平均风速'])
plt.xlabel('年份-月份（每月）')

plt.subplot(2,2,4)
plt.plot(list(range(len(zhengfa.iloc[:,-1]))),zhengfa.iloc[:,-1])
plt.xticks(tempXticks,tempticks,rotation='vertical')
plt.ylabel('蒸发量（mm)')
plt.xlabel('年份-月份（每月）')



# %% [markdown]
# ## 气象因素的互相关量

# %%
weather.iloc[:123,5].corr(zhengfa.iloc[:,-1],method="pearson")

# %%
weather.iloc[:123,5].corr(zhibei.iloc[:,-1],method="pearson")

# %%
weather.iloc[:123,14].corr(zhengfa.iloc[:,-1])

# %%
wendu=weather.iloc[:123,5]
jiangshui=weather.iloc[:123,14]
fengsu=weather.iloc[:123,-5]
zhengfa=zhengfa.iloc[:123,-1]
zhibei=zhibei.iloc[:123,-1]
data=pd.DataFrame({'wendu':wendu,'jiangshui':jiangshui,'fengsu':fengsu,'zhengfa':zhengfa,'zhibei':zhibei})

# %%
data.corr()

# %% [markdown]
# ## 生物因素的处理

# %%
shengwut.columns

# %%
plt.figure(figsize=(4,3))
peoplelen=len(people)
peopletick=[str(2012+i)+'年'+str(j)+'月' for j in [6,12] for i in range(9)]
cattle=[]
sheep=[]
for i in range(peoplelen):
    cattle.append(people.iloc[i,2])
    sheep.append(people.iloc[i,3])
    cattle.append(people.iloc[i,4])
    sheep.append(people.iloc[i,5])
plt.plot(list(range(2*peoplelen)),cattle)
plt.plot(list(range(2*peoplelen)),sheep)
plt.legend(['牛','羊'])
plt.xticks(list(range(2*peoplelen)),peopletick,rotation='vertical')
plt.ylabel('牧畜数量（万头）')

# %% [markdown]
# ## 地表因素

# %%
shengwut=pd.read_excel('shengwu.xlsx',sheet_name='2016-2020物种数据库',dtype={'干重(g)':float})
shengwut.columns

# %%
# 根据这四个可以唯一确定一个值
Ss=np.unique(shengwut.iloc[:,2])
year=np.unique(shengwut.iloc[:,0])
block=np.unique(shengwut.iloc[:,6])
rounds=np.unique(shengwut.iloc[:,1])

# %%
dataset=pd.DataFrame(columns=['S','year','plot','rounds','num','weight1','weight2'])
for s in Ss:
    # 不同的放牧策略
    for y in year:
        # 不同的时间点
        for b in block:
            for r in rounds:
                temp=shengwut[(shengwut.iloc[:,2]==s) & (shengwut.iloc[:,0]==y) & (shengwut.iloc[:,6]==b) & (shengwut.iloc[:,1]==r)]
                if len(temp)!=0:              
                    temp_num=sum(temp['株/丛数'])
                    tempw_1=sum(temp['鲜重(g)'])
                    tempw_2=sum(temp['干重(g)'])
                    dataset.loc[len(dataset.index)]=[s,y,b,r,temp_num,tempw_1,tempw_2]

# %%
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
sns.boxplot(x=dataset['year'],y=dataset['num'],color="skyblue")
plt.xlabel('年份')
plt.ylabel('数量')
plt.subplot(1,3,2)
sns.boxplot(x=dataset['year'],y=dataset['weight1'],color="bisque")
plt.title('植被随年份变化箱图')
plt.xlabel('年份')
plt.ylabel('鲜重(g)')
plt.subplot(1,3,3)
sns.boxplot(x=dataset['year'],y=dataset['weight2'],color="thistle")
plt.xlabel('年份')
plt.ylabel('干重(g)')



# %%
dataset.to_csv('dataset.csv',encoding='utf-8')

# %%
dataset['tem']=0
dataset['rain']=0
dataset['wind']=0
dataset['eva']=0
dataset['nvdi']=0
dataset['temstd']=0
dataset['rainstd']=0
dataset['windstd']=0
dataset['evastd']=0
dataset['nvdistd']=0
dataset['people']=0
dataset['cattle6']=0
dataset['sheep6']=0
dataset['cattle12']=0
dataset['sheep12']=0
dataset['income']=0
for y in year:
    dataset.loc[dataset['year']==y,'tem']=np.mean(weather[weather.iloc[:,0]==y].iloc[:,5])
    dataset.loc[dataset['year']==y,'rain']=np.mean(weather[weather.iloc[:,0]==y].iloc[:,14])
    dataset.loc[dataset['year']==y,'wind']=np.mean(weather[weather.iloc[:,0]==y].iloc[:,-5])
    dataset.loc[dataset['year']==y,'eva']=np.mean(zhengfa[zhengfa.iloc[:,0]==y].iloc[:,5])
    dataset.loc[dataset['year']==y,'nvdi']=np.mean(zhibei[zhibei.iloc[:,0]==y].iloc[:,-1])
    dataset.loc[dataset['year']==y,'temstd']=np.std(weather[weather.iloc[:,0]==y].iloc[:,5])
    dataset.loc[dataset['year']==y,'rainstd']=np.std(weather[weather.iloc[:,0]==y].iloc[:,14])
    dataset.loc[dataset['year']==y,'windstd']=np.std(weather[weather.iloc[:,0]==y].iloc[:,-5])
    dataset.loc[dataset['year']==y,'evastd']=np.std(zhengfa[zhengfa.iloc[:,0]==y].iloc[:,5])
    dataset.loc[dataset['year']==y,'nvdistd']=np.std(zhibei[zhibei.iloc[:,0]==y].iloc[:,-1])
    dataset.loc[dataset['year']==y,'people']=people[people.iloc[:,0]==y].iloc[0,1]
    dataset.loc[dataset['year']==y,'cattle6']=people[people.iloc[:,0]==y].iloc[0,2]
    dataset.loc[dataset['year']==y,'sheep6']=people[people.iloc[:,0]==y].iloc[0,3]
    dataset.loc[dataset['year']==y,'cattle12']=people[people.iloc[:,0]==y].iloc[0,4]
    dataset.loc[dataset['year']==y,'sheep12']=people[people.iloc[:,0]==y].iloc[0,5]
    dataset.loc[dataset['year']==y,'income']=people[people.iloc[:,0]==y].iloc[0,6]

# %% [markdown]
# ## 整理后的dataset（训练数据集）

# %%
dataset=pd.read_excel('dataset.xlsx')

# %%
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
X=X.iloc[:,3:]
X = (X - X.min()) / (X.max() - X.min())

# %%
plt.figure(figsize=(6,6))
sns.heatmap(X.corr(),cmap="Purples")

# %%
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier    #导入需要的模块
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

# model = GradientBoostingClassifier()
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# model = GradientBoostingRegressor()
# n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
# define dataset


# %% [markdown]
# ### model1 - light GBM调整参数

# %% [markdown]
# 

# %%
import lightgbm as lgb
from lightgbm import LGBMClassifier
model = LGBMClassifier()
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':4,
    'learning_rate': 0.1,
    'metric':'softmax',
    'num_leaves': 50, 
    'max_depth': 6,
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
}
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))



# %% [markdown]
# #### 决策树参数调整

# %%
data_train = lgb.Dataset(X, y, silent=True)
cv_results = lgb.cv(
    lgb_params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='softmax',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
print('best n_estimators:', len(cv_results['multi_logloss-mean']))
print('best cv score:', cv_results['multi_logloss-mean'][-1])

# %%
from sklearn.model_selection import GridSearchCV
### 我们可以创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目)
model = LGBMClassifier(objective= 'multiclass',
                       num_class=4,
                       learning_rate=0.1,
                       metric='softmax',
                       n_estimators=5,
                       max_depth=6,
                       subsample=0.8, 
                       colsample_bytree=0.8)

params_test1={
    'max_depth': range(2,30,1),
    'num_leaves':range(2,170, 3)
}
gsearch1 = GridSearchCV(estimator=model, param_grid=params_test1, scoring='f1_weighted', cv=5, verbose=1, n_jobs=4)

# %%
gsearch1.fit(X,y)
gsearch1.best_params_, gsearch1.best_score_
# 取max_depth=10,num_leaves=50

# %% [markdown]
# ### 降低过拟合

# %%
params_test3={
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002]
}
model_lgb = lgb.LGBMClassifier(objective='softmax',num_leaves=50, num_class=4,
                              learning_rate=0.1, n_estimators=35, max_depth=7, 
                              metric='softmax', bagging_fraction = 0.8, feature_fraction = 0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='f1_weighted', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(X,y)
gsearch3.best_params_, gsearch3.best_score_

# %% [markdown]
# ### 正则化项目参数调节

# %%
model_lgb = lgb.LGBMClassifier(objective='softmax',num_leaves=50, num_class=4,
                              learning_rate=0.1, n_estimators=35, max_depth=7, 
                              metric='softmax',min_child_samples=1,min_child_weight=0.001)
params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='f1_weighted', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(X,y)
gsearch4.best_params_, gsearch3.best_score_

# %%
params_test5={
    'bagging_fraction': [0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94 ]
}
model_lgb = lgb.LGBMClassifier(objective='softmax',num_leaves=50, num_class=4,
                              learning_rate=0.1, n_estimators=35, max_depth=7, 
                              metric='softmax',min_child_samples=1,min_child_weight=0.001,feature_fraction=0.6)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='f1_weighted', cv=10, verbose=1, n_jobs=4)
gsearch5.fit(X,y)
gsearch5.best_params_, gsearch3.best_score_

# %%
model_lgb = lgb.LGBMClassifier(objective='softmax',num_leaves=50, num_class=4,
                              learning_rate=0.1, n_estimators=35, max_depth=7, 
                              metric='softmax',min_child_samples=1,min_child_weight=0.001,feature_fraction=0.6,bagging_fraction=0.82)
params_test6={
    'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
}
gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='f1_weighted', cv=10, verbose=1, n_jobs=4)
gsearch6.fit(X,y)
gsearch6.best_params_, gsearch3.best_score_

# %% [markdown]
# #### 使用lightGBM自带的参数

# %%
import lightgbm as lgb
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':4,
    'learning_rate': 0.01,
    'metric':'softmax',
    'num_leaves': 50, 
    'max_depth': 7,
    'min_child_samples':1,
    'min_child_weight':0.001,
    'feature_fractio':0.6,
    'bagging_fraction':0.82,
    'n_estimators':35,
    'reg_alpha': 0.03,
    'reg_lambda': 0.001
}
model_lgb = lgb.LGBMClassifier(objective='softmax',num_leaves=50, num_class=4,
                              learning_rate=0.1, n_estimators=35, max_depth=7, 
                              metric='softmax',min_child_samples=1,min_child_weight=0.001,
                              feature_fraction=0.6,bagging_fraction=0.82,reg_alpha=0.03,reg_lambda=0.001)
data_train = lgb.Dataset(X, y, silent=True,feature_name=X.columns.tolist())
cv_results = lgb.cv(
    lgb_params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True,
    early_stopping_rounds=50, verbose_eval=100, show_stdv=True)


# %%
len(cv_results['multi_logloss-mean'])

# %%
import lightgbm as lgb
model_lgb = lgb.LGBMClassifier(objective='softmax',num_leaves=50, num_class=4,
                              learning_rate=0.1, n_estimators=35, max_depth=7, 
                              metric='softmax',min_child_samples=1,min_child_weight=0.001,
                              feature_fraction=0.6,bagging_fraction=0.82,reg_alpha=0.03,reg_lambda=0.001)
model_lgb.fit(X,y)
plt.figure(figsize=(12,6))
lgb.plot_importance(model_lgb)
plt.title("Featurertances")
plt.show()

booster = model_lgb.booster_
importance_lgb = booster.feature_importance(importance_type='split')

# %%
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X, y)
importances = forest.feature_importances_

# %%
plt.figure(figsize=(12,6))
plt.bar(list(range(len(importances))),importances,width=0.3)
plt.bar(list(i+0.3 for i in range(len(importances))),list(j/sum(importance_lgb) for j in importance_lgb),width=0.3)
plt.legend(['RandomForest(F1score 0.75)','LightGBM(F1score 0.63)'])
plt.xticks(list(range(len(importances))),X.columns.tolist())

# %%
q1=list(j/sum(importance_lgb) for j in importance_lgb)
q2=importances
import math
q=[math.sqrt(0.63*pow(q1[i],2)+0.75*pow(q2[i],2)) for i in range(len(q1))]
q

# %%
X.columns

# %% [markdown]
# ### 计算权重修正系数\eta

# %%
1-sum(q*X.mean())


# %%
0.42/0.61

# %%
for i in range(4):
    temp=X.loc[y==i,:]
    print(0.688*(1-sum(q*temp.mean())))

# %%



