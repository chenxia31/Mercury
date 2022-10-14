# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
import math
plt.style.use('ggplot')
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# %%
shiduW=pd.read_excel('土壤湿度2012-2022各月份.xls')
youjiwuO=pd.read_excel('内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）.xlsx')


# %%
shiduW.columns

# %%
plt.figure(figsize=(6,5))
shidulen=len(shiduW)
xticklabelsname=[str(shiduW.iloc[i,0])+'年'+str(shiduW.iloc[i,1])+'月' for i in range(len(shiduW))]
plt.plot(list(range(shidulen)),[math.log10(i) for i in shiduW.iloc[:,4]])
plt.plot(list(range(shidulen)),[math.log10(i) for i in shiduW.iloc[:,5]])
plt.plot(list(range(shidulen)),[math.log10(i) for i in shiduW.iloc[:,6]])
plt.plot(list(range(shidulen)),[math.log10(i) for i in shiduW.iloc[:,7]])
plt.xticks(list(range(0,shidulen,11)),xticklabelsname[::11],rotation='vertical')
plt.legend(['10cm','40cm','100cm','200cm'])
plt.ylabel('湿度（kg/m2)')
plt.yticks(list(range(1,3)),['10mm','100mm'])

# %%
# 模拟生成一组实验数据
x = [1,3,5,7]
y = [0.41,0.38,0.43,0.46]
plt.figure(figsize=(8,8))
fig, ax = plt.subplots()
ax.plot(x, y, 'b--')
ax.set_xlabel('x')
ax.set_ylabel('y')

# 二次拟合
coef = np.polyfit(x, y, 2)
y_fit = np.polyval(coef, x)
ax.plot(x, y_fit, 'g')
# 找出其中的峰值/对称点
if coef[0] != 0:
    x0 = -0.5 * coef[1] / coef[0]            
    x0 = round(x0, 2)        
    ax.plot([x0]*5, np.linspace(min(y),max(y),5),'r--')
    print(x0)
else:
    raise ValueError('Fail to fit.')
plt.xlabel('放牧强度')
plt.ylabel('沙漠化指数SM')
plt.show()
coef


# %%
import math
def f(x):
    res=0
    res+=0.68/(0.04*math.pow(2.7,-x*x/128)+1.36)
    print(0.68/(0.04*math.pow(2.7,-x*x/128)+1.36))
    res+=0.00375*x*x-0.002*x+0.422
    return res

temp=list(range(50))
x=[i/100 for i in temp]
y=[f(i) for i in x]
plt.plot(x,y)
plt.xlabel('放牧强度')
plt.ylabel('SM+B')


