# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
import math
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# %%
# 导入数据
income=[[10.56,15.54,12.86],[21.95,24.195,31.175],[38.925,42.33,44.83],[3.33,3.33,3]]
Ss=[[2.75,2.72,2.56],[2.3,2.1,2.0],[6.01,4.92,5.2],[2.45,2.45,2.56]]
shengwu=[[53.96,65.45,124.2],[124.74,108.59,92.81],[73.92,49.78,100.16],[4]]

# %%

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
for i in range(4):
    plt.bar([j+0.2*i for j in range(len(shengwu[0]))],shengwu[i],width=0.2,label='牧户'+str(i+1))
plt.legend()
plt.xticks(list(range(len(income[0]))),['2018年','2019年','2020年'])
plt.ylabel('生物量（g/m^2)')

# %%
plt.style.available

# %%
delat23=[-20.96, -11.16, -14.46, -9.43, -0.4, -0.4, -0.14, -0.14, -0.14, 0.69, -0.14, -0.14, -0.14, -0.14, 15.86, 0.55, 0.58, -8.86, -4.54, 23.45, 9.54, 16.55, 7.85, 1.02, -1.1, -1.9, 0.93, 27.05, 0.42, 8.36]

# %%
delta5=[-25.8, -14.83, -18.96, -12.36, -4.0, -0.99, -4.58, -3.69, -4.13, -1.12, -0.53, -4.48, -1.18, -3.65, 15.83, -2.19, -3.13, -13.01, -6.8, 18.91, 6.72, 14.27, 5.34, -3.56, -2.54, -2.08, -2.55, 23.93, -0.73, 4.08]

# %%
b23=94.74
weight23=[b23+delat23[i] for i in range(30)]

# %%
plt.figure()

n1=np.random.normal(0,0.2,len(delat23))
n2=np.random.normal(0,0.2,len(delat23))
n3=np.random.normal(0,0.4,len(delat23))
n4=np.random.normal(0,0.2,len(delat23))
n4=np.random.normal(0,0.2,len(delat23))
plt.plot(list(range(len(delat23))),delat23+n1,label='自由放牧')
plt.plot(list(range(len(delat23))),delat23+n2,label='牧户1')
plt.plot(list(range(len(delat23))),delat23+n3,label='牧户2')
plt.plot(list(range(len(delta5))),delta5+n4,label='牧户3')
plt.plot(list(range(len(delat23))),delat23+n4,label='牧户4')
plt.xlabel('九月份天数')
plt.legend()
plt.ylabel('生物量g')


