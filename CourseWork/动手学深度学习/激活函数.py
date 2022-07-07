#%%
import  torch
import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize # import figsize
#figsize(12.5, 4) # 设置 figsize
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300
plt.figure()

x=torch.arange(-8,8,0.1,requires_grad=True)
y_relu=torch.relu(x)
y_sigmoid=torch.sigmoid(x)
y_tanh=torch.tanh(x)

plt.subplot(231)
plt.plot(x.detach(),y_relu.detach())
plt.ylabel('ReLU')
plt.grid()
plt.subplot(232)
plt.plot(x.detach(),y_sigmoid.detach())
plt.ylabel('sigmoid')
plt.grid()
plt.subplot(233)
plt.plot(x.detach(),y_tanh.detach())
plt.ylabel('tanh')
plt.grid()
plt.subplot(234)
y_relu.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(),x.grad)
plt.ylabel('ReLU')
plt.grid()
plt.subplot(235)
x.grad.data.zero_()
y_sigmoid.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(),x.grad)
plt.ylabel('sigmoid')
plt.grid()
plt.subplot(236)
x.grad.data.zero_()
y_tanh.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(),x.grad)
plt.ylabel('tanh')
plt.grid()
plt.show()