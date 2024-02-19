import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import mpl      # 显示中文
from scipy import stats
# x = np.arange(-10, 10, 0.01) # 生成一个 [-10, 10] 之间差值为 0.01 的等差数列，代表图像中的 x axis
# y = []
# for num in x:
#     if num <= 0 :
#         y.append(0)
#     else:
#         y.append(num)
#
# plt.title('ReLU Function')
# plt.xlabel('z')
# plt.ylabel('ReLU(z)')
# plt.plot(x, y,color='#191970')
# plt.show()

x = np.linspace(-5,5,100)
y1 = stats.laplace.pdf(x)
y2 = stats.laplace.pdf(x,loc=0.0,scale=2.)
y3 = stats.laplace.pdf(x,loc=1.0,scale=2.)
plt.xlabel('x')
plt.ylabel('pdf')
#plt.plot(x,y1,label='u=0;r=1')
plt.plot(x,y2)
#plt.plot(x,y3)
plt.legend()
plt.savefig('D:/test1.jpg', dpi=700)