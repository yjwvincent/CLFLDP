import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(15, 5), dpi=70)  # 设置图像大小 # MNIST

x_labels = ['10','20','30','40','50','60','70','80','90', '100'] #T=100 p=0.5 different N
x_values = [
    [21840000 * 4, 43680000 * 4, 65520000 * 4, 87360000 * 4, 109200000 * 4, 131040000 * 4, 152880000 * 4, 174720000 * 4, 196560000 * 4, 218400000 * 4],
    [10920000 * 4, 21840000 * 4, 32760000 * 4, 43680000 * 4, 54600000 * 4, 65520000 * 4, 76440000 * 4, 87360000 * 4, 98280000* 4, 109200000 * 4], # PQ=172850184 QSGD = 157086182
    [8928570 * 4, 17545821 * 4, 24967105 * 4, 30851032 * 4, 37284081 * 4, 46244012 * 4, 57289102 * 4, 68820195 * 4, 77287105 * 4, 83974025 * 4],
    [8727419 * 4, 15802819 * 4, 22862019 * 4, 28180847 * 4, 35091835 * 4, 44010854 * 4, 55475912 * 4, 64921373 * 4, 74394204 * 4, 81034325 * 4], # Q=0.6 75947166 下面这两行不用了
    [7715150 * 4, 14410208 * 4, 20138270 * 4, 26391792 * 4, 33228981 * 4, 42818257 * 4, 52192741 * 4, 58291275 * 4, 65297091 * 4, 73275307 * 4] # Q=0.8 61058494
]
models = ['FedAvg', 'Fedsel (p=0.5,$\epsilon=4$)', 'CMFL (Q=0.7)', 'COFEL (Q=0.7,$\epsilon=4$)', 'GLCFL (p=0.5,Q=0.7,$\epsilon=2$)']
color = ['#4473c5', '#ec7e32', '#a5a5a5', '#FFFFCD', '#B03060']
hatch = ['//', '..', '--', '\\', 'OO']


def draw_time(models, x_labels, x_values, color, hatch):
    plt.cla()
    x = np.arange(len(x_labels)) * 2
    for i in range(5):
        plt.bar(x + 0.3 * i, x_values[i], color=color[i], hatch=hatch[i], width=0.3, label=models[i], edgecolor='black',
                linewidth=0.2, alpha=1)
    plt.xticks(x + 0.6, x_labels,fontsize=15) # 横坐标字体大小
    plt.yticks(fontsize=15) # 纵坐标字体大小

    ax = plt.gca()
    #ax.set_ylim(2e-1, 2e10)
    ax.set_yscale('log', base=10)
    ## ax.set_yticks([10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2)])

    plt.legend(loc="best", prop={"size": 14})
    plt.xlabel('用户数', fontsize=16, fontproperties='SimHei')
    plt.ylabel('通信代价(字节)', fontsize=16, fontproperties='SimHei')

    plt.title("MNIST communication cost",fontsize=16)

    ax.spines['right'].set_color('black') # 不用的话改成none
    ax.spines['top'].set_color('black')

    plt.rc('axes', axisbelow=True)
    plt.grid(1,linestyle='--', linewidth=0.3, color='gray', alpha=0.2)
    plt.savefig('D:/time.jpg', dpi=700)


draw_time(models, x_labels, x_values, color, hatch)
