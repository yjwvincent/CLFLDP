import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(15, 5), dpi=70)  # 设置图像大小 # FASHION-MNIST

x_labels = ['10','20','30','40','50','60','70','80','90', '100'] #T=100 p=0.5 different N
x_values = [
    [28928000 * 4, 57856000 * 4, 86784000 * 4, 115712000 * 4, 144640000 * 4, 173568000 * 4, 202496000 * 4, 231424000 * 4, 260352000 * 4, 289280000 * 4],
    [14464000 * 4, 28928000 * 4, 43392000 * 4, 57856000 * 4, 72320000 * 4, 86784000 * 4, 101248000 * 4, 115712000 * 4, 130176000 * 4, 144640000 * 4], # PQ = 217406192 QSGD =181196637
    [12710462 * 4, 26193174 * 4, 42016194 * 4, 55125719 * 4, 70184728 * 4, 84104817 * 4, 98248183 * 4, 111238812 * 4, 118028461 * 4, 138401993 * 4],
    [11208491 * 4, 24396481 * 4, 40274152 * 4, 53048102 * 4, 68201838 * 4, 81894012 * 4, 96201847 * 4, 108402864 * 4, 114903755 * 4, 121957002 * 4], # Q=0.6 115710372
    [9957194 * 4, 21720475 * 4, 38102745 * 4, 51074291 * 4, 66261947 * 4, 78036822 * 4, 92015882 * 4, 101669103 * 4, 104499142 * 4, 111038762 * 4]   # Q=0.8 105846199
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

    plt.title("FMNIST communication cost",fontsize=16)

    ax.spines['right'].set_color('black') # 不用的话改成none
    ax.spines['top'].set_color('black')

    plt.rc('axes', axisbelow=True)
    plt.grid(1,linestyle='--', linewidth=0.3, color='gray', alpha=0.2)
    plt.savefig('D:/time4.jpg', dpi=700)


draw_time(models, x_labels, x_values, color, hatch)
