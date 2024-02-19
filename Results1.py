import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

plt.figure(figsize=(15, 5), dpi=70)  # 设置图像大小 MNIST

x_labels = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9', '1'] #T=100 N=100 different p
x_values = [
    [218400000 * 4, 218400000 * 4, 218400000 * 4, 218400000 * 4, 218400000 * 4, 218400000 * 4, 218400000 * 4, 218400000 * 4, 218400000 * 4, 218400000 * 4],
    [21840000 * 4, 43680000 * 4, 65520000 * 4, 87360000 * 4, 109200000 * 4, 131040000 * 4, 152880000 * 4, 174720000 * 4, 196560000 * 4, 218400000 * 4], # 这里的数据有点乱，到时候更改一下2-3行的数据 介于FedAvg和Fedsel之间 跟表的数据对应上
    [19294301 * 4, 38946034 * 4, 62287531 * 4, 85124934 * 4, 107103581 * 4, 128437990 * 4, 140248735 * 4, 168347235 * 4, 192372954 * 4, 218400000 * 4],
    [17289675 * 4, 32341379 * 4, 56813801 * 4, 73103407 * 4, 85163954 * 4, 89676227 * 4, 102764818 * 4, 136912882 * 4, 175015811 * 4, 218400000* 4],
    [13527871 * 4, 27961321 * 4, 48714591 * 4, 66922047 * 4, 75108525 * 4, 84613958 * 4, 95274637 * 4, 115251947 * 4, 152479417 * 4, 174324049 * 4] #我们的方案，即便压缩率为1时，也是可能会压缩部分参数（不上传全部参数）（两层压缩）
]
models = ['FedAvg', 'PQ', 'QSGD', 'Fedsel(TOP-K)', 'Ours Q=0.7'] #cmfl没有压缩率这东西
color = ['#808A87', '#F0E68C', '#FF6347', '#802A2A', '#87CEEB']
hatch = ['', '', '', '', '']


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
    plt.xlabel('Compression rate', fontsize=16)
    plt.ylabel('Total communication cost (Byte)', fontsize=16)

    plt.title("MNIST communication cost",fontsize=16)

    ax.spines['right'].set_color('black') # 不用的话改成none
    ax.spines['top'].set_color('black')

    plt.rc('axes', axisbelow=True)
    plt.grid(1,linestyle='--', linewidth=0.3, color='gray', alpha=0.2)
    plt.savefig('D:/time1.jpg', dpi=700)


draw_time(models, x_labels, x_values, color, hatch)
