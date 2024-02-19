import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(15, 5), dpi=70)  # 设置图像大小 FMNIST

x_labels = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9', '1'] #T=100 N=100 different p
x_values = [
    [289280000 * 4, 289280000 * 4, 289280000 * 4, 289280000 * 4, 289280000 * 4, 289280000 * 4, 289280000 * 4, 289280000 * 4, 289280000 * 4, 289280000 * 4],
    [24906627 * 4, 52066913 * 4, 82611952 * 4, 98073805 * 4, 132075592 * 4, 138019437 * 4, 175018393 * 4, 198749511 * 4, 226936657 * 4, 289280000 * 4],
    [23510571 * 4, 49216692 * 4, 78519358 * 4, 95014386 * 4, 121406732 * 4, 130634882 * 4, 168185992 * 4, 179394627 * 4, 211692032 * 4, 272047992 * 4] #我们的方案，即便压缩率为1时，也是可能会压缩部分参数（不上传全部参数）（两层压缩）
]
models = ['FedAvg', 'Fedsel ($\epsilon=5.5$)', 'GLCFL (Q=0.7,$\epsilon=2$)'] #cmfl没有压缩率这东西
color = ['#808A87', '#F0E68C', '#FF6347', '#802A2A', '#87CEEB']
hatch = ['//', '--', '\\']


def draw_time(models, x_labels, x_values, color, hatch):
    plt.cla()
    x = np.arange(len(x_labels)) * 2
    for i in range(3):
        plt.bar(x + 0.3 * i, x_values[i], color=color[i], hatch=hatch[i], width=0.3, label=models[i], edgecolor='black',
                linewidth=0.2, alpha=1)
    plt.xticks(x + 0.6, x_labels,fontsize=15) # 横坐标字体大小
    plt.yticks(fontsize=15) # 纵坐标字体大小

    ax = plt.gca()
    #ax.set_ylim(2e-1, 2e10)
    ax.set_yscale('log', base=10)
    ## ax.set_yticks([10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2)])

    plt.legend(loc="best", prop={"size": 14})
    plt.xlabel('压缩率', fontsize=16, fontproperties='SimHei')
    plt.ylabel('通信代价(字节)', fontsize=16, fontproperties='SimHei')

    plt.title("FMNIST communication cost",fontsize=16)

    ax.spines['right'].set_color('black') # 不用的话改成none
    ax.spines['top'].set_color('black')

    plt.rc('axes', axisbelow=True)
    plt.grid(1,linestyle='--', linewidth=0.3, color='gray', alpha=0.2)
    plt.savefig('D:/time8.jpg', dpi=700)


draw_time(models, x_labels, x_values, color, hatch)
