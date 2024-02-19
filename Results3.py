import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

plt.figure(figsize=(15, 5), dpi=70)  # 设置图像大小 cifar-10

x_labels = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9', '1'] #T=100 N=100 different p
x_values = [
    [618700000 * 4, 618700000 * 4, 618700000 * 4, 618700000 * 4, 618700000 * 4, 618700000 * 4, 618700000 * 4, 618700000 * 4, 618700000 * 4, 618700000 * 4],
    [61870000 * 4, 123740000 * 4, 185610000 * 4, 247480000 * 4, 309350000 * 4, 371220000 * 4, 433090000 * 4, 494960000 * 4, 556830000 * 4, 618700000 * 4],
    [59247391 * 4, 103528561 * 4, 167360319 * 4, 225581503 * 4, 281649158 * 4, 350539577 * 4, 419428592 * 4, 478190691 * 4, 540829669 * 4, 618700000 * 4],
    [56102859 * 4, 95472126 * 4, 138567201 * 4, 200739562 * 4, 256818614 * 4, 327471052 * 4, 391058291 * 4, 447294013 * 4, 511948591 * 4, 618700000 * 4],
    [54093611 * 4, 91103951 * 4, 120471593 * 4, 183103951 * 4, 231957201 * 4, 307930471 * 4, 371003731 * 4, 429106715 * 4, 499163952 * 4, 556294619 * 4] #我们的方案，即便压缩率为1时，也是可能会压缩部分参数（不上传全部参数）（两层压缩）
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

    plt.title("CIFAR10 communication cost",fontsize=16)

    ax.spines['right'].set_color('black') # 不用的话改成none
    ax.spines['top'].set_color('black')

    plt.rc('axes', axisbelow=True)
    plt.grid(1,linestyle='--', linewidth=0.3, color='gray', alpha=0.2)
    plt.savefig('D:/time3.jpg', dpi=700)


draw_time(models, x_labels, x_values, color, hatch)
