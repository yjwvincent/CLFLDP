import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(15, 5), dpi=70)  # 设置图像大小 cifar-10

x_labels = ['10','20','30','40','50','60','70','80','90', '100'] #T=100 p=0.5 different N
x_values = [
    [61870000 * 4, 123740000 * 4, 185610000 * 4, 247480000 * 4, 309350000 * 4, 371220000 * 4, 433090000 * 4, 494960000 * 4, 556830000 * 4, 618700000 * 4],
    [30935000 * 4, 61870000 * 4, 92805000 * 4, 123740000 * 4, 154675000 * 4, 185610000 * 4, 216545000 * 4, 247480000 * 4, 278415000 * 4, 309350000 * 4], # PQ= 482037191 QSGD = 417037598
    [28234391 * 4, 55034159 * 4, 86129741 * 4, 102945613 * 4, 134619357 * 4, 163590129 * 4, 193604613 * 4, 227551046 * 4, 247341053 * 4, 286359137 * 4],
    [26459174 * 4, 52714576 * 4, 84246193 * 4, 100357153 * 4, 124540174 * 4, 144624697 * 4, 179357392 * 4, 201248547 * 4, 229741750 * 4, 261942679 * 4], # Q=0.6 252749882
    [24691646 * 4, 48947510 * 4, 80175296 * 4, 94454571 * 4, 112406729 * 4, 129540167 * 4, 165926608 * 4, 187571043 * 4, 201658510 * 4, 238560184 * 4] # Q=0.8 212700472
]
models = ['FedAvg', 'Fedsel (p=0.5,$\epsilon=5.5$)', 'CMFL (Q=0.7)', 'COFEL (Q=0.7,$\epsilon=5.5$)', 'GLCFL (p=0.5,Q=0.7,$\epsilon=3$)']
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

    plt.title("CIFAR10 communication cost",fontsize=16)

    ax.spines['right'].set_color('black') # 不用的话改成none
    ax.spines['top'].set_color('black')

    plt.rc('axes', axisbelow=True)
    plt.grid(1,linestyle='--', linewidth=0.3, color='gray', alpha=0.2)
    plt.savefig('D:/time2.jpg', dpi=700)


draw_time(models, x_labels, x_values, color, hatch)
