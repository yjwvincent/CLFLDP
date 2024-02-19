import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

from basis import local_differential_privacy_library as ldplib
import numpy as np

from src.models import CNNCifar
from src.options import args_parser

epsilon = 0.0001
def add_laplace_noise(data_list, μ=0, b=1):
    laplace_noise = np.random.laplace(μ, b, len(data_list)) # 为原始数据添加μ为0，b为1的噪声
    return laplace_noise + data_list
args = args_parser()
model1 = CNNCifar(args)
model1.train()
stact_dic = model1.state_dict()



# Find total parameters and trainable parameters
# total_params = sum(p.numel() for p in model1.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(
#     p.numel() for p in model1.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')

# for name,param in model1.named_parameters():
#   print(name,param)

#print(stact_dic)
#print(model1.fc1.weight.grad) #查看梯度
# print(stact_dic['fc1.weight'])
# print(stact_dic['fc1.weight'].numpy())
weight = stact_dic['fc1.weight'].numpy()
#weight1 = stact_dic['pool.weight'].numpy()
#print(weight1)
print(weight.shape)
weight_afterNor = ldplib.Normalization(weight, np.amax(weight), np.amin(weight))  #归一化【-1,1】
#print(np.amax(weight))
# print(weight)
# print(weight_afterNor)
#weight_afterdiscret = ldplib.discretization(value=weight_afterNor, lower=-1, upper=1)# 离散化
#print(weight_afterdiscret)
weight_pertubation = ldplib.perturbation(value=weight_afterNor, perturbed_value=-weight_afterNor, epsilon=epsilon)
#print(weight_pertubation) #扰动后的数据
weight_finalpertuba = ldplib.eps1p(epsilon=epsilon) * weight_pertubation #扰动后的数值型数据
#print(weight_finalpertuba)

print('加噪前的weight:\n',weight)
print('加噪后的weight:\n', weight_finalpertuba)
print('加噪前的weight 均值:\n', np.mean(weight))
print('加噪后的weight 均值:\n', np.mean(weight_finalpertuba))
print('误差: \n', np.fabs(np.mean(weight)-np.mean(weight_finalpertuba)))


#print(weight[0][1])
#print(type(weight))
# print('加噪前：', weight[0])
# noise_weight = add_laplace_noise(weight[0])
# print('加噪后：', noise_weight)

# a = np.array([[1,2,3], [4,5, 6]], dtype=int)
# p = (a-1)/2
# print(p)