import copy

import torch

from basis import local_differential_privacy_library as ldplib
import numpy as np

from basis.local_differential_privacy_library import sparsification, topk_upload, topk_upload_ldp_noise, \
    topk_upload_cifar, topk_upload_ldp_noise_cifar, CMFLcompre, CMFL_topk_mnist, CMFL_topk_ldpNo_mnist, RelevanceCalcu, \
    CMFLcompre_nums, CMFL_topk_mnist_nums, CMFL_topk_ldpNo_mnist_nums, CMFLcompre_numsAndsign, \
    CMFL_topk_mnist_numsAsign, CMFL_topk_ldpNo_mnist_numsAsign, CMFL_topk_ldpNo_mnist_numsDt, \
    CMFL_topk_ldpNo_mnist_numsAsignDt, RelevanceCalcu_numsAsignDt, CMFL_topk_ldpNo_mnist_numsAsignDtcalp


# 固定分配   1
# def average_weights_weighted(gw, gw1):
#     """
#     Returns the average of the weights.
#     """
#     g_w = copy.deepcopy(gw)
#     g_w1 = copy.deepcopy(gw1)
#     for key in g_w.keys():
#         g_w[key] = g_w[key] + g_w1[key] * 0.4
#     return g_w


#两参数的权重分配   4 效果最好
# def average_weights_weighted(gw, gw1, beta):
#     """
#     Returns the average of the weights.
#     """
#     g_w = copy.deepcopy(gw)
#     g_w1 = copy.deepcopy(gw1)
#     for key in g_w.keys():
#         g_w[key] = (1 - beta) * g_w[key] + g_w1[key] * beta
#     return g_w


#两参数的权重分配 + 自定义列表    5
def average_weights_weighted(gw, gw1, beta):
    """
    Returns the average of the weights.
    """
    g_w = copy.deepcopy(gw)
    g_w1 = copy.deepcopy(gw1)
    for key in g_w.keys():
        g_w[key] = (1 - beta) * g_w[key] + g_w1[key] * beta
    return g_w

#  两段式   2
# def average_weights_weighted_adpt(gw,gw1,epo, alp):
#     """
#     Returns the average of the weights.
#     """
#     g_w = copy.deepcopy(gw)
#     g_w1 = copy.deepcopy(gw1)
#     if (epo + 1) < 5:
#         alpha = alp
#     else:
#         alpha = alp / 2
#     for key in g_w.keys():
#         g_w[key] = g_w[key] + g_w1[key] * alpha
#     return g_w


#   三段式
# def average_weights_weighted_adpt(gw, gw1, epo, alp):
#     """
#     Returns the average of the weights.
#     """
#     g_w = copy.deepcopy(gw)
#     g_w1 = copy.deepcopy(gw1)
#     if (epo + 1) < 4:
#         alpha = alp
#     elif 3 < (epo + 1) < 8:
#         alpha = alp / 2
#     else:
#         alpha = alp / 2 / 2
#     for key in g_w.keys():
#         g_w[key] = g_w[key] + g_w1[key] * alpha
#     return g_w


#  自定义式 (自定义列表)   3
def average_weights_weighted_adpt(gw, gw1, alp):
    """
    Returns the average of the weights.
    """
    g_w = copy.deepcopy(gw)
    g_w1 = copy.deepcopy(gw1)
    for key in g_w.keys():
        g_w[key] = g_w[key] + g_w1[key] * alp
    return g_w


def noise_add(w, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = w_locals[i][key].numpy()
            w_locals[i][key] = ldplib.Normalization(w_locals[i][key], np.amax(w_locals[i][key]),
                                                    np.amin(w_locals[i][key]))
            w_locals[i][key] = ldplib.discretization(w_locals[i][key], lower=-1, upper=1, key=key)
            w_locals[i][key] = ldplib.perturbation(value=w_locals[i][key], perturbed_value=-w_locals[i][key],
                                                   epsilon=epsilon)
            w_locals[i][key] = ldplib.eps1p(epsilon=epsilon) * w_locals[i][key]
            w_locals[i][key] = torch.from_numpy(w_locals[i][key])
    return w_locals


# def noise_add1(w): #单层
#     w_locals = copy.deepcopy(w)
#     for i in range(len(w_locals)):
#         w_locals[i]['fc1.weight'] = w_locals[i]['fc1.weight'].numpy()
#         w_locals[i]['fc1.weight'] = ldplib.Normalization(w_locals[i]['fc1.weight'], np.amax(w_locals[i]['fc1.weight']), np.amin(w_locals[i]['fc1.weight']))
#         w_locals[i]['fc1.weight'] = ldplib.discretization(w_locals[i]['fc1.weight'], lower=-1, upper=1)
#         w_locals[i]['fc1.weight'] = ldplib.perturbation(value=w_locals[i]['fc1.weight'], perturbed_value=-w_locals[i]['fc1.weight'], epsilon=epsilon)
#         w_locals[i]['fc1.weight'] = ldplib.eps1p(epsilon=epsilon) * w_locals[i]['fc1.weight']
#         w_locals[i]['fc1.weight'] = torch.from_numpy(w_locals[i]['fc1.weight'])
#     return w_locals

# def noise_add2(w, epsilon): #单用户
#     w_locals = copy.deepcopy(w)
#     for key in w_locals.keys():
#         w_locals[key] = w_locals[key].numpy()
#         w_locals[key] = ldplib.Normalization(w_locals[key], np.amax(w_locals[key]), np.amin(w_locals[key]))
#         w_locals[key] = ldplib.discretization(w_locals[key], lower=-1, upper=1, key=key)
#         w_locals[key] = ldplib.perturbation(value=w_locals[key], perturbed_value=-w_locals[key], epsilon=epsilon)
#         w_locals[key] = ldplib.eps1p(epsilon=epsilon) * w_locals[key]
#         w_locals[key] = torch.from_numpy(w_locals[key])
#     return w_locals

def sparsifi(w, p):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = w_locals[i][key].numpy()
            w_locals[i][key] = sparsification(w_locals[i][key], p, key)
            w_locals[i][key] = torch.from_numpy(w_locals[i][key])
    return w_locals


def topk_weight(w):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = topk_upload(w_locals[i][key], key=key)
    return w_locals


def topk_weight_cifar(w):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = topk_upload_cifar(w_locals[i][key], key=key)
    return w_locals


# ldp - fl #没压缩前的方案
def LDP_noise_add(w, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = w_locals[i][key].numpy()
            # w_locals[i][key] = ldplib.Data_Normalization(w_locals[i][key], np.amax(w_locals[i][key]), np.amin(w_locals[i][key]),c=0,r=0.075)
            w_locals[i][key] = ldplib.Data_perturbation(w_locals[i][key], c=0, r=0.075, epsilon=epsilon, key=key)
            w_locals[i][key] = torch.from_numpy(w_locals[i][key])
    return w_locals





# ldp-noise和top k的结合 ##我们的方案 mnist

def topk_weight_ldp_noise(w, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = topk_upload_ldp_noise(w_locals[i][key], key=key, c=0, r=0.075, epsilon=epsilon[i]) # 测试不同用户有不同的隐私预算
    return w_locals


# cifar

def topk_weight_ldp_noise_cifar(w, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = topk_upload_ldp_noise_cifar(w_locals[i][key], key=key, c=0, r=0.075, epsilon=epsilon)
    return w_locals





# 新的压缩方案 基于层的参数选择 根据梯度符号比较相关性 1
def cmfl(w, global_w, vt):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFLcompre(w_locals[i][key], global_w[key], key=key, vt=vt)
    return w_locals



# 新的压缩方案 基于层的参数选择 根据梯度大小比较相关性 1--
def cmfl_nums(w, global_w, vt, dt):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFLcompre_nums(w_locals[i][key], global_w[key], key=key, vt=vt, dt=dt)
    return w_locals


# 新的压缩方案 基于层的参数选择 根据 梯度符号（方向）和大小（值）的 结合 方式来比较相关性 1**
def cmfl_numsAsign(w, global_w, vt, dt):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFLcompre_numsAndsign(w_locals[i][key], global_w[key], key=key, vt=vt, dt=dt)
    return w_locals





# -------------------------------------------------------------------------

# cmfl和top-k的压缩结合 mnist数据集 根据梯度符号（方向）比较 2
def cmflwithtopk_mnist(w, global_w, vt):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_mnist(w_locals[i][key], global_w[key], key=key, vt=vt)
    return w_locals

# cmfl和top-k的压缩结合 mnist数据集 根据梯度大小（值）比较 2--
def cmflwithtopk_mnist_nums(w, global_w, vt, dt):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_mnist_nums(w_locals[i][key], global_w[key], key=key, vt=vt,dt=dt)
    return w_locals


# cmfl和top-k的压缩结合 mnist数据集 梯度符号（方向）和大小（值）的 结合 方式来比较相关性 2**
def cmflwithtopk_mnist_numsAsign(w, global_w, vt, dt):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_mnist_numsAsign(w_locals[i][key], global_w[key], key=key, vt=vt,dt=dt)
    return w_locals



# --------------------------------------------------------------------------



# cmfl-topk-ldp_noise 三者的结合 新的创新点 根据梯度符号(方向)比较 3
def cmfl_topk_ldp_mnist(w, global_w, vt, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_ldpNo_mnist(w_locals[i][key], global_w[key], key=key, vt=vt, c=0, r=0.075, epsilon=epsilon)
    return w_locals


# cmfl-topk-ldp_noise 三者的结合 新的创新点 根据梯度大小(值)比较 3--
def cmfl_topk_ldp_mnist_nums(w, global_w, vt,dt, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_ldpNo_mnist_nums(w_locals[i][key], global_w[key], key=key, vt=vt, dt=dt, c=0, r=0.075, epsilon=epsilon)
    return w_locals



# cmfl-topk-ldp_noise 三者的结合 新的创新点 梯度符号（方向）和大小（值）的 结合 方式来比较相关性 3**
def cmfl_topk_ldp_mnist_numsAsign(w, global_w, vt,dt, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_ldpNo_mnist_numsAsign(w_locals[i][key], global_w[key], key=key, vt=vt, dt=dt, c=0, r=0.075, epsilon=epsilon)
    return w_locals



# --------------------------------------------------------------------------

# cmfl-topk-ldp_noise 三者的结合 新的创新点 根据梯度大小(值)比较 其中dt不再是自己设置的,而是取每层的平均值 4--
def cmfl_topk_ldp_mnist_numsDt(w, global_w, vt, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_ldpNo_mnist_numsDt(w_locals[i][key], global_w[key], key=key, vt=vt, c=0, r=0.075, epsilon=epsilon)
    return w_locals


# cmfl-topk-ldp_noise 三者的结合 新的创新点 梯度符号（方向）和大小（值）的 结合 方式来比较相关性 其中dt不再是自己设置的,而是取每层的平均值 4**
def cmfl_topk_ldp_mnist_numsAsignDt(w, global_w, vt, epsilon):
    w_locals = copy.deepcopy(w)
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key] = CMFL_topk_ldpNo_mnist_numsAsignDt(w_locals[i][key], global_w[key], key=key, vt=vt, c=0, r=0.075, epsilon=epsilon[i])
    return w_locals


# 5-- 在 4** 的基础上统计上传的参数数量（在实验中对比压缩的参数量）
def cmfl_topk_ldp_mnist_numsAsignDtcalp(w, global_w, vt, epsilon, p):
    w_locals = copy.deepcopy(w)
    User_parametersList = [0 for x in range(len(w_locals))]
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            w_locals[i][key], nums_para = CMFL_topk_ldpNo_mnist_numsAsignDtcalp(w_locals[i][key], global_w[key], key=key, vt=vt, c=0, r=0.075, epsilon=epsilon[i], p=p)
            User_parametersList[i] = User_parametersList[i] + nums_para # 计算每个用户上传的参数量
    return w_locals, User_parametersList



#  -----------为用户根据相关性分配不同的隐私预算------------

# 计算每个用户的’不‘相关性 存储在列表里

def irrelevanceCal(w, global_w):
    w_locals = copy.deepcopy(w)
    g_w = copy.deepcopy(global_w)
    relevanceList = [0 for x in range(len(w_locals))]
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            relevanceList[i] = relevanceList[i] + RelevanceCalcu(w_locals[i][key], g_w[key], key=key) # 每一层的相关性 叠加
        relevanceList[i] = 1 - (relevanceList[i] / 8.0)
    return relevanceList


# 计算每个用户的’不‘相关性 存储在列表里 相关性度量是基于梯度方向（符号）和大小（值）
# 最终的隐私预算分配方案
def irrelevanceCal_numsAsign(w, global_w):
    w_locals = copy.deepcopy(w)
    g_w = copy.deepcopy(global_w)
    relevanceList_numsAsign = [0 for x in range(len(w_locals))]
    for i in range(len(w_locals)):
        for key in w_locals[i].keys():
            relevanceList_numsAsign[i] = relevanceList_numsAsign[i] + RelevanceCalcu_numsAsignDt(w_locals[i][key], g_w[key], key=key) # 每一层的相关性 叠加
        relevanceList_numsAsign[i] = 1 - (relevanceList_numsAsign[i] / 8.0)
    return relevanceList_numsAsign