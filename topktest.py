import numpy as np
import pandas as pd
import torch

# a = torch.randn(5, 6)
# print(a)
# b, index = torch.topk(a.abs(), 3)
# print('--------------')
# print('b=', b)
# print('--------------')
# print(index)
# c = a.flatten()
# print('c=', c)
# d, index1 = torch.sort(c)
# print('--------------')
# print('d=', d)
# print('--------------')
# print('index1=', index1)
# print(index1[-2:])
#
# # reshape
# e = torch.reshape(c, (5, 6))
# print('e = ', e)
#
# # d,index1 = torch.topk(c,3)
# # print('--------------')
# # print('d=',d)
# # print('--------------')
# # print('index=', index1)
# # print(index1[-2:])
# print(c[index1[16]])
#
# aa = torch.randn(10)
# index22 = torch.topk(aa, 5)[1]
# print('aa = ', aa)
# # print('bb = ',bb)
# print('index22 = ', index22)
# print('afterselection = ', aa[index22])
#
# test = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
# test1 = torch.tensor([[[2, 3, 4], [5, 6, 7]], [[2, 3, 4], [5, 6, 7]]])
# print(test.shape)
# test2 = test + test1
# print('两数相加的结果是：', test2)
#
# print('相乘：', 5 * test)
#
# list1 = [1, 2, 3, 55]
# print("list 1:", list1[3])
#
# for i in range(10):
#     print(i)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# flag = torch.cuda.is_available()
# print(torch.cuda.current_device()) # gpu 索引
# print(torch.cuda.get_device_name(0)) # gpu 名字
# print(torch.cuda.device_count()) # GPU 数量
# print(flag)
# print(device)
#
#
# list2 = [1, 3, 5]
# list3 = [2, 4, 6]
# list4 = list2 + list3
# print("list2 + list3 = ", list4)
#
# temp2 = np.array(list2)  # list 转 numpy
# temp3 = np.array(list3)
#
#
# temp4 = temp2 + temp3
# print("temp2 + temp3 = ",temp4)
# print(type(temp4))
#
# temp5 = temp4.tolist() # numpy 转 list
# print("temp2 + temp3 = ",temp5)
# print(type(temp5))
#
#
# a = torch.tensor([1,5,6,29,60])
# b = torch.zeros(5)
# c = a * 0.2
# print("c = ",c)
# print("a + c =" ,a+c)



# array = np.arange(12).reshape(3,4)
# print(array)
# print(array[:2])
# print(array[:2][1:])  # 行
# print(array[:2, 1:])  # 列

df = pd.DataFrame(pd.read_excel("C:/Users/vincent/Desktop/戴师兄数据分析启蒙课_Excel练习.xlsx"))
print(df)

# df['total'] = [25,26,27,28]
#print(df)
#print(df['年龄'].sum())
print("-----------------------------------")
print(df['品牌名称'])
print("-----------------------------------")
print(df[['品牌名称','品牌ID','cpc访问量']])
print("-----------------------------------")
# print(df.groupby('品牌名称')['品牌名称'].describe())
print("-----------------------------------")
print(df.groupby('品牌名称').sum()) # 输出所有字段
print("-----------------------------------")
print(df.groupby('品牌名称').sum()[['GMV','cpc访问量']]) # 输出多个字段
print("-----------------------------------")
print(df.groupby('品牌名称').sum()['GMV']) # 输出单个字段
print("-----------------------------------")   #两种都可以 下面这种可以取名字
print(df.groupby('品牌名称').agg(总的gmv = ('GMV','sum')))# 输出单个字段
print("-----------------------------------")
print(df.groupby('品牌名称').agg(总的gmv = ('GMV','sum'),总的cpc访问量 = ('cpc访问量','sum')))# 输出多个字段
# print(df['GMV'].sum())
print("-----------------------------------")
print(df.query('cpc曝光量>3000')['cpc曝光量']) # 单
print("-----------------------------------")
print(df.query('cpc曝光量>3000')[['cpc曝光量','品牌ID']]) # 多
print("-----------------------------------")
print(df.sort_values(by='cpc曝光量',ascending=True))
print("-----------------------------------")
print(df.loc[df['品牌ID']==4636,['品牌ID','门店ID','cpc曝光量']]) # select 门店ID,cpc曝光量 where 品牌ID = 4636
print("-----------------------------------")
print(df.groupby('门店ID').agg(cpc曝光量总和 =('cpc曝光量','sum')).query('cpc曝光量总和>100000').sort_values(by='cpc曝光量总和',ascending=True))