import numpy as np
import pandas as pd
import torch
import sklearn

df = pd.DataFrame(pd.read_excel('air_data.xlsx'))
# print(df)

# df = pd.DataFrame(pd.read_csv('air_data.csv'))
# print(df)
print(df[df.WORK_CITY.isnull()])
#a = ['MEMBER_NO','FFP_DATE','FIRST_FLIGHT_DATE','FFP_TIER','WORK_CITY','WORK_PROVINCE','WORK_COUNTRY','AGE','LOAD_TIME','FLIGHT_COUNT','BP_SUM','EP_SUM_YR_1','EP_SUM_YR_2','SUM_YR_1','SUM_YR_2','SEG_KM_SUM','WEIGHTED_SEG_KM','LAST_FLIGHT_DATE','AVG_FLIGHT_COUNT','AVG_BP_SUM','BEGIN_TO_FIRST','LAST_TO_END','AVG_INTERVAL','MAX_INTERVAL','ADD_POINTS_SUM_YR_1']
#print(df.groupby('WORK_CITY').agg(最大值 =('BP_SUM','max'),最大值1 =('EP_SUM_YR_2','max'),最大值2 =('SUM_YR_1','max'),最大值3 =('SUM_YR_2','max'),最大值4 =('SEG_KM_SUM','max'),最大值5 =('WEIGHTED_SEG_KM','max'),最大值6 =('AVG_BP_SUM','max'),最大值7 =('P1Y_BP_SUM','max'),最大值8 =('L1Y_BP_SUM','max'),最大值9 =('EP_SUM','max'),最大值10 =('Eli_Add_Point_Sum','max'),最大值11 =('L1Y_ELi_Add_Points','max'),最大值12 =('L1Y_Points_Sum','max')))
