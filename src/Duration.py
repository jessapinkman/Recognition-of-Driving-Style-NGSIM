import pandas as pd
import numpy as np
import scipy.signal as signal
import  process_data
#!!!找一下是否存在一直不变道的数据

data = pd.read_csv(r'data/i-80_LC_16s.csv')
data = data.drop(['Total_Frames','Global_Time','Global_X','Global_Y','v_Length','v_Class','Preceding','Following'],axis=1)

#111平滑处理并得到Lateral_Acc\Vel
data['Smoothed_Local_X/m']=0
data['Smoothed_Local_Y/m']=0
data['Smoothed_v_Vel/(km/h)']=0
data['Smoothed_v_Acc/(m/s^2)']=0
data['Smoothed_Space_Headway/m']=0
data['Smoothed_Time_Headway/s']=0
data['Lateral_Vel/(m/s)']=None
data['Lateral_Acc/(m/s^2)']=None
for i in range(int(data.index.size/160)):
    sub_data = data.loc[(i*160):(i*160+159)]
    sub_data['Smoothed_Local_X/m'] = signal.savgol_filter(sub_data['Local_X'],21,3)*0.3048
    sub_data['Smoothed_Local_Y/m'] = signal.savgol_filter(sub_data['Local_Y'],21,3)*0.3048
    sub_data['Smoothed_v_Vel/(km/h)'] = signal.savgol_filter(sub_data['v_Vel'],21,3)*0.3048*3.6
    sub_data['Smoothed_v_Acc/(m/s^2)'] = signal.savgol_filter(sub_data['v_Acc'],21,3)*0.3048
    sub_data['Smoothed_Space_Headway/m'] = signal.savgol_filter(sub_data['Space_Headway'],21,3)*0.3048
    sub_data['Smoothed_Time_Headway/s'] = signal.savgol_filter(sub_data['Time_Headway'],21,3)
    sub_data['Lateral_Vel/(m/s)'] = sub_data['Smoothed_Local_X/m'].diff()*10
    sub_data['Lateral_Acc/(m/s^2)'] = sub_data['Lateral_Vel/(m/s)'].diff()*10
    for j in range(160):
        data.loc[(i*160+j)] = sub_data.iloc[j]


counts = pd.read_csv('data/i-80_LC_counts.csv')
data = pd.merge(left=data, right=counts, how='left', on='Vehicle_ID')
data = data.drop(columns='Unnamed: 0', axis=1)
data.to_csv(r'1.csv', index=None, encoding='utf-8_sig')



# 去除v_Lateral/(m/s) Lateral_Acc(m/s^2)为null的值
for i in range(data.index.size-1):
    if pd.isnull(data.loc[i, 'Lateral_Vel/(m/s)']):
        data.loc[i, 'Lateral_Vel/(m/s)'] = data.loc[i+1, 'Lateral_Vel/(m/s)']

for i in range(data.index.size-1):
    if pd.isnull(data.loc[i, 'Lateral_Acc/(m/s^2)']):
        data.loc[i, 'Lateral_Acc/(m/s^2)'] = data.loc[i+1, 'Lateral_Acc/(m/s^2)']
for i in range(int(data.index.size / 160)):
    data.loc[i*160, 'Lateral_Acc/(m/s^2)'] = data.loc[i*160+1, 'Lateral_Acc/(m/s^2)']


#222删除没有前车的数据（可以用space_Headway或者preceding
for i in range(int(data.index.size / 160)):
    for j in range(160):
        if(data.loc[i*160+j, 'Space_Headway']) == 0:
            data.drop(labels=range(i*160, (i+1)*160))
# 重新为车辆id赋值
# for i in range(int(data.index.size/160)):
#     for j in range(160):
#         data.loc[(i*160+j),'Vehicle_ID'] = i

#333删除速度为0或者Headway出现异常的车辆
for i in range(int(data.index.size / 160)):
    for j in range(160):
        if (data.loc[i*160+j, 'v_Vel'] == 0) | (data.loc[i*160+j, 'Smoothed_Space_Headway/m'] < 0) | \
            (data.loc[i*160+j, 'Smoothed_Time_Headway/s'] < 0):
            data.drop(labels=range(i*160, (i+1)*160))
# 重新为车辆id赋值
# for i in range(int(data.index.size/160)):
#     for j in range(160):
#         data.loc[(i*160+j),'Vehicle_ID'] = i

data.to_csv(r'2.csv', index=None,encoding='utf-8_sig') #(45120, 18)

# data.to_csv(r'data/i-80_LC_16s_Smoothed.csv', index=None,encoding='utf-8_sig') #(45120, 18)




