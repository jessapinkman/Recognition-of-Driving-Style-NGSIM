
# 1.数据筛选：
# a车辆选择为小轿车 vehicle_class=2,剔除其他车辆
# b剔除没有前车的车辆数据
# c剔除剔除7->6或者6->8的换道数据
# d选取一个时间周期内的完整变道数据
# 2.因为样本数据存在系统误差和测量误差的噪声，先做平滑处理：中值滤波算法（时间窗长度5）

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
pd.set_option('display.max_columns', None)



#抽取出US101数据，定义应该函数按照全局时间进行筛选，在此处road默认为i-80
def cutbyRoad(df=None, road=None):
    '''
    :param df: 打开后文件
    :param road: 路段名称
    :return: 切路df,按照全局时间排序
    '''
    road_df = df[df['Location'] == road]
    return road_df.sort_values(by='Global_Time', ascending=True)

#原数据集的单位，时间是ms，长度单位全是ft。给它转换一下：
def unitConversion(df):
    '''
    转换后长度单位为m，时间单位为0.1秒
    :param df: 被转换df
    :return: 转换后df
    '''
    ft_to_m = 0.3048
    df['Global_Time'] = df['Global_Time'] / 100
    for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_length", "v_Width"]:
        try:
            df[strs] = df[strs] * ft_to_m
        except:
            df[strs] = df[strs].apply(lambda x: float(x.replace(',', ''))) * ft_to_m
    df["v_Vel"] = df["v_Vel"] * ft_to_m * 3.6
    df["v_Acc"] = df["v_Acc"] * ft_to_m
    return df

#计算每辆车的变道次数
def LaneChangeCount(df):
    counts = df['Vehicle_ID'].value_counts()
    return counts


#1111开始处理数据
with open("src/data_param.json", "r") as f:
    conf = json.load(f)


#打开i-80数据集有用的列,并根据'Vehicle_ID', 'Frame_ID'升序排列
data = pd.read_csv(conf["data_source"], usecols=conf["useCols"])
data.sort_values(by=['Vehicle_ID', 'Frame_ID'], axis=0, ascending=[True, True], inplace=True)
data.reset_index(drop=True, inplace=True)
data.to_csv(r'data/i-80_ascending.csv', index=None, encoding='utf-8_sig')
print("done")

data = pd.read_csv('data/i-80_ascending.csv')
data['Lane_ID1'] = data['Lane_ID'].shift(-1) #上移
data['LaneChange'] = data['Lane_ID1'] - data['Lane_ID']
data_LC1 = data[((data['LaneChange']==1)|(data['LaneChange']==-1)) & (data['Vehicle_ID']\
                                ==data['Vehicle_ID'].shift(-1)) & (-(((data['Lane_ID']==8)&(data['Lane_ID1']\
                                ==7))|(data['Lane_ID']==7)&(data['Lane_ID1']==8))) & (data['v_Class']==2)]
data_LC1 = data[((data['LaneChange']==1)|(data['LaneChange']==-1)) & (data['Vehicle_ID']\
                                ==data['Vehicle_ID'].shift(-1)) & (-(((data['Lane_ID']==6)&(data['Lane_ID1']\
                                ==7))|(data['Lane_ID']==7)&(data['Lane_ID1']==6))) & (data['v_Class']==2)]
data_LC2 = data_LC1[-(((data_LC1['Frame_ID'].shift(-1)-data_LC1['Frame_ID'])<100) & (data_LC1['Vehicle_ID']\
                                      ==data_LC1['Vehicle_ID'].shift(-1)))]
data_LC2 = data_LC2.reset_index()
counts = LaneChangeCount(data_LC2)
print(data_LC2.shape) #[387 rows x 22 columns]


#2222找到变道前后100帧的数据索引
index1 = list(data_LC2['index'])
for i in range(len(index1)):
    for j in range(index1[i]-49,index1[i]+51):
        index1.append(j)
del index1[:387] #此处删除的行数需要人为查看，有待改进
index2 = sorted(index1)
print(len(index2)) #59680条变道数据
#在data中拿索引找到160帧变道的数据集data_LC
data_LC = pd.DataFrame(columns=['Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y','Global_X','Global_Y',
                        'v_Length','v_Width','v_Class','v_Vel','v_Acc','Lane_ID','Preceding','Following','Space_Headway','Time_Headway'])
for k in range(len(index2)):
    data_LC.loc[k]=data.loc[index2[k]]
data_LC.to_csv(r'2',index=None,encoding='utf-8_sig')
print("done")


# 3333删除不属于变道车的数据
data1 = pd.read_csv('2.csv')
index_V_ID = list(data_LC2['Vehicle_ID'])
for i in range(data1.index.size-1):
    if np.array(data1.loc[[i], ['Vehicle_ID']]).tolist()[0][0] != index_V_ID[math.floor(i / 160)]:
        data1 = data1.drop(i)

index3 = list(data1['Vehicle_ID'])
index4 = sorted(list(set(index3))) #去重变道车辆
for num in index4:
    if (index3.count(num) % 160) != 0:
        data1 = data1[~data1['Vehicle_ID'].isin([num])]

data1.to_csv(r'data/i-80_LC_16s.csv',index=None,encoding='utf-8_sig')

data = pd.read_csv('data/i-80_LC_16s.csv')
# print(data.shape) #(45120, 18)
#绘画出行车轨迹
t = np.arange(0,16,0.1)
for V_ID in list(set(list(data['Vehicle_ID']))):
    if list(data['Vehicle_ID']).count(V_ID) == 160:
        b = data[data['Vehicle_ID']==V_ID]
        fig = plt.figure(1,dpi=250)
        ax = plt.subplot(111)
        plt.sca(ax)
        plt.plot(t,b['Local_X'],linewidth=0.2)
plt.title('trajectory')
plt.show()


#转换时间、长度单位
unitConversion(data)

#剔除数据
data[data['v_Class'] == 1]
data[data['v_Class'] == 3]
data[data["Preceding"] == 0]

#怎么删除进出匝道的车辆数据 6\7\8
#根据车辆划分






