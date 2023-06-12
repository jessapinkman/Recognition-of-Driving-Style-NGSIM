import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import process_data as f
# 每一个时间戳+每一辆车的id：速度、加速度  输出二维图 heatmap

#拿时间和速度数据
df = pd.read_csv('data/us-101.csv', usecols=['v_Vel', 'v_Acc'])
ft_to_m = 0.3048
df["v_Vel"] = df["v_Vel"] * ft_to_m
df["v_Acc"] = df["v_Acc"] * ft_to_m
bins = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3] #对加速度划分区间
x1, x2 = pd.cut(x=df['v_Acc'], bins=bins, right=True, retbins=True)
df['v_Vel'] = df['v_Vel'].round(0).astype(int)
df['v_Acc'] = x1
# df[4098933 rows x 2 columns] 加速度和速度的dataframe

#根据加速度和速度统计频次并归一化
a = df.groupby(['v_Vel', 'v_Acc']).size()
a = pd.DataFrame(a) #360*1
a.reset_index(inplace=True) #还原索引列
b = a.set_index(['v_Vel', 'v_Acc']) #drop=True 删除设置索引的列 360*1
b = b.unstack() #将行中的数据透视到列 unstack将列中的数据透视到行  30*12
b.columns = a['v_Acc'].unique().tolist()
# b = b.sort_index(axis=1, ascending=True)
b = (b-b.min())/(b.max()-b.min()) #b[30 rows x 12 columns]  速度是从0->95.30，加速度是-11->11.20,有的数据没有速度或者加速度

sns.set()
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示，必须放在sns.set之后
uniform_data = b.values #设置二维矩阵
f, ax = plt.subplots(figsize=(30, 12)) #设置像素也就是网格数

#heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
#参数annot=True表示在对应模块中注释值
#参数linewidths是控制网格间间隔
#参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
#参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
sns.heatmap(b, ax=ax,vmin=0,vmax=1,cmap='YlOrRd',annot=None,linewidths=0.1,cbar=True)

ax.set_title('us-101') #plt.title('热图'),均可设置图片标题
ax.set_ylabel('v_Vel')  #设置纵轴标签
ax.set_xlabel('v_Acc')  #设置横轴标签

#设置坐标字体方向，通过rotation参数可以调节旋转角度
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')

plt.show()
