import pandas as pd
import pickle
import numpy as np
import operator as op
# import Timer
pd.set_option("display.max_columns", None)
key = 'i-80'
file_name = f"vldfs-{key}.pkl"
print(file_name)

with open('data/vldfs-i-80.pkl', "rb") as infile:
    vldfs = pickle.load(infile)
# print("vldfs loaded")
# print(len(vldfs))

df = pd.read_csv(f'data/{key}.csv', usecols=['Vehicle_ID',
                                             'Frame_ID',
                                             'Total_Frames',
                                             'Global_Time',
                                             'Local_X',
                                             'Local_Y',
                                             'Global_X',
                                             'Global_Y',
                                             'v_length',
                                             'v_Width',
                                             'v_Class',
                                             'v_Vel',
                                             'v_Acc',
                                             'Lane_ID',
                                             'Preceding',
                                             'Following',
                                             'Space_Headway',
                                             'Time_Headway',
                                             'Location'
                                             ])
# 根据by后面的参数进行升序排列，使用drop参数避免将旧索引添加为列
df = df.sort_values(by=['Vehicle_ID']).reset_index(drop=True)
print(df.shape) #(1048575, 25)

print(df.head(10))

