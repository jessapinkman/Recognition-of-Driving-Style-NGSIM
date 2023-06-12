import pandas as pd
import  numpy as np
import math
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from factor_analyzer import factor_analyzer,Rotator
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import sklearn.metrics as metrics
from scipy.spatial import distance

data = pd.read_csv('data/i-80_PCA_param.csv')
data = data.drop(['车辆编号'],axis=1)
c = pd.read_csv('data/因子得分系数矩阵.csv')

para = np.array(data)
d = np.matmul(para, np.array(c))
estimator = DBSCAN(eps=0.3, min_samples=10).fit(d)
#开始聚类

label_pred = estimator.labels_
length = len(label_pred)
arr = label_pred.flatten() #只保留dim之前的维度，其他维度的数据全都挤在dim这一维。
arr = pd.Series(arr)
arr = arr.value_counts()

print(metrics.davies_bouldin_score(d, labels=label_pred))