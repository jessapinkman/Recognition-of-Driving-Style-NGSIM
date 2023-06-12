import pandas as pd
import  numpy as np
import math
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from factor_analyzer import factor_analyzer,Rotator
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from scipy.spatial import distance

pd.set_option('display.max_columns', None)

#出去车辆编号总共19个特征变量
#换道时间？
param = pd.DataFrame(columns=['车辆编号',
                          '速度最大值','速度均值','速度标准差',
                          '横向速度最大值','横向速度均值','横向速度标准差',
                          '加速度绝对值最大值','加速度绝对值均值','加速度标准差',
                          '横向加速度绝对值最大值','横向加速度绝对值均值','横向加速度标准差',
                          '最小跟车距离','跟车距离均值','跟车距离标准差',
                          '最小车头时距','车头时距均值','车头时距标准差',
                          '行驶距离'])

data = pd.read_csv('data/i-80_LC_16s_Smoothed.csv')

for i in range(int(data.index.size / 160)):
    param.loc[i, '车辆编号'] = i

    param.loc[i, '速度最大值'] = max(data.loc[(i*160) : (i*160+159), 'Smoothed_v_Vel/(km/h)'])
    param.loc[i, '速度均值'] = np.mean(data.loc[(i*160) : (i*160+159), 'Smoothed_v_Vel/(km/h)'])
    param.loc[i, '速度标准差'] = np.std((data.loc[(i*160) : (i*160+159), 'Smoothed_v_Vel/(km/h)']), ddof=1)

    param.loc[i, '横向速度最大值'] = max(abs(data.loc[(i*160) : (i*160+159), 'Lateral_Vel/(m/s)']))
    param.loc[i, '横向速度均值'] = np.mean(abs(data.loc[(i*160) : (i*160+159), 'Lateral_Vel/(m/s)']))
    param.loc[i, '横向速度标准差'] = np.std(abs(data.loc[(i*160) : (i*160+159), 'Lateral_Vel/(m/s)']), ddof=1)

    param.loc[i, '加速度绝对值最大值'] = max(abs(data.loc[(i*160) : (i*160+159), 'Smoothed_v_Acc/(m/s^2)']))
    param.loc[i, '加速度绝对值均值'] = np.mean(abs(data.loc[(i*160) : (i*160+159), 'Smoothed_v_Acc/(m/s^2)']))
    param.loc[i, '加速度标准差'] = np.std(abs(data.loc[(i*160) : (i*160+159), 'Smoothed_v_Acc/(m/s^2)']), ddof=1)

    param.loc[i, '横向加速度绝对值最大值'] = max(abs(data.loc[(i * 160): (i * 160 + 159), 'Lateral_Acc/(m/s^2)']))
    param.loc[i, '横向加速度绝对值均值'] = np.mean(abs(data.loc[(i * 160): (i * 160 + 159), 'Lateral_Acc/(m/s^2)']))
    param.loc[i, '横向加速度标准差'] = np.std(abs(data.loc[(i * 160): (i * 160 + 159), 'Lateral_Acc/(m/s^2)']), ddof=1)

    param.loc[i, '最小跟车距离'] = min(data.loc[(i * 160):(i * 160 + 159), 'Smoothed_Space_Headway/m'])
    param.loc[i, '跟车距离均值'] = np.mean(data.loc[(i * 160):(i * 160 + 159), 'Smoothed_Space_Headway/m'])
    param.loc[i, '跟车距离标准差'] = np.std((data.loc[(i * 160):(i * 160 + 159), 'Smoothed_Space_Headway/m']), ddof=1)

    param.loc[i, '最小车头时距'] = min(data.loc[(i * 160):(i * 160 + 159), 'Smoothed_Time_Headway/s'])
    param.loc[i, '车头时距均值'] = np.mean(data.loc[(i * 160):(i * 160 + 159), 'Smoothed_Time_Headway/s'])
    param.loc[i, '车头时距标准差'] = np.std((data.loc[(i * 160):(i * 160 + 159), 'Smoothed_Time_Headway/s']), ddof=1)

    param.loc[i, '行驶距离'] = data.loc[i * 160 + 159, 'Smoothed_Local_Y/m'] - data.loc[i * 160, 'Smoothed_Local_Y/m']

# param.to_csv(r'data/i-80_PCA_param.csv', index=None,encoding='utf-8_sig') # [282 rows x 20 columns]


data = pd.read_csv('data/i-80_PCA_param.csv')
data = data.drop(['车辆编号'],axis=1)
#降低到纬度4
pca = PCA(n_components=5)
#训练
pca.fit(data)
print('各主成分贡献度:{}'.format(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

# kmo检验
# print(round(factor_analyzer.calculate_kmo(data)[1],4))
# print(round(factor_analyzer.calculate_bartlett_sphericity(data)[1], 3))

a = factor_analyzer.FactorAnalyzer(n_factors=5,method='principal',rotation='varimax')
a.fit(data)

#特征值 特征向量
print(a.get_eigenvalues())

# 计算变量共同度
# 变量共同度也就是变量方差，是每个原始变量在每个共同因子的负荷量的平方和，也就是指原始变量方差中由共同因子所决定的比率。变量的方差由共同因子和唯一因子组成。
# 共同性表明了原始变量方差中能被共同因子解释的部分，共同性越大，变量能被因子说明的程度越高，即因子可解释该变量的方差越多。共同性的意义在于说明如果用共同因子替代原始变量后，
# 原始变量的信息被保留的程度。 一般共同度大于0.5比较好。
# [0.76765489 0.6303584  0.67001607 0.87335483 0.84219397 0.88516507
#  0.60763868 0.69001741 0.80449704 0.96708168 0.60814645 0.90731911
#  0.34524563 0.7816116  0.77909612 0.58342612 0.78947553 0.90452909
#  0.59185833]
#除了x13 最小跟车距离之外其他因素都是>0.5的
print(a.get_communalities())

#因子贡献率，每个因子都对变量有一定的贡献，依次是总方差贡献率、方差贡献率、累计方差贡献率
print(a.get_factor_variance())

#绘制旋转后的因子载荷矩阵
#因子载荷矩阵可以得到原始指标变量的线性组合，如X1=a11*F1+a12*F2+a13*F3
#可以找出某一因子和哪个成分最为相关
b = Rotator().fit_transform(a.loadings_)
b = pd.DataFrame(b)
# b.columns = ['成分1', '成分2', '成分3', '成分4']
b.columns = ['成分1', '成分2', '成分3', '成分4', '成分5']
b = round(b, 5)
b.to_csv(r'data/旋转后的因子载荷矩阵5.csv', index=None, encoding='utf-8_sig')

#画heatmap
plt.figure(figsize = (14,14))
ax = sns.heatmap(b, annot=True, cmap="BuPu")
plt.title("Factor-load matrix heatmap")
plt.ylabel("factor")
plt.xlabel("component")
plt.show()

#绘制因子得分系数矩阵
#通过因子得分矩阵可以得到公因子的线性组合，如F1=a11*X1+a21*X2+a31*X3
corr = np.matrix(data.corr().values)
c = np.dot(np.linalg.inv(corr), np.matrix(a.loadings_))
c = pd.DataFrame(c)
c = round(c, 5)
# c.columns = ['成分1', '成分2', '成分3', '成分4']
c.columns = ['成分1', '成分2', '成分3', '成分4', '成分5']
c.to_csv(r'data/因子得分系数矩阵5.csv', index=None, encoding='utf-8_sig')


para = np.array(data)
d = np.matmul(para, np.array(c))
estimator = KMeans(n_clusters=2)
estimator1 = KMeans(n_clusters=2, init='k-means++')
#开始聚类
estimator1.fit(d)
k_centers = estimator1.cluster_centers_  #找出聚类中心
label_pred = estimator1.labels_
length = len(label_pred)
arr = label_pred.flatten() #只保留dim之前的维度，其他维度的数据全都挤在dim这一维。
arr = pd.Series(arr)
arr = arr.value_counts()
print(arr)
# 0    241
# 1     29
# 2     12   length=282

#返回样本平均轮廓系数 0.8496045499938667  n_clusters=2  0.8723963210321642
#值越大表示聚类效果越好 [-1, 1]
score_outline = metrics.silhouette_score(d, labels=label_pred, metric='euclidean')
#返回所有样本的平均轮廓系数
# score_avg_outline = metrics.silhouette_samples(d, labels=label_pred, metric='euclidean')
print(score_outline)
silhouettescore=[]
for i in range(2,15):
    kmeans=KMeans(n_clusters=i,init='k-means++').fit(d)
    score=metrics.silhouette_score(d,kmeans.labels_)
    silhouettescore.append(score)
plt.figure(figsize=(10,6))
plt.plot(range(2,15),silhouettescore,linewidth=1.5,linestyle='-')
plt.show()
print(silhouettescore)


#CH计算簇间离散度和簇内离散度的比值，越大越好，通过DBSCAN得到的聚类指数一般比较高
#1368.9512944351372
score_CH = metrics.calinski_harabasz_score(d, labels=label_pred)
# print(score_CH)

#DB系数，该指数标识居群之间的平均相似性，越接近0越好
#0.5153440911163698
score_DB = metrics.davies_bouldin_score(d, labels=label_pred)
# print(score_DB)


#SSE误差平方和，每类中的点到对应之心的欧氏距离平方的和，反应了簇的凝聚度，越小越好
# score_SSE = sum(np.min(distance.cdist(d, k_centers, metric='euclidean')))
# score_SSE = estimator.score(d, label_pred)
# print(score_SSE)