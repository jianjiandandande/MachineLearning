#!usr/bin/python3.6
'''

Create by Vincent on 2017/12/26 14:28

'''

import numpy as np
import pandas as pd

df = pd.read_csv('iris.data')
head = df.head()
# print(head)

# 添加列名(属性名) sepal 萼片; petal 花瓣;
df.columns = ['sepal_len','sepal_wid','petal_len','petal_wid','class']
head = df.head()
# print(head)

# 切分特征与label值
X = df.ix[:,0:4].values
y = df.ix[:,4].values
# print(X)
# print(y)

# 绘制四个特征中任一个特征对其他三个特征的影响
from matplotlib import pyplot as plt
import math

label_dict = {1:'Iris-setosa',
              2:'Iris-versicolor',
              3:'Iris-virginica'}
feature_dict = {0:'sepal length [cm]',
                1:'sepal width [cm]',
                2:'petal length [cm]',
                3:'petal width [cm]'}
plt.figure(figsize=(8,6))
for cnt in range(4):
    plt.subplot(2,2,cnt+1)
    for lab in ('Iris-setosa','Iris-versicolor','Iris-virginica'):
        plt.hist(X[y==lab,cnt],
                 label=lab,
                 bins=10,
                 alpha=0.3)
        plt.xlabel(feature_dict[cnt])
        plt.legend(loc='upper right',fancybox=True,fontsize=8)

plt.tight_layout()
# plt.show()

# 标准化数据，让所有的值的浮动范围在同样的区间上
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
# print(X_std)

#计算协方差
#1.计算某种特征的均值
mean_vec = np.mean(X_std,axis=0)
#2.计算最终的协方差
cov_mat = (X_std-mean_vec).T.dot(X_std-mean_vec)/(X_std.shape[0] - 1)
# print('Covariance matrix \n%s' % cov_mat)# 打印协方差矩阵

cov_mat = np.cov(X_std.T)
eig_vals,eig_vecs = np.linalg.eig(cov_mat)# 计算特征值和特征向量
# print('eigenvectors \n%s' % eig_vecs)# 打印特征向量
# print('\neigenvalues \n%s' % eig_vals)# 打印特征值,特征值代表了当前特征向量的重要程度

#将特征值与特征向量做成一个对的形式
eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
print(eig_pairs)# 特征值与特征向量构成的那个 对

#根据特征值对这个 对 进行排序
eig_pairs.sort(key=lambda x:x[0],reverse=True)#(倒序)
for i in eig_pairs:
    print(i[0])# 打印特征值


# 对特征值进行归一化
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals,reverse=True)] #对特征值进行归一化
# print(var_exp)

# np.cumsum方法可以将数组中第i个位置的值变为前i个值得累加和
cum_var_exp = np.cumsum(var_exp)
# print(cum_var_exp)

# 绘图
plt.figure(figsize=(6,4))
plt.bar(range(4),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(4),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('principal components')
plt.legend(loc='best')
plt.tight_layout()
# plt.show()

# 通过绘图，我们可以看到四个特征当中的前两个特征是比较重要的，
# 将前两个特征变化为4*1的矩阵，相结合之后得到一个4*2矩阵。
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4, 1)))

# print(matrix_w)

# 将原始数据转化为n*2维
Y = X_std.dot(matrix_w)

# 原始四维数据时的点集划分
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(X[y==lab, 0],
                X[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()


# PCA降维后数据的点集划分
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(Y[y==lab, 0],
                Y[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()


