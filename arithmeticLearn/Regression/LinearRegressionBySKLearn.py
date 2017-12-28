# 探索房屋数据集
import pandas as pd

df = pd.read_csv('house_data.csv')#读取训练样本
# print(df.head())#将获取到的数据展示出来
##可视化房屋数据集的特征
import matplotlib.pyplot as plt #可视化包
#指定维度
# CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV
cols = ['LSTAT','AGE','DIS','CRIM','MEDV','TAX','RM']# 人口百分比，年龄，距离市中心，犯罪率，房价，税，平均房间数
# 画图
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

df.plot('LSTAT','MEDV',kind='scatter',ax=ax1)# 人口百分比对房价的影响
df.plot('RM','MEDV',kind='scatter',ax=ax2)# 平均房间数对房价的影响
# plt.show()

# 从sklearn库中找到线性回归的模型
import sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# print(df[['LSTAT']])
lr.fit(df[['LSTAT']],df['MEDV'])

# 预测

predictions = lr.predict(df[['LSTAT']])

print(predictions[0:5])# 预测数据
print(df['MEDV'][0:5])# 真实数据

# 将预测数据与真实数据在图中表示出来
fig = plt.figure()
plt.scatter(df['LSTAT'],df['MEDV'],c='red')
plt.scatter(df['LSTAT'],predictions,c='blue')
# plt.show()

# 衡量模型的好坏，通过均方误差来衡量

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['MEDV'],predictions)
print('均方误差：',mse)

'''
## 实现一个线性回归模型
## 通过梯度下降法计算回归参数，实现线性回归模型
import numpy as np #科学计算包
## 线性回归模型
class LinearRegressionByMyself(object):
    # 构造函数
    def __init__(self,Learn_rate=0.001,epoch = 20):
        self.Learn_rate = Learn_rate # 学习率
        self.epoch = epoch # 迭代次数

    # 训练方法 x为训练数据(除房价以外的特征)，y为标签数据(房价这个特征)
    def fit(self,X,y):
        # w是线性回归当中的那些特征，也就是x前面的那些系数，我们将它初始化为一个零向量
        # X.shape取到的是X的维度，X是二维的，我们取它的第一个维度X.shape[1],
        # 加1的原因是：常数项(线性模型中常数项)
        self.w = np.zeros(1+X.shape[1])
        self.cost_list = [] # 误差 后期会将误差和迭代次数放在一起进行画图，以此来表现出误差与迭代次数的关系

        for i in range(self.epoch):
            output = self.Regression_input(X)
            error = (y - output)
            self.w[1:] += self.Learn_rate * X.T.dot(error) # 更新系数
            self.w[0] += self.Learn_rate * error.sum() # 更新截距
            cont = (error**2).sum()/2.0 # 误差
            self.cost_list.append(cont)
        return self

    #计算第一个预测值
    def Regression_input(self,X):
        return np.dot(X,self.w[1:]) + self.w[0] # 点击

    # 预测
    def predict(self,X):
        return self.Regression_input(X)

X = df[['LSTAT']].values
y = df['MEDV'].values



# # 将数据进行标准化
# from sklearn.preprocessing import StandardScaler # 对数据进行归一化
#
# StandardScaler_x = StandardScaler()
# StandardScaler_y = StandardScaler()
#
# X_Standard = StandardScaler_x.fit_transform(X)
# y_Standard = StandardScaler_y.fit_transform(y)

model = LinearRegressionByMyself()

model.fit(X,y)

plt.plot(range(1,model.epoch+1),model.cost_list)

plt.ylabel('SSE')
plt.xlabel('EPOCH')

plt.show()
'''

