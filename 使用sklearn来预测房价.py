# 探索房屋数据集
import pandas as pd

df = pd.read_csv('house_data.csv')#读取训练样本
print(df.head())#将获取到的数据展示出来
##可视化房屋数据集的特征
import matplotlib.pyplot as plt #可视化包
import seaborn as sns

#指定维度
cols = ['LSTAT','AGE','DIS','CRIM','MEDV','TAX','RM']# 人口百分比，年龄，距离市中心，犯罪率，房价，税，平均房间数
# 画图
sns.pairplot(df[cols],size=2.5)#参数：data(数据集)，size(图像大小)
plt.show()
# 出现的结果中的每一个小图表示的是两个维度(元素)之间的相关性，而对角线上的图表示的是一个维度的直方图

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
        self.w = np.zeros(1+X.shapep[1])
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

