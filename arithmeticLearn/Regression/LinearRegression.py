#!usr/bin/python3.6
'''

Create by Vincent on 2017/12/28 12:06

'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

class LinearRegression():
    def __init__(self):
        self.w = None

    def fit(self,X,y):

        print(X.shape)
        X = np.insert(X,0,1,axis=1)# 在X中添加了一项X0,使其值为1，便于计算w
        print(X.shape)
        X_ = np.linalg.inv(X.T.dot(X))# np.linalg.inv的作用是取逆
        self.w = X_.dot(X.T).dot(y)

    def predicrt(self,X):
        X = np.insert(X,0,1,axis=1)
        y_pred = X.dot(self.w)
        return y_pred

def mean_squared_error(y_true,y_pred):
    mse = np.mean(np.power(y_true-y_pred,2))
    return mse

def main():
    # 加载数据集,即特征
    diabetes = datasets.load_diabetes()
    # 用一个特征
    X = diabetes.data[:,np.newaxis,2]
    print(X.shape)

    x_train,x_test = X[:-20],X[-20:]
    y_train,y_test = datasets.target[:-20],datasets.target[-20:]

    clf = LinearRegression()
    clf.fit(x_train,y_train)
    y_pred = clf.predicrt(x_test)

    print('误差，即目标函数最终的值',mean_squared_error(y_test,y_pred))

    fig = plt.figure()
    plt.scatter(x_test[:,0],y_test,color='black')
    plt.plot(x_test[:,0],y_pred,color='blue',linewidth=3)
    plt.show()

if __name__ == '__main__':
    main()

















