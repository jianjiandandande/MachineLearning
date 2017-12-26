#!usr/bin/python3.6
'''

Create by Vincent on 2017/12/26 11:44

'''

import numpy as np
import operator

# 构建dataSet
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

# 分类算法
# inX 待分类的点
# dataSet 已知的数据集
# lables 已知的类别
# k 距离最近的点的个数
# 计算距离是根据欧式距离公式来做的
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #1.求差值
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet # np.tile方法对inX进行了拉伸，使得inX变成与dataSet行数相同的矩阵，从而计算inX到dataSet中每个点的距离
    #2.求平方
    sqDiffMat = diffMat**2
    #3.求和
    sqDistance = sqDiffMat.sum(axis=1)# 指定矩阵的维度
    #4.求根号
    distance = sqDistance**0.5
    #5.排序
    sorteDistIndicies = distance.argsort()# 排完序之后返回index
    #6.classCount是一个dict，它里面存储的是当前的这个点，属于每一个类别的比例（这个比例是用数量来表示的）
    classCount = {}
    for i in range(k):
        voteLabel = labels[sorteDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1 # 对每一个类别所属的比例进行更新
    print(classCount)
    # 对classCount进行排序(根据classCount中各个类别所占的比例)（倒序），从而得到当前的点更符合的类别
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # sortedClassCount 是一个list,它里面包含了多个tuple
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group, labels = createDataSet()
    test = classify0([2,0.2],group,labels,3)
    print(test)

