# 通过gpa和gre来预测一个学生是否可以被录取
import pandas as pd
import matplotlib.pyplot as plt

admissions = pd.read_csv('admissions.csv')
print(admissions.head())
# gpa-平均成绩（绩点）； gre--英语成绩
plt.scatter(admissions['gpa'],admissions['admit'])
# plt.show()

import numpy as np

# logit Function
#sigmod函数
def logit(x):
    return np.exp(x)/(1+np.exp(x))

x = np.linspace(-6,6,50,dtype=float)

y = logit(x)

fig = plt.figure()
plt.plot(x,y)
plt.ylabel("probability")
# plt.show()

# 使用sklearn中的逻辑回归的库
import sklearn
from sklearn.linear_model import LogisticRegression
logistic_mode = LogisticRegression()

# print(admissions[['gpa']])
# 训练
logistic_mode.fit(admissions[['gpa']],admissions['admit'])
# 预测可能性
pred_probs = logistic_mode.predict_proba(admissions[['gpa']])# 预测对应的概率值
fig = plt.figure()
plt.scatter(admissions['gpa'],pred_probs[:,0])# 1，表示被录取的可能性，如果写成0表示的是不被录取的可能性
# plt.show()

# 预测最终的结果
fitted_labels = logistic_mode.predict(admissions[['gpa']])
fig = plt.figure()
plt.scatter(admissions['gpa'],fitted_labels)
# plt.show()

admissions['predicted_label'] = fitted_labels
print(admissions['predicted_label'].value_counts())# 打印通过预测得到的结果中被录取与不被录取的人数
print(admissions.head())

# 通过精度来评判模型的好与坏
admissions['actual_label'] = admissions['admit']
matchs = admissions['predicted_label'] == admissions['actual_label']

print('*****************')
# print(matchs)
print(admissions[matchs])
correct_predictions = admissions[matchs]# 预测对的样本
accuracy = len(correct_predictions)/float(len(admissions))
print(accuracy)

# 通过True Positive Rate来评判模型的好与坏

true_positive_filter = (admissions['predicted_label']==1)& (admissions['actual_label']==1)
true_positives = len(admissions[true_positive_filter])#TP,预测对了且是正例的个数

true_negative_filter = (admissions['predicted_label']==0) &(admissions['actual_label']==0)
true_negatives = len(admissions[true_negative_filter])#TN，预测对了且是负例的个数


false_positive_filter = (admissions['predicted_label']==1)& (admissions['actual_label']==0)
false_positives = len(admissions[false_positive_filter])#FP,预测错了且是正例的个数

false_negative_filter = (admissions['predicted_label']==0) &(admissions['actual_label']==1)
false_negatives = len(admissions[false_negative_filter])#FN，预测错了且是负例的个数

sensitivity = true_positives/float(true_positives+false_negatives)
print('检测出正例的效果',sensitivity)

specificity = true_negatives/float(true_negatives+false_positives)
print('检测出正例的效果',specificity)


