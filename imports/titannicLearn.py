import pandas as pd
import numpy as np
titanc_survival = pd.read_csv('titanic_train.csv');
print(titanc_survival.head())
print(titanc_survival.columns)
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
# 上边的元素分别代表的含义
#PassengerId:人的编号
#Survived 取值为0或者1 label 分类任务的分类标准
#Pclass 船舱的等级
#SibSp 当前这个人的兄弟姐妹的个数
#Parch 当前乘客的老人和小孩的数量
#Ticket 火车票的编码
#Fare 当前乘客的船票价格
#Cabin 当前乘客的船舱的编号
#Embarked 当前乘客的登船的地点

age = titanc_survival['Age']

print(age.loc[0:10]) # NaN表示一个缺失值

age_null = pd.isnull(age) # 获取到的是一个bool数列
ageIsNull = age[age_null] # 获取当前age为Null的所有值
print(ageIsNull)
age_null_len = len(ageIsNull)
print(age_null_len)# age为Null的数量

# age中有部分的值为NaN，所以在计算平均值的时候会有比较大的影响
# 1.在所有的年龄中，我们将年龄不为NaN的这部分年龄找出来，用这些值来计算平均值
good_age = age[age_null==False]
average_age = sum(good_age)/len(good_age)
print(average_age)
# 2.mean()方法，等同于上边的那个方法
average_age = age.mean()
print(average_age)

# 获取每一个船舱等级中车票的平均价格
## 普通方法
pclass = [1,2,3]
faersByPclass = {}

for i in pclass:
    pclass_rows = titanc_survival[titanc_survival['Pclass']==i]# 获取pclass对应于某一个具体值的所有行
    pclass_fare = pclass_rows['Fare']# 获取到这些行中对应的票价这一列
    pclass_ave = pclass_fare.mean()
    faersByPclass[i] = pclass_ave

print(faersByPclass)

## pivot_table方法,pivot_table中包含三个参数(index,values,aggfunc),对应这个题，pclass相当于index,fare相当于values,mean相当于aggfunc,aggfunc默认是np.mean

evrange_fare = titanc_survival.pivot_table(index='Pclass',values='Fare',aggfunc=np.mean)
print(evrange_fare)

# dropna函数(他可以将数据中为空的数据所在的行删除)
drop_na_columns = titanc_survival.dropna(axis=1)# 定义一个dropna
new_titanic_survival = titanc_survival.dropna(axis=0,subset=['Age','Sex'])# Age 和 Sex这两列

# 根据属性index找一个具体的值，如 第83个人的Age
age_83 = titanc_survival.loc[83,'Age']
print(age_83)

# 对数据集根据Age进行排序,改变原先的索引值
new_titanic = titanc_survival.sort_values('Age',ascending=False)
print(new_titanic[0:10])
print('----------------------')
titanc_reindexed = new_titanic.reset_index(drop = True)#drop = True表示原来的排序不要了，使用新的排序
print(titanc_reindexed[0:10])




