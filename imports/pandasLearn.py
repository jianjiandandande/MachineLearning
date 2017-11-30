import pandas
house_info = pandas.read_csv('house_data.csv')
print(type(house_info))# DataFrame类型 -- >当做一个矩阵结构
print(house_info.dtypes)# 包含几种类型的结构(属性的类型)
print(help(pandas.read_csv))
print(house_info.head())# 默认显示前5条数据，可以根据需要，给head中传参数，指定你想让他显示的数据项

head_three = house_info.head(3)# 获取前三行
tail_three = house_info.tail(3)# 获取尾三行

print('前三行\n',head_three)
print('尾三行\n',tail_three)

# 获取数据中的属性名
print('属性名\n',house_info.columns)

# 获取数据集中index为0的数据
print(house_info.loc[0])

#DataFrame中的object相当于str

# 取数据集中index为3-6的数据
print(house_info.loc[3:6])

# 通过列名来定位数据
print('第一列的数据为:\n',house_info['CRIM'])# 取一列

columns = ['CRIM','ZN']
print('前两列的数据为:\n',house_info[columns])

# 条件查找 -->找列名以'M'结尾的列的数据

columns = house_info.columns.tolist()
print(columns)
##找到列名中以'M'结尾的列名
end_m = []
for i in columns:
    if i.endswith('M'):
        end_m.append(i)

data_m = house_info[end_m]

print('最终结果:\n',data_m.head(4))# 取前4列

# 将数据添加到数据集当中(给数据集新增一列)

print(house_info.shape)

newLine = house_info['CRIM']*house_info['CRIM']
newLine = newLine*100
house_info['NEWLINE'] = newLine
print(house_info.shape)

# 数据的排序

house_info.sort_values('CRIM',inplace=True)#对某一列进行从小到大的排序，并且指定结果是新生成一个DataFrame还是在原来的DataFrame中做修改

print('排序完的结果\n',house_info['CRIM'])

# 实现从大到小的排序
house_info.sort_values('CRIM',inplace=True ,ascending=False)# ascending默认指的是升序排序，改为False之后就变成降序排序
print('从大到小的结果\n',house_info['CRIM'])