import numpy as np

# 通过numpy来获取一个txt文本中的内容
world_alcohol = np.genfromtxt('world_alcohol.txt',delimiter=',',dtype=str)# 参数：文件名，分隔符，读取方式(均以字符串的形式来读)
print(type(world_alcohol))# --> ndarray类型
print(world_alcohol)
# 获取numpy中关于某一个函数的帮助文档
#print(help(np.genfromtxt))

# numpy的array方法来构造数组
# 一维
vector = np.array([1,2,3,4]) # 使用numpy.array()这个方法时一定要注意，它里面传入的值一定是一种类型

print(vector.dtype) # 获取vector里面值的类型

# 通过numpy构造的数组同样支持切片
print(vector[0:2])# -->获取数组中第0到第2个位置的值，不包括第2个位置的值

# 二维
matrix = np.array([[1,2,3],[4,5,6]]) #

print(vector.shape)# 获取行列信息 (4,)-->四列

print(matrix.shape)# (2,3)-->2行3列

# 获取矩阵当中具体某一列的值
print(matrix[:,1])

# 获取矩阵当中具体某几列的值
print(matrix[:,1:3])

# 判断某一个数(2)是否在一个ndarray当中
print(vector==2) # 它会将要比较的值与ndarray中的每个值进行比较，最终再返回一个array
result = vector==2 # 比较完之后产生的那个bool数组
print(vector[result]) # 将bool数组当做一个缩影，返回等于True那个位置对应的值

# 对numpy.array中所有的数据进行类型转换
vector = vector.astype(str)

print("?",vector.dtype)
print(vector)
vector = vector.astype(int)
print(vector.min()) # 最小值

# 对矩阵进行行，列求和
print(matrix.sum(axis=1))# 行
print(matrix.sum(axis=0))# 列

# numpy.arange
a = np.arange(15)# 获取0-14的一个数列
print(a)
a = a.reshape(3,5)# 将a转换成一个3x5的矩阵
print(a)
print(a.ndim)# a的维度
print(a.dtype)# a的类型
print(a.dtype.name)# a的类型名
print(a.size)# a的大小

# 初始化3x4的全0矩阵
zeros = np.zeros((3,4))
print(zeros)
ones = np.ones((3,4),dtype=np.int32)# 全是1的矩阵,指定类型为整型
print(ones)

# 构造一个从10到30之间的等差数列，公差为5
print(np.arange(10,30,5))

# 产生一个2x3的随机数矩阵
print(np.random.random((2,3)))# 默认这些值是在-1到1之间

from numpy import pi
# pi就是π(3.14)
print(np.linspace(0,2*pi,100))# 产生从0-2π之间的100个数，间距相同

a1 = np.array([20,30,40,50])
b1 = np.arange(4)
c = a1-b1 # 对应位置相减
c = c-1 # c的每一个位置都减1
b1**2 # b1的每一个位置都进行平方

print(a1<35) # 输出的是一个bool array

A= np.array([[1,1],[0,1]])
B= np.array([[2,0],[3,4]])
print('--------------------')
print(A*B)# 对应位置乘法

print('--------------------')
print(A.dot(B))
print('--------------------')
print(np.dot(A,B))

C = np.arange(3)
print(C)
print(np.exp(C))# 求e的多少次幂
print(np.sqrt(C))# 求一个数的开平方

z = np.floor(10*np.random.random((3,4)))# 先取随机数，然后向下取整
print(z)
print('-------------')

print(z.ravel())# 将3x4的矩阵转换为一个数列
print('--------------')
z.shape = (2,6)
print(z) # 将z转换成一个2x6的矩阵 使用z.reshape(2,-1)可以达到同样的效果
print('--------------')

z.reshape(2,-1)
print(z)
print('---------------')
print(z.T) # z的转置

#拼接的方法
c = np.floor(10*np.random.random((2,2)))
d = np.floor(10*np.random.random((2,2)))
print('------------------')
print(c)
print('------------------')
print(d)
print('------------------')
print('横向拼接\n',np.hstack((c,d)))
print('------------------')
print('纵向拼接\n',np.vstack((c,d)))

#切片的方法(以水平切片为例 hsplit, 竖直切片使用vsplit完成)
e = np.floor(10*np.random.random((2,12)))
print('-------------------')
print(e)
print('-------------------')
print('横向切分\n',np.hsplit(e,3))# 平均切成3块
print('-------------------')
print('随机切分\n',np.hsplit(e,(3,4))) # 在第三，第四列位置各切一刀，最终将矩阵分为三个随机大小的矩阵

##python中的复制的概念
# 1. 使用等于号直接复制
f = np.floor(10*np.random.random((2,6)))
g = f # 使用等号进行赋值之后， f与g同时指向了这个矩阵，在a中对矩阵做变换，b中的矩阵同样会做出相应的变化
g.shape = (3,4)

print('f.shape=',f.shape)
print(id(f)==id(g))

# 2.使用f.view()进行复制, 这种方式两个矩阵共用一个数据

h = f.view()
h.shape = (2,6)
print(f.shape)# 这里可以看到，在改变h的过程中，f并没有随着h的变化而变化

print(h[1][5])
# h[1][5] = 3355
print(f)
h[1][5] = 2324
print(f) # 发现f中的值也发生了改变
print(id(h)==id(f))# 它们的id不相同


# 3.深复制，复制完成之后，两个矩阵之间没有任何关系
i = f.copy()
i[2][1] = 3782
print(f)
print(id(i)==id(f))# id 也不同

