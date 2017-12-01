import matplotlib.pyplot as plt

fig = plt.figure()# 指定默认的绘图区间，可以在figure方法中传入一个元组，来指示区域的大小eg:(3,5)
ax1 = fig.add_subplot(2,2,1)# 表示一个子图，一个2x2的区间上的死一个位置的子图
ax2 = fig.add_subplot(2,2,2)
ax4 = fig.add_subplot(2,2,4)

ax1.plot()#对第一个子图进行绘制，可以在plot中传入label来指示当前的图像所代表的含义
ax2.plot()#对第二个子图进行绘制
ax4.plot()#对第四个子图进行绘制

plt.legend(loc='best')# 这个方法会让上边绘图时指定的label显示出来，loc用来指定label的位置

plt.show()