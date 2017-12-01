import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [3.4, 3.8, 4.0, 3.9, 3.5, 3.6, 3.6, 3.9, 3.8, 3.7, 3.8, 4.0]

fig=plt.figure()
ax=fig.add_subplot(111)

xs = ["1948.%02d" %t for t in x]
ax.axes.set_xticks(x)
ax.axes.set_xticklabels(xs,rotation=30)

ax.plot(x, y, linestyle=' ', marker='o', color='b')
ax.plot(x, y, color='r')
plt.xlabel('Month')  # 横坐标表示的意义
plt.ylabel('shiyelv')  # 纵坐标表示的意义
plt.title('tongji')  # 标题

plt.show()
