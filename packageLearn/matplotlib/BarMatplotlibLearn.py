import matplotlib.pyplot as plt
import numpy as np
ax = plt.subplot()
x =np.arange(12)+0.75 # 每一个条形图距离x轴0点的距离
y = [3.4,3.8,4.0,3.9,3.5,3.6,3.6,3.9,3.8,3.7,3.8,4.0]
ax.bar(x,y,0.2)# 0.2表示的是条形图的宽度，默认画出来的图是垂直显示的，如果想让结果水平显示，只需要将bar改成barh即可
plt.show()