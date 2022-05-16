# -*- coding: utf-8 -*-
"""
@File    : 测试中文.py
@Time    : 2022/5/12 13:37
@Author  : Leonardo Zhou
@Email   : 2974519865@qq.com
@Software: PyCharm
"""

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pylab import mpl


mpl.rcParams['font.sans-serif']=['SimHei']

mpl.rcParams['axes.unicode_minus'] = False


x = [1,2]
y = [3,4]

plt.plot(x,y)
plt.yticks(np.linspace(0,4,9))

plt.legend(['撒大苏打'])

plt.title('啊啊')

plt.show()