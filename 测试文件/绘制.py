# -*- coding: utf-8 -*-
"""
@File    : 绘制.py
@Time    : 2022/5/12 23:01
@Author  : Leonardo Zhou
@Email   : 2974519865@qq.com
@Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


g = []
gearbox_initial = []

feature_names = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
color = ['red', 'green', 'blue', 'yellow', 'black']

for i in range(5):
    gearbox_initial.append(pd.read_excel('附件1.xlsx', i, dtype=np.float32))
    gearbox_initial[i]['target'] = i
    g.append(gearbox_initial[i].to_numpy())

gearbox = np.array(g)
length = 1000
col = 2

fig, axes = plt.subplots(6, 1, figsize=(100, 20))
for i, (ax, gear) in enumerate(zip(axes.ravel(), gearbox)):
    ax.set_title('situation {}'.format(i))
    ax.plot(range(1, length+1), gear[:length, col])
    axes[-1].plot(range(1, length+1), gear[:length, col], color=color[i])

plt.savefig('sensor{}.png'.format(col))
plt.show()
