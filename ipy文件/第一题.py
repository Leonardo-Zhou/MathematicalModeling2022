# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

g = []
gearbox_initial = []

feature_names = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
color = ['red', 'green', 'blue', 'yellow', 'black']

for i in range(5):
    gearbox_initial.append(pd.read_excel('附件1.xlsx', i, dtype=np.float32))
    gearbox_initial[i]['target'] = i
    g.append(gearbox_initial[i].to_numpy())

# %%
# 标准差图像绘制
_, axes = plt.subplots(4, 1, figsize=(30, 15))
length, start = 500, 100
for j, ax in enumerate(axes):
    for i in range(5):
        ax.plot(range(length), g[i][start: start + length, j + 5])
        ax.set_title(feature_names[j])
plt.legend(['00', '10', '20', '30', '40'])
plt.show()

# %%
# 原始图像绘制
_, axes = plt.subplots(4, 1, figsize=(30, 15), dpi=100)
length, start = 1000, 100
for j, ax in enumerate(axes):
    for i in range(2):
        ax.plot(range(length), g[i][start: start + length, j + 1])
        ax.set_title(feature_names[j])
plt.legend(['00', '10', '20', '30', '40'])
plt.show()


# %%
# 01标准化，进行方差特征筛选
class VarianceSelect:
    def __init__(self):
        pass

    # 将数据进行0-1标准化
    def MaxMinTransform(self, data):
        for var_name in data.columns:
            mi = data[var_name].min()
            ma = data[var_name].max()
            data[var_name] = data[var_name].apply(
                lambda x: (x - mi) / (ma - mi))
        return data

    # 方差特征筛选

    def VarianceFunc(self, data_final):
        selector = VarianceThreshold(threshold=0.01)
        result_select = selector.fit_transform(data_final)
        result_support = selector.get_support(indices=True)
        return result_select, result_support

    # 加载及调用

    def load_transform(self):
        print('查看初始数据 : ', data.head())
        print('查看各变量均值方差 : ', np.mean(
            data.iloc[:, :]), np.var(data.iloc[:, :]))

        data_final = self.MaxMinTransform(data.iloc[:, :])
        print('查看0-1标准化后数据 : ', data_final.head())
        print('标准化后查看各变量均值方差 : ', np.mean(data_final), np.var(data_final))

        result_select, result_support = self.VarianceFunc(data_final)
        print('筛选方差大于0.01的特征 : ', result_select)
        print('方差筛选后保留特征索引 : ', result_support)


boxes = ['gearbox00', 'gearbox10', 'gearbox20', 'gearbox30', 'gearbox40']
for i in range(5):
    print('*' * 50)
    print("对于{}".format(boxes[i]))

    data = gearbox_initial[i].iloc[:500, 1:5]
    VarianceSelect().load_transform()

    print('\n\n')

# %%
# 原始数据读取（未经处理）
g = []
gearbox_initial = []

feature_names = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
color = ['red', 'green', 'blue', 'yellow', 'black']

for i in range(5):
    gearbox_initial.append(pd.read_excel('原始数据未处理.xlsx', i, dtype=np.float32))
    gearbox_initial[i]['target'] = i

# %%
# 绘制各个散点图
boxes = ['gearbox00', 'gearbox10', 'gearbox20', 'gearbox30', 'gearbox40']
feature_names = ['sensor1', 'sensor2', 'sensor3', 'sensor4']

fig, axes = plt.subplots(1, 5, figsize=(100, 15))
length, start = 10000, 0

for i in range(4):
    for j, ax_ in enumerate(axes):
        ax_.scatter(range(length),
                    gearbox_initial[j][feature_names[i]][start: start + length])
        ax_.set_title('{}-{}'.format(boxes[j], feature_names[i]))
    plt.savefig('{}.png'.format(feature_names[i]))
plt.show()
