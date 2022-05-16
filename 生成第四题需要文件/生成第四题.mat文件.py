# %%
import scipy.io as scio
import numpy as np
import pandas as pd
import os

# %%
# 读取原始文件
gearbox_initial = []

for i in range(5):
    gearbox_initial.append(pd.read_excel('附件1.xlsx', i, dtype=np.float32))

# %%
# 读取测试文件
test_data = []
for i in range(12):
    test_data.append(pd.read_excel('附件2.xls', i, dtype=np.float32))

# %%
# 生成文件夹
for i in range(1,13):
    os.mkdir('test{}'.format(i))

# %%
# 生成.mat文件
save_path = 'test{}/g{}0.mat'

for i in range(1,13):
    for j in range(5):
        temp1 = gearbox_initial[j].iloc[:,1:4]
        temp2 = test_data[i-1].iloc[:,1:4]
        temp = np.array(pd.concat([temp1, temp2]))
        scio.savemat(save_path.format(i,j),{'g{}0'.format(j):np.array(temp)})
