# -*- coding: utf-8 -*-
"""
@File    : csv文件.py
@Time    : 2022/5/12 19:17
@Author  : Leonardo Zhou
@Email   : 2974519865@qq.com
@Software: PyCharm
"""
import numpy as np
import pandas as pd

temp=np.loadtxt("test_s.csv",dtype=np.float32,delimiter=',')
a=pd.DataFrame(temp,index=[10,20,30,40,50,60,70,80,90,100],columns=[10,20,30,40,50,60,70,80,90,100])

# print(temp)