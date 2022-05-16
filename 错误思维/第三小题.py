# -*- coding: utf-8 -*-
"""
@File    : 第三小题.py
@Time    : 2022/5/12 8:39
@Author  : Leonardo Zhou
@Email   : 2974519865@qq.com
@Software: PyCharm
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pylab import mpl
from sklearn.tree import export_graphviz
import graphviz
import mglearn
import time
from sklearn.ensemble import GradientBoostingClassifier


def use_dis_tree():
    """
    使用决策树进行预测
    :return:
    """
    n_depth = 30
    # for 循环查找最合适的决策树的深度
    best_depth = 1
    highest_score = 0
    train_s, test_s = [], []

    for i in range(1, n_depth):
        tree = DecisionTreeClassifier(max_depth=i)
        tree.fit(X_train, y_train)
        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)

        train_s.append(train_score)
        test_s.append(test_score)

        best_depth = i if test_score > highest_score else best_depth
        highest_score = test_score if test_score > highest_score else highest_score

    plt.plot(range(1, n_depth), train_s)
    plt.plot(range(1, n_depth), test_s)

    plt.legend(['训练集准确度', '测试集准确度'])

    plt.xticks(range(1, n_depth))
    plt.yticks(np.linspace(0, 1, 11))

    plt.title('决策树不同复杂度准确情况')
    plt.savefig('决策树不同复杂度准确情况.png', dpi = 1000)
    plt.show()
    print('最佳深度为：{}'.format(best_depth))
    print('最好成绩为；{}'.format(highest_score))
    """
    最佳深度为：9
    最好成绩为；0.6939591836734694
    """


def draw_dis(n_depth=9):
    """
    绘制决策树示意图
    :param n_depth:
    :return:
    """
    tree = DecisionTreeClassifier(max_depth=n_depth)
    tree.fit(X_temp, y_temp)
    export_graphviz(tree, out_file='tree.dot', feature_names=feature_names,
                    class_names=['0', '1', '2', '3', '4'], filled=True)

    with open('tree.dot') as file:
        dot_graph = file.read()

    graphviz.Source(dot_graph)


def get_feature_importance_of_tree():
    """
    获得树的特征重要性
    :return:
    """
    tree = DecisionTreeClassifier(max_depth=9)
    tree.fit(X_train, y_train)
    plt.barh(range(len(feature_names)), tree.feature_importances_, align='center')
    plt.yticks(np.arange(len(feature_names)),feature_names)
    plt.xlabel('特征重要性')
    plt.ylabel('特征')

    plt.title('决策树最佳深度特征重要性')
    plt.savefig('决策树最佳深度特征重要性.png')
    plt.show()


def use_random_forest():
    """
    使用随机森林进行预测，并获得最佳树的个数
    :return:
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    best_n = 1
    best_depth = 1
    highest_score = 0
    train_s, test_s = [], []

    tree_num = range(10,101,10)
    depth = range(10,101,10)
    train_s, test_s = [], []

    for i in tree_num:
        train_s_temp = []
        test_s_temp = []
        for j in range(len(depth)):
            forest = RandomForestClassifier(n_estimators=i,n_jobs= 12, max_features=2, max_depth=depth[j])

            forest.fit(X_train,y_train)

            train_score = forest.score(X_train, y_train)
            test_score = forest.score(X_test, y_test)

            train_s_temp.append(train_score)
            test_s_temp.append(test_score)

            best_n = i if test_score > highest_score else best_n
            best_depth = i if test_score > highest_score else best_depth
            highest_score = test_score if test_score > highest_score else highest_score
        train_s.append(train_s_temp)
        test_s.append(test_s_temp)

    print('最佳树的数量为:{}'.format(best_n))
    print('最佳深度为:{}'.format(best_depth))
    print('最好成绩为:{}'.format(highest_score))
    """
    最佳树的数量为:90
    最佳深度为:90
    最好成绩为:0.6902312925170068
    """

    train_s = np.array(train_s)
    test_s = np.array(test_s)

    depth, tree_num = np.meshgrid(depth, tree_num)
    ax.plot_surface(depth, tree_num, train_s, cmap='winter',
                           linewidth=0, antialiased=False)
    ax.plot_surface(depth, tree_num, test_s, cmap='spring',)
    ax.set_zlim(0, 1.01)
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_xlabel('depth')
    ax.set_ylabel('tree num')

    ax.set_title('随机森林不同树的数量准确情况')
    plt.savefig('随机森林不同树的数量准确情况1.png', dpi=1000)
    plt.show()
    return train_s, test_s


def draw_forest(n_depth=90, n_estimators=90):
    forest = RandomForestClassifier(n_estimators=n_estimators,max_depth=n_depth)

    forest.fit(X_train, y_train)
    names, importance = feature_names, forest.feature_importances_
    plt.figure(figsize=(10, 5))

    plt.barh(names, importance)
    plt.xlabel('比重')
    plt.title('随机森林各要素所占比重')
    plt.savefig('随机森林各要素所占比重.png')
    plt.show()


def random_forest_sketch_map():
    """
    随机森林示意
    :return:
    """
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    forest = RandomForestClassifier(n_estimators=5, random_state=3)
    forest.fit(X, y)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title('Tree {}'.format(i))
        mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

    mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1][-1], alpha=0.4)
    axes[-1][-1].set_title('RandomForest')
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.title('随机森林示意')
    plt.savefig('随机森林示意.png')
    plt.show()


def use_boosting_tree():
    """
    使用梯度回升树，获得最佳深度
    :return:
    """
    n_depth = 10
    # for 循环查找最合适的决策树的深度
    best_depth = 1
    highest_score = 0
    train_s, test_s = [], []

    for i in range(1, n_depth):
        print(i)
        tree = GradientBoostingClassifier(max_depth=i)
        tree.fit(X_train, y_train)
        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)

        train_s.append(train_score)
        test_s.append(test_score)

        best_depth = i if test_score > highest_score else best_depth
        highest_score = test_score if test_score > highest_score else highest_score

    plt.plot(range(1, n_depth), train_s)
    plt.plot(range(1, n_depth), test_s)

    plt.legend(['训练集准确度', '测试集准确度'])

    plt.xticks(range(1, n_depth))
    plt.yticks(np.linspace(0, 1, 11))

    plt.title('梯度回升树不同复杂度准确情况')
    plt.savefig('梯度回升树不同复杂度准确情况.png', dpi=1000)
    plt.show()
    print('最佳深度为：{}'.format(best_depth))
    print('最好成绩为；{}'.format(highest_score))
    """
    最佳深度为：3
    最好成绩为；0.7020408163265306
    """


def draw_boosting_tree(max_depth=3):
    tree = GradientBoostingClassifier(max_depth=max_depth)
    tree.fit(X_train, y_train)
    names, importance = feature_names, tree.feature_importances_
    plt.figure(figsize=(10, 5))

    plt.barh(names, importance)
    plt.xlabel('比重')
    plt.title('梯度回升树各要素所占比重')
    plt.savefig('梯度回升树各要素所占比重.png')
    plt.show()
    return importance


if __name__ == '__main__':
    start_time = time.time()
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    gearbox_initial = []
    g = []

    feature_names = ['sensor1', 'sensor2', 'sensor3', 'sensor4']

    for i in range(5):
        gearbox_initial.append(pd.read_excel('附件1.xlsx', i, dtype=np.float32))
        gearbox_initial[i]['target'] = i if i ==0 or 1 else 1
        g.append(gearbox_initial[i].to_numpy())

    gearbox = np.array(g)

    X, y = np.vstack(gearbox[:, :, 1:5]), np.hstack(gearbox[:, :, 5])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    X_temp = np.vstack(gearbox[:,1:10,1:5])
    y_temp = np.hstack(gearbox[:,1:10,5])

    # use_dis_tree()
    # draw_dis()
    # get_feature_importance_of_tree()
    # random_forest_sketch_map()
    # train_s, test_s = use_random_forest()
    # draw_forest()
    # random_forest_sketch_map()
    use_boosting_tree()
    # importance = draw_boosting_tree()

    end_time = time.time()
    t = end_time-start_time
    print(t)