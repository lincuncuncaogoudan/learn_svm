#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/9 14:05
# @Author  : hhy
# @FileName: test_kmeans.py
# @Software: PyCharm

from numpy import *
import time
import matplotlib.pyplot as plt
from  k_means import *
from sklearn.cluster import KMeans

## step 1: load data
print("step 1: load data...")
dataSet = []
fileIn = open('./k_means.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: clustering...
print("step 2: clustering...")
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet, k)

## step 3: show the result
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment)


## step 4: Call sklearn library directly
# y_pred=KMeans(n_clusters=4,random_state=9).fit_predict(dataSet)
# dataSet=np.array(dataSet)
# plt.scatter(dataSet[:,0],dataSet[:,1],c=y_pred)
# plt.show()
## 聚类分数
# metrics.calinski_harabaz_score(dataSet, y_pred)


