#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/18 22:25
# @Author  : hhy
# @FileName: test.py
# @Software: PyCharm

# h5f = h5py.File(args["index"], 'r')
# feats = h5f['dataset_1'][:]
# imgNames = h5f['dataset_2'][:]
#
# np.savetxt("feature.txt",start)
#
#
# scores=np.dot(start,feats.T)
# rank_ID=np.argsort(scores)[::-1]
# rank_score=scores[rank_ID]
#
# imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]

# import matplotlib.pyplot as plt
# plt.plot(1,1,"^r")
# plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

print("step 1: load data...")
dataSet = []
fileIn = open('./k_means.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

y_pred=KMeans(n_clusters=4,random_state=9).fit_predict(dataSet)
dataSet=np.array(dataSet)
plt.scatter(dataSet[:,0],dataSet[:,1],c=y_pred)
plt.show()