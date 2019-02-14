# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:26:17 2018

For additional information concerning heirachical clustering in python, look at;
https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

@author: dnor
"""
#%%
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

dataPath = "C:\\Users\\dnor\\Desktop\\02582\\Lecture7\\Exercises\\Data\\"

responseLabels = np.loadtxt(dataPath + "ziplabel.csv")
X = np.loadtxt(dataPath + "zipdata.csv", delimiter = ",")

n, p = np.shape(X)

'''
for scipy dendogram look at;
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
'''
d_group = "ward"
N_leafs = 10

Z = linkage(X, d_group)

plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
den = dendrogram(
    Z,
    leaf_rotation=90.,
    leaf_font_size=8.,
    truncate_mode='lastp',
    p = N_leafs,
)
plt.show()