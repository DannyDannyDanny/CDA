# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:35:42 2018

@author: dnor
"""
#%%
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

#dataPath = "C:\\Users\\dnor\\Desktop\\02582\\Lecture7\\Exercises\\Data\\"
dataPath = "../Data/"

def plotting(X, kmeans, y_pred, ax):
    ax[0].clear()
    ax[1].clear()

    ax[0].scatter(X[:,0], X[:,1], c = y_pred[:,0], marker= ".")
    ax[1].scatter(X[:,0], X[:,1], c = y_pred[:,1], marker= ".")

    ax[0].set_title("Last iteration")
    ax[1].set_title("Current iteration")

pause = False
X = np.loadtxt(dataPath + "simulatedData.csv", delimiter = ",")

clusters = 8 # Control how many clusters we want

kmeans = [0] * 2
y_pred = np.zeros((np.size(X, axis = 0), 2))
fig, ax = plt.subplots(1,2)
plt.suptitle("Press anykey to advance plot, ctrl-C to kill loop")
ax[0].set_title("Last iteration")
ax[1].set_title("Current iteration")

kmeans[1] = KMeans(n_clusters = clusters, n_init = 1, random_state = 1, max_iter = 1)
y_pred[:,1] = kmeans[1].fit_predict(X)

ax[1].scatter(X[:,0], X[:,1], marker = '.', c = y_pred[:,1])

current_iter = 0
while(True):
    kmeans[0] = kmeans[1]
    y_pred[:,0] = y_pred[:,1]

    # Note, below is not a elegant solution, since we run through a lot of steps that are similar each time
    kmeans[1] = KMeans(n_clusters = clusters, init = "random", n_init = 1, random_state = 0, max_iter = 1 + current_iter)
    y_pred[:,1] = kmeans[1].fit_predict(X)

    plotting(X, kmeans, y_pred, ax)
    plt.draw()
    plt.waitforbuttonpress()
    current_iter += 1
    print(current_iter)
