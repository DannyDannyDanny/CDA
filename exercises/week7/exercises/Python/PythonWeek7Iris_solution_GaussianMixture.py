# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:26:17 2018

@author: dnor
"""
#%%
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

dataPath = "C:\\Users\\dnor\\Desktop\\02582\\Lecture7\\Exercises\\Data\\"

dataFrame = pd.read_csv(dataPath + "FisherIris.csv")
X = np.asarray(dataFrame.ix[:,0:4])

# Make class encoding as integers
le = preprocessing.LabelEncoder()
le.fit(["Setosa", "Versicolor", "Virginica"])
y = le.transform(dataFrame.ix[:, 4])

# Make scatter matrix
scatter_matrix(dataFrame)

k = 10
BIC = np.zeros((k))
# Investigate BIC
for k in range(1,k+1):
    # See http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
    gaussMix = GaussianMixture(n_components = k, covariance_type  = "full", max_iter = 1000, reg_covar = 0.01)
    BIC[k-1] = gaussMix.fit(X).bic(X)
    
plt.figure()
plt.plot(BIC)
plt.xlabel("Number of Gaussians")
plt.ylabel("BIC")
plt.title("Number of Gaussians vs BIC")