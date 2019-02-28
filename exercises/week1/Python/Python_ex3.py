# -*- coding: utf-8 -*-
"""
Computer intensive datahandling exercise 3

@author: dnor
"""

import numpy as np
import scipy.linalg as lng
import matplotlib.pyplot as plt

# If using as script and not from console;
path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture1\\S1"

if __name__ == "__main__":
    def ridgeMulti(X, _lambda, p, y):
        inner_prod = np.linalg.inv(np.matmul(X.T, X) + _lambda * np.eye(p,p))
        outer_prod = np.matmul(X.T, y)
        betas = np.matmul(inner_prod, outer_prod)
        return betas
        
    diabetPath = path + '\\DiabetesDataNormalized.txt'
    T = np.loadtxt(diabetPath, delimiter = ' ', skiprows = 1)
    y = T[:, 10]
    X = T[:,:10]
    
    [n, p] = np.shape(X)
    
    off = np.ones(n)
    M = np.c_[off, X] # Include offset / intercept
    
    # Linear solver
    beta_ols, res, rnk, s = lng.lstsq(M, y)
    
    k = 100; # try k values of lambda
    lambdas = np.logspace(-4, 3, k)
    
    betas = np.zeros((p,k))
    
    for i in range(k):
        betas[:, i] = ridgeMulti(X, lambdas[i], p, y)
        
    plt.figure()
    plt.semilogx(lambdas, betas.T )
    plt.xlabel("Lambdas")
    plt.ylabel("Betas")
    
    # Bias and variance of the ridge regression, same as exercise 2 - just for ridge
    n = 10 # nr of obs
    p = 3 # features / vars
    beta_true = np.array([1, 2, 3])
    X = np.random.randn(n, p) # random drawing, feature matrix
    
    m = 100 # nr of experiments
    
    betas2 = np.zeros((k, p, m)) # all variable estimates
    
    sigma = 0.2
    
    for i in range(k):
        for j in range(m):
            y = np.matmul(X, beta_true) + sigma * np.random.randn(n) # Measures response - true value plus noise level
            betas2[i, :, j] = ridgeMulti(X, lambdas[i], p, y)
        
    betas_mean = np.mean(betas2, axis = 2)

    plt.figure()
    plt.semilogx(lambdas, betas_mean)
    plt.xlabel("Lambdas")
    plt.ylabel("Betas")
    